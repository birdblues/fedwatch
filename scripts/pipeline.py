# scripts/pipeline.py
"""
Macro Regime Pipeline (Regime-only to Supabase) + Stress 강화 + Regime Backtest

Adds:
- FRED: STLFSI4 (Financial Stress Index)
- yfinance: ^VIX, ^GSPC (SPX)
- Stress score/flag redesigned (OR + composite score)
- Monthly SPX return backtest by regime (+ stress conditioning)
- Optionally store summary table to Supabase (recommended: macro_regime_perf)

Run:
  uv run scripts/pipeline.py --start 2000-01-01 --print-latest --print-backtest
  uv run scripts/pipeline.py --start 2000-01-01 --save-supabase --save-backtest

Env:
  FRED_API_KEY
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_API_KEY)
Optional:
  SUPABASE_TABLE=macro_regime
  SUPABASE_PERF_TABLE=macro_regime_perf
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf  # optional
except Exception:
    yf = None


# -----------------------------
# Config
# -----------------------------
DEFAULT_FRED_SERIES = {
    # Inflation
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",

    # Growth / labor (PMI proxy)
    "mfg_new_orders": "AMTMNO",
    "unrate": "UNRATE",

    # Rates / curve
    "dgs2": "DGS2",
    "dgs10": "DGS10",

    # Credit
    "hy_oas": "BAMLH0A0HYM2",

    # USD
    "usd_broad": "DTWEXBGS",

    # Commodities
    "wti": "DCOILWTICO",
    "gold": "GOLDAMGBD228NLBM",

    # Stress (FRED)
    "stlfsi4": "STLFSI4",  # St. Louis Fed Financial Stress Index
}

DEFAULT_YF_FALLBACKS = {
    "usd_broad": "DX-Y.NYB",  # DXY proxy (not the same as DTWEXBGS)
    "wti": "CL=F",
    "gold": "GC=F",
}

# yfinance assets for stress & backtest
YF_ASSETS = {
    "vix": "^VIX",
    "spx": "^GSPC",
}

USER_AGENT = "macro-regime-pipeline/1.2"


# -----------------------------
# Helpers
# -----------------------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def safe_float(x) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip() in (".", ""):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def robust_zscore(s: pd.Series, window: int = 36, min_periods: int = 12) -> pd.Series:
    med = s.rolling(window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window, min_periods=min_periods).median()
    denom = (1.4826 * mad).replace(0, np.nan)
    return (s - med) / denom


def pct_change_12m(index_level: pd.Series) -> pd.Series:
    return 100.0 * (index_level / index_level.shift(12) - 1.0)


def month_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.resample("ME").last()


def to_month_end_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return month_end(s.to_frame()).iloc[:, 0]


def ann_return_from_monthly(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    # geometric annualized
    n = len(r)
    growth = float((1.0 + r).prod())
    return growth ** (12.0 / n) - 1.0


def ann_vol_from_monthly(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(12.0))


def max_drawdown_from_monthly(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


# -----------------------------
# FRED Fetch
# -----------------------------
class FredClient:
    def __init__(self, api_key: Optional[str], base_url: str = "https://api.stlouisfed.org/fred"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def fetch_series(
        self,
        series_id: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        max_retries: int = 4,
        backoff: float = 1.5,
    ) -> pd.Series:
        url = f"{self.base_url}/series/observations"
        params = {"series_id": series_id, "file_type": "json"}
        if self.api_key:
            params["api_key"] = self.api_key

        if start is not None:
            params["observation_start"] = pd.to_datetime(start).strftime("%Y-%m-%d")
        if end is not None:
            params["observation_end"] = pd.to_datetime(end).strftime("%Y-%m-%d")

        headers = {"User-Agent": USER_AGENT}

        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=30)

                if r.status_code == 400 and "series does not exist" in r.text.lower():
                    return pd.Series(dtype=float, name=series_id)

                if r.status_code >= 400:
                    raise RuntimeError(f"FRED HTTP {r.status_code}: {r.text[:300]}")

                payload = r.json()
                obs = payload.get("observations", [])
                if not obs:
                    return pd.Series(dtype=float, name=series_id)

                dates = [pd.to_datetime(o["date"]) for o in obs]
                vals = [safe_float(o.get("value")) for o in obs]
                s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=series_id).sort_index()
                s = s[~s.isna()]
                return s

            except Exception as ex:
                last_err = ex
                sleep_s = (backoff ** attempt) + np.random.rand() * 0.2
                time.sleep(float(sleep_s))

        raise RuntimeError(f"Failed to fetch FRED series {series_id}: {last_err}")


def fetch_yfinance_daily(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float, name=ticker)

    df = yf.download(
        ticker,
        start=str(start.date()),
        end=str((end + pd.Timedelta(days=1)).date()),
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)

    # choose close column
    if "Adj Close" in df.columns:
        px = df["Adj Close"]
    elif "Close" in df.columns:
        px = df["Close"]
    else:
        return pd.Series(dtype=float, name=ticker)

    # normalize to Series
    if isinstance(px, pd.DataFrame):
        if ticker in px.columns:
            s = px[ticker]
        else:
            s = px.iloc[:, 0]
    else:
        s = px

    s = s.dropna()
    s.name = ticker
    return s


def fetch_yfinance_monthly_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s = fetch_yfinance_daily(ticker, start, end)
    if s.empty:
        return pd.Series(dtype=float, name=ticker)
    return to_month_end_index(s)


# -----------------------------
# Regime Model
# -----------------------------
@dataclasses.dataclass
class RegimeResult:
    df: pd.DataFrame


def compute_regime(features: pd.DataFrame) -> RegimeResult:
    """
    4-regime (Growth x Inflation) + Stress score/flag

    Stress inputs:
      - hy_oas (FRED)
      - stlfsi4 (FRED)
      - vix (yfinance, monthly close)
    """
    df = features.copy().sort_index()

    # Ensure month-end index
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp("M")

    # Inflation
    df["cpi_yoy"] = pct_change_12m(df["cpi"])
    df["core_cpi_yoy"] = pct_change_12m(df["core_cpi"])

    # Growth (PMI proxy)
    df["orders_yoy"] = pct_change_12m(df["mfg_new_orders"])
    df["orders_mom"] = df["mfg_new_orders"].diff(3)
    df["unrate_chg_3m"] = df["unrate"].diff(3)

    # Curve
    df["yc_10y2y"] = df["dgs10"] - df["dgs2"]

    # Z-scores (stress candidates)
    df["hy_oas_z"] = robust_zscore(df["hy_oas"])
    df["stlfsi4_z"] = robust_zscore(df["stlfsi4"]) if "stlfsi4" in df.columns else np.nan

    # If vix exists
    if "vix" in df.columns:
        df["vix_z"] = robust_zscore(df["vix"])
    else:
        df["vix_z"] = np.nan

    # Inflation state
    core_med = df["core_cpi_yoy"].rolling(36, min_periods=18).median()
    df["infl_hot"] = (df["core_cpi_yoy"] > core_med) | (df["core_cpi_yoy"] >= 3.0)

    # Growth state (proxy rules)
    mom_med = df["orders_mom"].rolling(12, min_periods=6).median()
    df["growth_ok"] = (df["orders_yoy"] >= 0.0) & (df["orders_mom"] >= mom_med) & (df["unrate_chg_3m"] <= 0.2)

    # Regime quadrant
    def label_row(g_ok: bool, i_hot: bool) -> Tuple[int, str]:
        if g_ok and (not i_hot):
            return 1, "Goldilocks"
        if g_ok and i_hot:
            return 2, "Reflation"
        if (not g_ok) and i_hot:
            return 3, "Stagflation"
        return 4, "Recession"

    out = df.copy()
    ids: List[int] = []
    labels: List[str] = []
    for _, row in out.iterrows():
        rid, rlab = label_row(bool(row.get("growth_ok")), bool(row.get("infl_hot")))
        ids.append(rid)
        labels.append(rlab)

    out["regime_id"] = ids
    out["regime_label"] = labels

    # -----------------------------
    # Stress score/flag (redefined)
    # -----------------------------
    # Core idea:
    # - A single strong stress signal should be able to trigger stress_flag
    # - Keep curve inversion as its own binary risk signal (slower, macro warning)
    #
    # stress_score = max(hy_oas_z, stlfsi4_z, vix_z)
    # stress_flag  = (stress_score > 1.0) OR (yc_10y2y < 0)
    #
    out["stress_score"] = pd.concat(
        [
            out["hy_oas_z"].rename("hy"),
            out["stlfsi4_z"].rename("fs"),
            out["vix_z"].rename("vx"),
        ],
        axis=1,
    ).max(axis=1)

    out["stress_flag"] = (out["stress_score"] > 1.0) | (out["yc_10y2y"] < 0)

    # Optional: classify stress driver (for debugging/dashboard)
    def stress_driver(row) -> str:
        if pd.isna(row.get("stress_score")):
            return "none"
        parts = {
            "hy": row.get("hy_oas_z", np.nan),
            "fs": row.get("stlfsi4_z", np.nan),
            "vx": row.get("vix_z", np.nan),
        }
        # curve inversion tag if present
        inv = bool(row.get("yc_10y2y", 1.0) < 0)
        k = max(parts, key=lambda kk: (-np.inf if pd.isna(parts[kk]) else parts[kk]))
        driver = {"hy": "credit", "fs": "fsi", "vx": "vix"}[k]
        if inv:
            return f"{driver}+curve" if (row.get("stress_score", -np.inf) > 1.0) else "curve"
        return driver if (row.get("stress_score", -np.inf) > 1.0) else "none"

    out["stress_driver"] = out.apply(stress_driver, axis=1)

    out["date"] = out.index.date
    return RegimeResult(df=out)


# -----------------------------
# Supabase Write
# -----------------------------
class SupabaseWriter:
    def __init__(self, url: str, api_key: str, table: str, schema: str = "public"):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.table = table
        self.schema = schema

    def upsert_rows(self, rows: List[Dict[str, Any]], on_conflict: str, chunk: int = 500) -> None:
        if not rows:
            return

        endpoint = f"{self.url}/rest/v1/{self.table}"
        headers = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        params = {"on_conflict": on_conflict}

        for i in range(0, len(rows), chunk):
            batch = rows[i : i + chunk]
            r = requests.post(endpoint, headers=headers, params=params, json=batch, timeout=30)
            if r.status_code >= 400:
                raise RuntimeError(f"Supabase upsert failed HTTP {r.status_code}: {r.text[:800]}")


# -----------------------------
# Data assembly
# -----------------------------
def build_feature_dataframe(
    fred: FredClient,
    series_map: Dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    yf_fallbacks: Optional[Dict[str, str]] = None,
    yf_assets: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Returns monthly dataframe with:
      cpi, core_cpi, mfg_new_orders, unrate, dgs2, dgs10, hy_oas, usd_broad, wti, gold, stlfsi4, vix, spx(optional)
    """
    yf_fallbacks = yf_fallbacks or {}
    yf_assets = yf_assets or {}

    data: Dict[str, pd.Series] = {}

    # FRED series
    for key, sid in series_map.items():
        s = fred.fetch_series(sid, start=start, end=end)
        if s.empty:
            tkr = yf_fallbacks.get(key)
            if tkr:
                s2 = fetch_yfinance_monthly_close(tkr, start=start, end=end)
                if not s2.empty:
                    data[key] = s2
                    continue
        data[key] = s

    # yfinance assets (monthly close)
    for key, tkr in yf_assets.items():
        s = fetch_yfinance_monthly_close(tkr, start=start, end=end)
        if not s.empty:
            data[key] = s

    df = pd.concat(data.values(), axis=1)
    df.columns = list(data.keys())
    df = df.sort_index()
    df = month_end(df)
    df = df.ffill()

    # Trim
    s_ts = start.tz_localize(None) if start.tzinfo else start
    e_ts = end.tz_localize(None) if end.tzinfo else end
    df = df[(df.index >= s_ts.to_period("M").to_timestamp("M")) & (df.index <= e_ts.to_period("M").to_timestamp("M"))]

    # Validate minimal required for regime
    required = ["cpi", "core_cpi", "mfg_new_orders", "unrate", "dgs2", "dgs10", "hy_oas", "stlfsi4"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after fetch: {missing}")

    empty_cols = [c for c in required if df[c].dropna().empty]
    if empty_cols:
        raise RuntimeError(f"Required columns fetched but empty: {empty_cols}")

    return df


# -----------------------------
# Backtest (SPX monthly return by regime/stress)
# -----------------------------
def build_spx_monthly_returns(df_feat: pd.DataFrame) -> pd.Series:
    """
    Uses df_feat['spx'] monthly close to compute monthly returns.
    If missing, raises.
    """
    if "spx" not in df_feat.columns:
        raise RuntimeError("SPX (^GSPC) monthly close not found. Ensure yfinance is available and ticker fetched.")
    px = df_feat["spx"].dropna()
    r = px.pct_change().rename("spx_ret")
    return r


def summarize_returns(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {
            "n": 0,
            "mean_1m": np.nan,
            "median_1m": np.nan,
            "ann_ret": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
            "worst_1m": np.nan,
            "best_1m": np.nan,
        }
    ann_ret = ann_return_from_monthly(r)
    ann_vol = ann_vol_from_monthly(r)
    sharpe = ann_ret / ann_vol if (ann_vol and not np.isnan(ann_vol) and ann_vol > 0) else np.nan
    return {
        "n": int(r.shape[0]),
        "mean_1m": float(r.mean()),
        "median_1m": float(r.median()),
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
        "max_dd": float(max_drawdown_from_monthly(r)),
        "worst_1m": float(r.min()),
        "best_1m": float(r.max()),
    }


def regime_backtest_table(regime_df: pd.DataFrame, spx_ret: pd.Series) -> Dict[str, Any]:
    """
    Returns a JSON-serializable dict with:
      - overall
      - by_regime
      - by_regime_and_stress
    """
    x = regime_df.copy()
    x = x.join(spx_ret, how="left")
    x = x.dropna(subset=["spx_ret"])

    overall = summarize_returns(x["spx_ret"])

    by_regime: Dict[str, Any] = {}
    for rid, g in x.groupby("regime_id"):
        lbl = str(g["regime_label"].iloc[0])
        key = f"{int(rid)}_{lbl}"
        by_regime[key] = summarize_returns(g["spx_ret"])

    by_regime_stress: Dict[str, Any] = {}
    for (rid, stress), g in x.groupby(["regime_id", "stress_flag"]):
        lbl = str(g["regime_label"].iloc[0])
        skey = "stress" if bool(stress) else "normal"
        key = f"{int(rid)}_{lbl}__{skey}"
        by_regime_stress[key] = summarize_returns(g["spx_ret"])

    # add some simple diagnostics
    diag = {
        "stress_true_ratio": float(x["stress_flag"].mean()) if len(x) > 0 else np.nan,
        "start": str(x.index.min().date()) if len(x) > 0 else None,
        "end": str(x.index.max().date()) if len(x) > 0 else None,
        "n": int(len(x)),
    }

    return {
        "overall": overall,
        "by_regime": by_regime,
        "by_regime_and_stress": by_regime_stress,
        "diag": diag,
    }


# -----------------------------
# Supabase payload builders
# -----------------------------
from typing import Any, Dict, List
import pandas as pd
import datetime as dt

def make_rows_for_supabase_regime(regime_df: pd.DataFrame) -> List[Dict[str, Any]]:
    now_utc = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out: List[Dict[str, Any]] = []

    for idx, row in regime_df.iterrows():
        date_str = pd.to_datetime(idx).date().isoformat()
        if pd.isna(row.get("regime_id")) or pd.isna(row.get("regime_label")):
            continue

        # --- add these safely ---
        stress_score = row.get("stress_score", None)
        if pd.isna(stress_score):
            stress_score = None

        stress_driver = row.get("stress_driver", None)
        # NaN 처리
        if isinstance(stress_driver, float) and pd.isna(stress_driver):
            stress_driver = None

        out.append(
            {
                "date": date_str,
                "regime_id": int(row["regime_id"]),
                "regime_label": str(row["regime_label"]),
                "stress_flag": bool(row.get("stress_flag", False)),
                "stress_score": stress_score,      # <-- added
                "stress_driver": stress_driver,    # <-- added (optional)
                "updated_at": now_utc,
            }
        )
    return out



def make_row_for_supabase_perf(backtest_metrics: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    now_utc = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "asof_date": pd.Timestamp.utcnow().date().isoformat(),
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "n_months": int(backtest_metrics.get("diag", {}).get("n", 0)),
        "metrics": backtest_metrics,
        "updated_at": now_utc,
    }


# -----------------------------
# CLI / Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2000-01-01")
    p.add_argument("--end", type=str, default=None, help="default: today (UTC)")
    p.add_argument("--save-supabase", action="store_true", help="upsert macro_regime (regime only)")
    p.add_argument("--save-backtest", action="store_true", help="upsert macro_regime_perf (summary)")
    p.add_argument("--print-latest", action="store_true", help="print latest regime row")
    p.add_argument("--print-sample", action="store_true", help="print last 12 rows")
    p.add_argument("--print-backtest", action="store_true", help="print backtest summary")
    p.add_argument("--supabase-table", type=str, default=None)
    p.add_argument("--supabase-perf-table", type=str, default=None)
    p.add_argument("--supabase-schema", type=str, default="public")
    return p.parse_args()


def main() -> int:
    try:
        import dotenv  # type: ignore
        dotenv.load_dotenv()
    except Exception:
        pass

    args = parse_args()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else pd.Timestamp.utcnow().normalize()

    fred_key = env("FRED_API_KEY")
    if not fred_key:
        raise RuntimeError("FRED_API_KEY is required.")

    supa_url = env("SUPABASE_URL")
    supa_key = env("SUPABASE_SERVICE_ROLE_KEY") or env("SUPABASE_KEY")

    supa_table = args.supabase_table or env("SUPABASE_TABLE", "macro_regime")
    supa_perf_table = args.supabase_perf_table or env("SUPABASE_PERF_TABLE", "macro_regime_perf")

    fred = FredClient(api_key=fred_key)

    # Features (monthly)
    feat = build_feature_dataframe(
        fred=fred,
        series_map=DEFAULT_FRED_SERIES,
        start=start,
        end=end,
        yf_fallbacks=DEFAULT_YF_FALLBACKS,
        yf_assets=YF_ASSETS,  # adds vix, spx monthly
    )

    # Compute regime + stress
    result = compute_regime(feat)
    regime_df = result.df

    # Backtest (SPX monthly returns)
    spx_ret = build_spx_monthly_returns(feat)
    bt = regime_backtest_table(regime_df, spx_ret)

    # Print outputs
    if args.print_latest:
        last = regime_df.dropna(subset=["regime_id"]).tail(1)
        if last.empty:
            print("No regime computed yet.")
        else:
            cols = [
                "regime_id",
                "regime_label",
                "stress_flag",
                "stress_score",
                "stress_driver",
                "core_cpi_yoy",
                "orders_yoy",
                "unrate_chg_3m",
                "yc_10y2y",
                "hy_oas",
                "stlfsi4",
                "vix",
            ]
            print(last[[c for c in cols if c in last.columns]])

    if args.print_sample:
        tail = regime_df.dropna(subset=["regime_id"]).tail(12)
        cols = ["regime_id", "regime_label", "stress_flag", "stress_score", "stress_driver", "yc_10y2y", "hy_oas", "stlfsi4", "vix"]
        print(tail[[c for c in cols if c in tail.columns]])

    if args.print_backtest:
        # show concise console table
        print("\n[Backtest] overall:", bt["overall"])
        print("[Backtest] diag:", bt["diag"])
        print("[Backtest] by_regime:")
        for k, v in bt["by_regime"].items():
            print(" ", k, v)
        print("[Backtest] by_regime_and_stress:")
        for k, v in bt["by_regime_and_stress"].items():
            print(" ", k, v)

    # Save to Supabase
    if args.save_supabase or args.save_backtest:
        if not (supa_url and supa_key):
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_API_KEY) are required.")

    if args.save_supabase:
        writer = SupabaseWriter(url=supa_url, api_key=supa_key, table=supa_table, schema=args.supabase_schema)
        rows = make_rows_for_supabase_regime(regime_df)
        writer.upsert_rows(rows, on_conflict="date")
        print(f"Upserted {len(rows)} rows into {supa_table}.")

    if args.save_backtest:
        writer2 = SupabaseWriter(url=supa_url, api_key=supa_key, table=supa_perf_table, schema=args.supabase_schema)
        row = make_row_for_supabase_perf(bt, start=start, end=end)
        writer2.upsert_rows([row], on_conflict="asof_date")
        print(f"Upserted 1 row into {supa_perf_table}.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as ex:
        eprint(f"[ERROR] {ex}")
        raise
