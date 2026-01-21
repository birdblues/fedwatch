# scripts/pipeline.py
"""
Macro Regime Pipeline (As-Of / Release-Lag Aware + Stress Score + Backtest)

What this does
- Fetch macro indicators from FRED (+ optional yfinance fallbacks)
- Convert to month-end series
- Apply "as-of availability lags" per series (proxy for release timing; avoids look-ahead)
- Compute 4-regime label (Growth x Inflation quadrant)
- Compute stress_score / stress_driver / stress_flag using:
    - HY OAS, YC inversion, STLFSI4, VIX
- Backtest vs SPX (^GSPC) with monthly returns (signals applied next month)
- Upsert rows to Supabase (macro_regime)

Run examples
  uv run scripts/pipeline.py --start 2000-01-01 --end 2026-01-18 --print-latest
  uv run scripts/pipeline.py --start 2000-01-01 --print-backtest
  uv run scripts/pipeline.py --start 2000-01-01 --save-supabase
  uv run scripts/pipeline.py --start 2000-01-01 --save-supabase --print-backtest

Env vars
  FRED_API_KEY
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY   (or SUPABASE_API_KEY)
Optional
  SUPABASE_TABLE=macro_regime
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
    # Inflation (monthly)
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",

    # Growth / labor (monthly)
    "mfg_new_orders": "AMTMNO",
    "unrate": "UNRATE",

    # Rates / curve (daily)
    "dgs2": "DGS2",
    "dgs10": "DGS10",

    # Credit (daily; units depend on series; often bps level)
    "hy_oas": "BAMLH0A0HYM2",

    # USD / commodities (daily)
    "usd_broad": "DTWEXBGS",
    "wti": "DCOILWTICO",
    "gold": "GOLDAMGBD228NLBM",

    # Stress additions
    "stlfsi4": "STLFSI4",     # weekly
}

# Optional yfinance fallbacks (only used if FRED series missing/empty)
DEFAULT_YF_FALLBACKS = {
    "usd_broad": "DX-Y.NYB",  # DXY proxy
    "wti": "CL=F",
    "gold": "GC=F",
}

# Yahoo Finance tickers for market series used for stress/backtest
YF_MARKET = {
    "vix": "^VIX",
    "spx": "^GSPC",
}

USER_AGENT = "macro-regime-pipeline/2.0"

# "As-of availability lag" proxy (month units)
# At month-end t, you generally *do not* know that month's CPI/UNRATE/etc yet.
# So we use previous month's value (lag=1), or even lag=2 for slower releases.
ASOF_LAG_MONTHS = {
    "cpi": 1,
    "core_cpi": 1,
    "unrate": 1,
    "mfg_new_orders": 2,  # often comes later; proxy with 2-month lag
    # daily/weekly series are assumed known by month-end
    "dgs2": 0,
    "dgs10": 0,
    "hy_oas": 0,
    "usd_broad": 0,
    "wti": 0,
    "gold": 0,
    "stlfsi4": 0,
    "vix": 0,
    "spx": 0,
}

# Stress score weights (tuneable)
STRESS_WEIGHTS = {
    "hy_oas_z": 0.35,
    "vix_z": 0.35,
    "stlfsi4_z": 0.20,
    "yc_inv_z": 0.10,   # curve inversion component
}

# stress_flag threshold (tuneable)
STRESS_FLAG_THRESHOLD = 1.0


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


def month_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.resample("ME").last()


def pct_change_12m(index_level: pd.Series) -> pd.Series:
    return 100.0 * (index_level / index_level.shift(12) - 1.0)


def robust_zscore(s: pd.Series, window: int = 36, min_periods: int = 12) -> pd.Series:
    med = s.rolling(window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window, min_periods=min_periods).median()
    denom = (1.4826 * mad).replace(0, np.nan)
    return (s - med) / denom


def clamp01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def to_month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.to_datetime(idx).to_period("M").to_timestamp("M")


# -----------------------------
# FRED Fetch (no extra deps)
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
        realtime_start: Optional[str] = None,
        realtime_end: Optional[str] = None,
        max_retries: int = 4,
        backoff: float = 1.5,
    ) -> pd.Series:
        """
        Returns Series indexed by date, values float.
        If series does not exist, returns empty series.
        """
        url = f"{self.base_url}/series/observations"
        params = {"series_id": series_id, "file_type": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        if start is not None:
            params["observation_start"] = pd.to_datetime(start).strftime("%Y-%m-%d")
        if end is not None:
            params["observation_end"] = pd.to_datetime(end).strftime("%Y-%m-%d")
        if realtime_start:
            params["realtime_start"] = realtime_start
        if realtime_end:
            params["realtime_end"] = realtime_end

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
    )
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col]

    # yfinance may return MultiIndex columns
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = s.dropna()
    s.name = ticker
    return s


def fetch_yfinance_monthly(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s_daily = fetch_yfinance_daily(ticker, start, end)
    if s_daily.empty:
        return pd.Series(dtype=float, name=ticker)
    me = s_daily.resample("ME").last()
    me.name = ticker
    return me


# -----------------------------
# Feature build
# -----------------------------
def build_feature_dataframe(
    fred: FredClient,
    series_map: Dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    yf_fallbacks: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Fetch FRED series and return month-end dataframe.
    Also fetch VIX/SPX via yfinance month-end close (for stress/backtest).
    """
    yf_fallbacks = yf_fallbacks or {}
    data: Dict[str, pd.Series] = {}

    # 1) FRED fetch
    for key, sid in series_map.items():
        s = fred.fetch_series(sid, start=start, end=end)
        if s.empty:
            tkr = yf_fallbacks.get(key)
            if tkr:
                s2 = fetch_yfinance_monthly(tkr, start=start, end=end)
                if not s2.empty:
                    data[key] = s2
                    continue
        data[key] = s

    # 2) Add VIX + SPX from yfinance
    vix_me = fetch_yfinance_monthly(YF_MARKET["vix"], start=start, end=end)
    spx_me = fetch_yfinance_monthly(YF_MARKET["spx"], start=start, end=end)
    if not vix_me.empty:
        data["vix"] = vix_me
    if not spx_me.empty:
        data["spx"] = spx_me

    if not data:
        raise RuntimeError("No data fetched.")

    # concat (index union)
    df = pd.concat(data.values(), axis=1)
    df.columns = list(data.keys())
    df = df.sort_index()

    # month-end
    df = month_end(df)
    df = df.ffill()

    # trim to month-end range
    s_ts = start.tz_localize(None) if getattr(start, "tzinfo", None) else start
    e_ts = end.tz_localize(None) if getattr(end, "tzinfo", None) else end
    df = df[(df.index >= s_ts.to_period("M").to_timestamp("M")) & (df.index <= e_ts.to_period("M").to_timestamp("M"))]

    return df


def apply_asof_lags(df_me: pd.DataFrame, lags: Dict[str, int]) -> pd.DataFrame:
    """
    For each column, shift by lag months to represent "available as of month-end".
    Example: CPI lag=1 means at 2020-01-31 you only use CPI of 2019-12-31.
    """
    out = pd.DataFrame(index=df_me.index)
    for col in df_me.columns:
        lag = int(lags.get(col, 0))
        s = df_me[col]
        out[col] = s.shift(lag)
    return out


# -----------------------------
# Regime + Stress
# -----------------------------
@dataclasses.dataclass
class RegimeResult:
    df: pd.DataFrame


def compute_regime_and_stress(features_asof: pd.DataFrame) -> RegimeResult:
    """
    Compute:
    - regime_id / regime_label
    - stress_score / stress_driver / stress_flag
    - plus diagnostics columns used in prints/backtest
    """
    df = features_asof.copy().sort_index()
    df.index = to_month_end_index(df.index)

    # Inflation
    df["cpi_yoy"] = pct_change_12m(df["cpi"])
    df["core_cpi_yoy"] = pct_change_12m(df["core_cpi"])

    # Growth proxy
    df["orders_yoy"] = pct_change_12m(df["mfg_new_orders"])
    df["orders_mom"] = df["mfg_new_orders"].diff(3)
    df["unrate_chg_3m"] = df["unrate"].diff(3)

    # Rates / curve
    df["yc_10y2y"] = df["dgs10"] - df["dgs2"]

    # Credit
    df["hy_oas"] = df["hy_oas"]

    # Stress inputs
    df["stlfsi4"] = df.get("stlfsi4", np.nan)
    df["vix"] = df.get("vix", np.nan)

    # Z features for stress
    df["hy_oas_z"] = robust_zscore(df["hy_oas"])
    df["vix_z"] = robust_zscore(df["vix"])
    df["stlfsi4_z"] = robust_zscore(df["stlfsi4"])

    # Curve inversion proxy:
    # When yc_10y2y < 0 => stress. Convert to positive "more inverted = higher stress"
    # Use (-yc) then z-score. If yc positive, value is 0.
    yc_inv = (-df["yc_10y2y"]).clip(lower=0.0)
    df["yc_inv_z"] = robust_zscore(yc_inv)

    # Inflation state (as-of)
    core_med = df["core_cpi_yoy"].rolling(36, min_periods=18).median()
    df["infl_hot"] = (df["core_cpi_yoy"] > core_med) | (df["core_cpi_yoy"] >= 3.0)

    # Growth state (as-of)
    mom_med = df["orders_mom"].rolling(12, min_periods=6).median()
    df["growth_ok"] = (df["orders_yoy"] >= 0.0) & (df["orders_mom"] >= mom_med) & (df["unrate_chg_3m"] <= 0.2)

    # Quadrant regime
    def label_row(g_ok: bool, i_hot: bool) -> Tuple[int, str]:
        if g_ok and (not i_hot):
            return 1, "Goldilocks"
        if g_ok and i_hot:
            return 2, "Reflation"
        if (not g_ok) and i_hot:
            return 3, "Stagflation"
        return 4, "Recession"

    regime_id: List[int] = []
    regime_label: List[str] = []
    for _, r in df.iterrows():
        rid, rlab = label_row(bool(r.get("growth_ok")), bool(r.get("infl_hot")))
        regime_id.append(rid)
        regime_label.append(rlab)

    df["regime_id"] = regime_id
    df["regime_label"] = regime_label

    # Stress score (weighted sum of z's)
    # Note: robust_zscore can be NaN early; fill those with 0 for score.
    z_cols = ["hy_oas_z", "vix_z", "stlfsi4_z", "yc_inv_z"]
    z = df[z_cols].copy().fillna(0.0)

    stress_score = (
        STRESS_WEIGHTS["hy_oas_z"] * z["hy_oas_z"]
        + STRESS_WEIGHTS["vix_z"] * z["vix_z"]
        + STRESS_WEIGHTS["stlfsi4_z"] * z["stlfsi4_z"]
        + STRESS_WEIGHTS["yc_inv_z"] * z["yc_inv_z"]
    )
    df["stress_score"] = stress_score

    # stress_driver = largest weighted contributor
    contrib = pd.DataFrame(
        {
            "hy_oas": STRESS_WEIGHTS["hy_oas_z"] * z["hy_oas_z"],
            "vix": STRESS_WEIGHTS["vix_z"] * z["vix_z"],
            "stlfsi4": STRESS_WEIGHTS["stlfsi4_z"] * z["stlfsi4_z"],
            "yc_inv": STRESS_WEIGHTS["yc_inv_z"] * z["yc_inv_z"],
        },
        index=df.index,
    )

    def pick_driver(row: pd.Series) -> str:
        if row.isna().all():
            return "none"
        k = row.idxmax()
        v = row.loc[k]
        # if all contributions tiny -> none
        if not np.isfinite(v) or float(v) <= 0.10:
            return "none"
        return str(k)

    df["stress_driver"] = contrib.apply(pick_driver, axis=1)

    # stress_flag
    df["stress_flag"] = (df["stress_score"] >= STRESS_FLAG_THRESHOLD)

    # date column for convenience
    df["date"] = df.index.date

    return RegimeResult(df=df)


# -----------------------------
# Backtest
# -----------------------------
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_stats(r: pd.Series, equity: pd.Series) -> Dict[str, Any]:
    r = r.dropna()
    if len(r) == 0:
        return {}
    mean_1m = float(r.mean())
    med_1m = float(r.median())
    ann_ret = float((1.0 + mean_1m) ** 12 - 1.0)
    ann_vol = float(r.std(ddof=0) * np.sqrt(12))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan
    return {
        "n": int(len(r)),
        "mean_1m": mean_1m,
        "median_1m": med_1m,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_drawdown(equity),
        "worst_1m": float(r.min()),
        "best_1m": float(r.max()),
    }


def run_backtest(regime_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Backtest using SPX monthly returns from 'spx' month-end close included in features.
    Strategy: simple example:
      - base equity weight by regime
      - continuous risk cut by stress_score (optional)
      - apply signals next month (shift(1)) to avoid look-ahead
    """
    df = regime_df.copy().sort_index()
    if "spx" not in df.columns:
        raise RuntimeError("Backtest requires 'spx' month-end close column in regime_df.")
    spx_close = pd.to_numeric(df["spx"], errors="coerce")
    spx_ret_1m = spx_close.pct_change()
    spx_ret_1m.name = "spx_ret_1m"

    # align and drop first nan
    bt = pd.DataFrame(
        {
            "regime_id": df["regime_id"],
            "regime_label": df["regime_label"],
            "stress_flag": df["stress_flag"],
            "stress_score": df["stress_score"],
            "spx_ret_1m": spx_ret_1m,
        },
        index=df.index,
    )
    bt = bt.dropna(subset=["spx_ret_1m", "regime_id"])

    # base weights (example; tune)
    base_weight = {1: 0.70, 2: 0.80, 3: 0.50, 4: 0.40}
    base = bt["regime_id"].map(base_weight).astype(float).fillna(0.0)

    # continuous risk cut
    threshold = 1.0
    k = 0.50
    floor = 0.20
    cap = 1.00

    stress_score = pd.to_numeric(bt["stress_score"], errors="coerce").fillna(0.0)
    excess = (stress_score - threshold).clip(lower=0.0)
    risk_cut_factor = (1.0 - k * excess).clip(lower=floor, upper=cap)

    w_eq_raw = (base * risk_cut_factor).clip(lower=0.0, upper=1.0)

    # apply next month
    w_eq = w_eq_raw.shift(1).fillna(0.0).clip(0.0, 1.0)

    port_ret = w_eq * bt["spx_ret_1m"]
    bh_ret = bt["spx_ret_1m"]

    port_equity = (1.0 + port_ret).cumprod()
    bh_equity = (1.0 + bh_ret).cumprod()

    overall = perf_stats(port_ret, port_equity)
    bh = perf_stats(bh_ret, bh_equity)

    # by regime
    by_regime: Dict[str, Any] = {}
    for rid, g in bt.groupby("regime_id"):
        key = f"{int(rid)}_{g['regime_label'].iloc[0]}"
        r = (w_eq.loc[g.index] * g["spx_ret_1m"])
        eq = (1.0 + r).cumprod()
        by_regime[key] = perf_stats(r, eq)

    # by regime x stress
    by_regime_stress: Dict[str, Any] = {}
    for (rid, sf), g in bt.groupby(["regime_id", "stress_flag"]):
        tag = "stress" if bool(sf) else "normal"
        key = f"{int(rid)}_{g['regime_label'].iloc[0]}__{tag}"
        r = (w_eq.loc[g.index] * g["spx_ret_1m"])
        eq = (1.0 + r).cumprod()
        by_regime_stress[key] = perf_stats(r, eq)

    diag = {
        "stress_true_ratio": float(bt["stress_flag"].mean()),
        "start": str(bt.index.min().date()),
        "end": str(bt.index.max().date()),
        "n": int(len(bt)),
        "params": {"threshold": threshold, "k": k, "floor": floor, "cap": cap},
    }

    return {
        "overall": overall,
        "buyhold": bh,
        "diag": diag,
        "by_regime": by_regime,
        "by_regime_and_stress": by_regime_stress,
    }


# -----------------------------
# Supabase Write (REST / PostgREST)
# -----------------------------
class SupabaseWriter:
    def __init__(self, url: str, api_key: str, table: str = "macro_regime", schema: str = "public"):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.table = table
        self.schema = schema

    def upsert_rows(self, rows: List[Dict[str, Any]], on_conflict: str = "date", chunk: int = 500) -> None:
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
                raise RuntimeError(f"Supabase upsert failed HTTP {r.status_code}: {r.text[:1200]}")


def make_rows_for_supabase(regime_df: pd.DataFrame) -> List[Dict[str, Any]]:
    now_utc = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out: List[Dict[str, Any]] = []

    for idx, row in regime_df.iterrows():
        date_str = pd.to_datetime(idx).date().isoformat()
        if pd.isna(row.get("regime_id")) or pd.isna(row.get("regime_label")):
            continue

        out.append(
            {
                "date": date_str,
                "regime_id": int(row["regime_id"]),
                "regime_label": str(row["regime_label"]),
                "stress_flag": bool(row.get("stress_flag", False)),
                "stress_score": float(row.get("stress_score")) if pd.notna(row.get("stress_score")) else None,
                "stress_driver": str(row.get("stress_driver")) if pd.notna(row.get("stress_driver")) else None,
                "updated_at": now_utc,
            }
        )
    return out


# -----------------------------
# CLI / Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2000-01-01")
    p.add_argument("--end", type=str, default=None, help="default: today (UTC)")
    p.add_argument("--save-supabase", action="store_true", help="upsert to Supabase macro_regime")
    p.add_argument("--print-latest", action="store_true", help="print latest regime row")
    p.add_argument("--print-sample", action="store_true", help="print last 12 rows")
    p.add_argument("--print-backtest", action="store_true", help="run and print backtest summary")
    p.add_argument("--supabase-table", type=str, default=None)
    p.add_argument("--supabase-schema", type=str, default="public")
    return p.parse_args()


def main() -> int:
    # Optional .env load
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
    supa_key = env("SUPABASE_SERVICE_ROLE_KEY") or env("SUPABASE_API_KEY")
    supa_table = args.supabase_table or env("SUPABASE_TABLE", "macro_regime")

    fred = FredClient(api_key=fred_key)

    # 1) raw month-end features
    feat_raw = build_feature_dataframe(
        fred=fred,
        series_map=DEFAULT_FRED_SERIES,
        start=start,
        end=end,
        yf_fallbacks=DEFAULT_YF_FALLBACKS,
    )

    # 2) as-of lagging (release timing proxy)
    feat_asof = apply_asof_lags(feat_raw, ASOF_LAG_MONTHS)

    # 3) regime + stress
    result = compute_regime_and_stress(feat_asof)
    regime_df = result.df

    # 4) prints
    if args.print_latest:
        last = regime_df.dropna(subset=["regime_id"]).tail(1)
        if last.empty:
            print("No regime computed yet (insufficient data window).")
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
            cols = [c for c in cols if c in last.columns]
            print(last[cols])

    if args.print_sample:
        tail = regime_df.dropna(subset=["regime_id"]).tail(12)
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
        cols = [c for c in cols if c in tail.columns]
        print(tail[cols])

    if args.print_backtest:
        bt = run_backtest(regime_df)
        print("\n[Backtest] overall:", bt["overall"])
        print("[Backtest] buy&hold:", bt["buyhold"])
        print("[Backtest] diag:", bt["diag"])
        print("[Backtest] by_regime:")
        for k, v in bt["by_regime"].items():
            print(" ", k, v)
        print("[Backtest] by_regime_and_stress:")
        for k, v in bt["by_regime_and_stress"].items():
            print(" ", k, v)

    # 5) save to supabase
    if args.save_supabase:
        if not (supa_url and supa_key):
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_API_KEY) are required.")
        writer = SupabaseWriter(url=supa_url, api_key=supa_key, table=supa_table, schema=args.supabase_schema)
        rows = make_rows_for_supabase(regime_df)
        writer.upsert_rows(rows, on_conflict="date")
        print(f"Upserted {len(rows)} rows into {supa_table}.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as ex:
        eprint(f"[ERROR] {ex}")
        raise
