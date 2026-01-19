# scripts/pipeline.py
"""
Macro Regime Pipeline (Regime-only to Supabase)

- Fetch macro indicators from FRED (+ optional yfinance fallbacks)
- Compute 4-regime label (Growth x Inflation quadrant) + optional diagnostics
- Save ONLY regime rows to Supabase (upsert)

Run (local):
  python scripts/pipeline.py --start 2000-01-01 --end 2026-01-18 --print-latest
  python scripts/pipeline.py --start 2000-01-01 --save-supabase

GitHub Actions env vars (recommended):
  FRED_API_KEY
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY   (or SUPABASE_API_KEY)
Optional:
  SUPABASE_TABLE=macro_regime
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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
    "cpi": "CPIAUCSL",                 # CPI (SA, index)
    "core_cpi": "CPILFESL",            # CPI less food & energy (SA, index)

    # Growth / labor  (PMI is removed from FRED; use proxy)
    "mfg_new_orders": "AMTMNO",        # Manufacturers' New Orders: Total Manufacturing
    "unrate": "UNRATE",                # unemployment rate

    # Rates / curve
    "dgs2": "DGS2",                    # 2Y treasury
    "dgs10": "DGS10",                  # 10Y treasury

    # Credit
    "hy_oas": "BAMLH0A0HYM2",          # ICE BofA US High Yield OAS (bps)

    # USD
    "usd_broad": "DTWEXBGS",           # Trade Weighted U.S. Dollar Index: Broad (goods)

    # Commodities
    "wti": "DCOILWTICO",               # WTI spot (daily)
    "gold": "GOLDAMGBD228NLBM",        # Gold price (daily)
}

# Optional yfinance fallbacks (only used if FRED series missing/empty)
DEFAULT_YF_FALLBACKS = {
    "usd_broad": "DX-Y.NYB",           # DXY proxy (not the same as DTWEXBGS)
    "wti": "CL=F",
    "gold": "GC=F",
}

USER_AGENT = "macro-regime-pipeline/1.1"


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
    denom = (1.4826 * mad).replace(0, np.nan)  # 1.4826*MAD ~ std
    return (s - med) / denom


def pct_change_12m(index_level: pd.Series) -> pd.Series:
    return 100.0 * (index_level / index_level.shift(12) - 1.0)


def month_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.resample("ME").last()


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
        Returns a Series indexed by date (UTC-naive), values float.
        If series does not exist (HTTP 400 / 'series does not exist'), returns empty Series.
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


def fetch_yfinance_monthly(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
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
    s = df[col].copy()

    # If yfinance returns a DataFrame (e.g. multi-index columns), handle it
    if isinstance(s, pd.DataFrame):
        s_df = s
    else:
        s.name = ticker
        s_df = s.to_frame()

    s_df = s_df[~s_df.isna().all(axis=1)]  # Drop rows where all cols are NaN

    # Resample
    me_df = month_end(s_df)

    # Extract single series
    if ticker in me_df.columns:
        return me_df[ticker]
    
    if len(me_df.columns) > 0:
        return me_df.iloc[:, 0]
        
    return pd.Series(dtype=float, name=ticker)


# -----------------------------
# Regime Model
# -----------------------------
@dataclasses.dataclass
class RegimeResult:
    df: pd.DataFrame


def compute_regime(features: pd.DataFrame) -> RegimeResult:
    """
    4-regime (Growth x Inflation quadrant)

    Growth proxy:
      - Manufacturing New Orders (AMTMNO): YoY + 3m momentum
      - Unemployment 3m change

    Inflation proxy:
      - Core CPI YoY relative to rolling median + soft absolute threshold

    Diagnostics:
      - Curve (10y-2y)
      - Credit stress (HY OAS)
      - USD, Oil, Gold z-scores (optional)
    """
    df = features.copy().sort_index()

    # Ensure month-end index
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp("M")

    # Inflation
    df["cpi_yoy"] = pct_change_12m(df["cpi"])
    df["core_cpi_yoy"] = pct_change_12m(df["core_cpi"])

    # Growth (proxy replacing PMI)
    df["orders_yoy"] = pct_change_12m(df["mfg_new_orders"])
    df["orders_mom"] = df["mfg_new_orders"].diff(3)  # 3m change
    df["unrate_chg_3m"] = df["unrate"].diff(3)

    # Rates / curve
    df["yc_10y2y"] = df["dgs10"] - df["dgs2"]

    # Credit
    df["hy_oas"] = df["hy_oas"]

    # Commodities / USD (z)
    for col in ["usd_broad", "wti", "gold"]:
        if col in df.columns:
            df[f"{col}_z"] = robust_zscore(df[col])

    # Inflation state
    core_med = df["core_cpi_yoy"].rolling(36, min_periods=18).median()
    df["infl_hot"] = (df["core_cpi_yoy"] > core_med) | (df["core_cpi_yoy"] >= 3.0)

    # Growth state (proxy rules)
    mom_med = df["orders_mom"].rolling(12, min_periods=6).median()
    df["growth_ok"] = (df["orders_yoy"] >= 0.0) & (df["orders_mom"] >= mom_med) & (df["unrate_chg_3m"] <= 0.2)

    # Quadrant regime:
    # 1: Goldilocks (G+, I-)
    # 2: Reflation  (G+, I+)
    # 3: Stagflation(G-, I+)
    # 4: Recession  (G-, I-)
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

    # Stress tag (for 4번 세분화 베이스)
    out["stress_flag"] = (out["yc_10y2y"] < 0) & (robust_zscore(out["hy_oas"]) > 1.0)

    out["date"] = out.index.date
    return RegimeResult(df=out)


# -----------------------------
# Supabase Write (REST / PostgREST)
# -----------------------------
class SupabaseWriter:
    def __init__(self, url: str, api_key: str, table: str = "macro_regime", schema: str = "public"):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.table = table
        self.schema = schema

    def upsert_rows(self, rows: List[Dict], on_conflict: str = "date", chunk: int = 500) -> None:
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
                raise RuntimeError(f"Supabase upsert failed HTTP {r.status_code}: {r.text[:500]}")


# -----------------------------
# Main pipeline
# -----------------------------
def build_feature_dataframe(
    fred: FredClient,
    series_map: Dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    yf_fallbacks: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Fetch required series and return a monthly dataframe with canonical columns:
      cpi, core_cpi, mfg_new_orders, unrate, dgs2, dgs10, hy_oas, usd_broad, wti, gold
    """
    yf_fallbacks = yf_fallbacks or {}
    data: Dict[str, pd.Series] = {}

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

    df = pd.concat(data.values(), axis=1)
    df.columns = list(data.keys())
    df = df.sort_index()
    df = month_end(df)
    df = df.ffill()

    # Trim
    # Ensure start/end are tz-naive before period conversion to silence warning
    s_ts = start.tz_localize(None) if start.tzinfo else start
    e_ts = end.tz_localize(None) if end.tzinfo else end
    
    df = df[(df.index >= s_ts.to_period("M").to_timestamp("M")) & (df.index <= e_ts.to_period("M").to_timestamp("M"))]

    required = ["cpi", "core_cpi", "mfg_new_orders", "unrate", "dgs2", "dgs10", "hy_oas", "usd_broad", "wti", "gold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after fetch: {missing}")

    empty_cols = [c for c in required if df[c].dropna().empty]
    if empty_cols:
        raise RuntimeError(
            f"Required columns fetched but empty (no data). Columns={empty_cols}. "
            f"Check FRED series IDs / date range."
        )

    return df


def make_rows_for_supabase(regime_df: pd.DataFrame) -> List[Dict]:
    now_utc = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out: List[Dict] = []

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
                "updated_at": now_utc,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2000-01-01")
    p.add_argument("--end", type=str, default=None, help="default: today (UTC)")
    p.add_argument("--save-supabase", action="store_true", help="upsert to Supabase")
    p.add_argument("--print-latest", action="store_true", help="print latest regime row")
    p.add_argument("--print-sample", action="store_true", help="print last 12 rows")
    p.add_argument("--supabase-table", type=str, default=None)
    p.add_argument("--supabase-schema", type=str, default="public")
    return p.parse_args()


def main() -> int:
    # Optional .env load (won't crash if python-dotenv not installed)
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

    feat = build_feature_dataframe(
        fred=fred,
        series_map=DEFAULT_FRED_SERIES,
        start=start,
        end=end,
        yf_fallbacks=DEFAULT_YF_FALLBACKS,
    )

    result = compute_regime(feat)
    regime_df = result.df

    if args.print_latest:
        last = regime_df.dropna(subset=["regime_id"]).tail(1)
        if last.empty:
            print("No regime computed yet (insufficient data window).")
        else:
            print(
                last[
                    [
                        "regime_id",
                        "regime_label",
                        "stress_flag",
                        "core_cpi_yoy",
                        "orders_yoy",
                        "unrate_chg_3m",
                        "yc_10y2y",
                        "hy_oas",
                    ]
                ]
            )

    if args.print_sample:
        tail = regime_df.dropna(subset=["regime_id"]).tail(12)
        print(
            tail[
                [
                    "regime_id",
                    "regime_label",
                    "stress_flag",
                    "core_cpi_yoy",
                    "orders_yoy",
                    "unrate_chg_3m",
                    "yc_10y2y",
                    "hy_oas",
                ]
            ]
        )

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