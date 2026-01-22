# scripts/pipeline_daily.py
"""
Macro Regime Pipeline (Option C: Release-time aware DAILY state machine)

- Monthly macro indicators are applied from their FRED "realtime_start" (availability date) -> daily forward-fill
- Daily market stress is computed daily (SPX/VIX + HY OAS + STLFSI4)
- Daily state is saved to Supabase (upsert by date)

Run:
  python scripts/pipeline_daily.py --start 2000-01-01 --end 2026-01-18 --print-latest
  python scripts/pipeline_daily.py --start 2000-01-01 --save-supabase

Env:
  FRED_API_KEY
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY   (or SUPABASE_API_KEY)
Optional:
  SUPABASE_TABLE=macro_state_daily
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# Config
# -----------------------------
DEFAULT_FRED_SERIES = {
    # Inflation (monthly)
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    # Growth proxy (monthly)
    "mfg_new_orders": "AMTMNO",
    "unrate": "UNRATE",
    # Rates/credit (daily)
    "dgs2": "DGS2",
    "dgs10": "DGS10",
    "hy_oas": "BAMLH0A0HYM2",
    # Weekly stress index
    "stlfsi4": "STLFSI4",
}

YF_TICKERS = {
    "spx": "^GSPC",
    "vix": "^VIX",
}

USER_AGENT = "macro-regime-pipeline/daily-1.0"


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


# ✅ pandas datetime64[ns] safe upper bound (prevents 9999-12-31 crash)
PANDAS_TS_MAX_SAFE = pd.Timestamp("2262-04-11")


def safe_fred_ts(x) -> pd.Timestamp:
    """
    FRED realtime_start/end can be '9999-12-31' etc.
    Pandas datetime64[ns] cannot represent years > 2262.
    We cap anything above 2262-04-11 to PANDAS_TS_MAX_SAFE.
    """
    if x is None:
        return pd.NaT

    if isinstance(x, pd.Timestamp):
        # ensure tz-naive comparison
        ts = x.tz_localize(None) if getattr(x, "tzinfo", None) else x
        return ts if ts <= PANDAS_TS_MAX_SAFE else PANDAS_TS_MAX_SAFE

    s = str(x)

    # quick year check to avoid out-of-bounds conversion
    try:
        y = int(s[:4])
        if y > 2262:
            return PANDAS_TS_MAX_SAFE
    except Exception:
        pass

    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    ts = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
    return ts if ts <= PANDAS_TS_MAX_SAFE else PANDAS_TS_MAX_SAFE


def robust_zscore(s: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    med = s.rolling(window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window, min_periods=min_periods).median()
    denom = (1.4826 * mad).replace(0, np.nan)
    return (s - med) / denom


def pct_change_12m_monthly(level: pd.Series) -> pd.Series:
    return 100.0 * (level / level.shift(12) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_stats_from_returns(r: pd.Series, freq: str = "D") -> Dict[str, Any]:
    r = r.dropna()
    if r.empty:
        return {}
    n = int(len(r))
    mean = float(r.mean())
    med = float(r.median())

    if freq == "D":
        ann_ret = float((1.0 + mean) ** 252 - 1.0)
        ann_vol = float(r.std(ddof=0) * np.sqrt(252))
    else:  # monthly
        ann_ret = float((1.0 + mean) ** 12 - 1.0)
        ann_vol = float(r.std(ddof=0) * np.sqrt(12))

    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan
    eq = (1.0 + r).cumprod()
    return {
        "n": n,
        "mean": mean,
        "median": med,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_drawdown(eq),
        "worst": float(r.min()),
        "best": float(r.max()),
    }


# -----------------------------
# FRED Client (realtime-aware)
# -----------------------------
class FredClient:
    def __init__(self, api_key: str, base_url: str = "https://api.stlouisfed.org/fred"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def fetch_observations(
        self,
        series_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        output_type: Optional[int] = None,
        realtime_start: Optional[str] = None,
        realtime_end: Optional[str] = None,
        max_retries: int = 4,
        backoff: float = 1.6,
    ) -> pd.DataFrame:
        """
        Returns DF with columns:
          - obs_date (datetime)
          - value (float)
          - realtime_start (datetime)
          - realtime_end (datetime)

        ✅ 핵심:
        - FRED는 realtime_end가 '오늘'보다 미래면 안 되고, 예외적으로 '9999-12-31'만 허용.
        - pandas는 9999-12-31을 timestamp로 못 읽으므로 "요청은 9999", "파싱은 safe_fred_ts로 캡" 전략.
        """
        url = f"{self.base_url}/series/observations"
        params: Dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": pd.to_datetime(start).strftime("%Y-%m-%d"),
            "observation_end": pd.to_datetime(end).strftime("%Y-%m-%d"),
        }

        if output_type is not None:
            ot = int(output_type)
            params["output_type"] = ot

            # output_type=2/3/4 (vintage 관련) 은 realtime 범위 필요
            if ot in (2, 3, 4):
                params["realtime_start"] = realtime_start or "1776-07-04"
                # FRED 규칙상 미래면 안 되는데 '9999-12-31'만 허용이므로 기본은 항상 9999.
                params["realtime_end"] = realtime_end or "9999-12-31"
            else:
                # ot=1 등은 realtime 범위 굳이 안 줌 (caller가 주면 그대로 전달)
                if realtime_start is not None:
                    params["realtime_start"] = realtime_start
                if realtime_end is not None:
                    params["realtime_end"] = realtime_end
        else:
            if realtime_start is not None:
                params["realtime_start"] = realtime_start
            if realtime_end is not None:
                params["realtime_end"] = realtime_end

        headers = {"User-Agent": USER_AGENT}

        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=30)
                if r.status_code == 400 and "series does not exist" in r.text.lower():
                    return pd.DataFrame(columns=["obs_date", "value", "realtime_start", "realtime_end"])
                if r.status_code >= 400:
                    raise RuntimeError(f"FRED HTTP {r.status_code}: {r.text[:300]}")

                payload = r.json()
                obs = payload.get("observations", [])
                if not obs:
                    return pd.DataFrame(columns=["obs_date", "value", "realtime_start", "realtime_end"])

                rows = []
                for o in obs:
                    rows.append(
                        {
                            "obs_date": pd.to_datetime(o["date"]),
                            "value": safe_float(o.get("value")),
                            "realtime_start": safe_fred_ts(o.get("realtime_start")),
                            "realtime_end": safe_fred_ts(o.get("realtime_end")),
                        }
                    )
                df = pd.DataFrame(rows).dropna(subset=["obs_date"])
                df = df.sort_values(["obs_date", "realtime_start"]).reset_index(drop=True)
                df = df[df["value"].notna()]
                return df
            except Exception as ex:
                last_err = ex
                time.sleep(float((backoff ** attempt) + np.random.rand() * 0.2))

        raise RuntimeError(f"Failed to fetch FRED observations {series_id}: {last_err}")

    def asof_daily_from_realtime(
        self,
        obs_df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        effective_col: str = "realtime_start",
        date_col: str = "obs_date",
        value_col: str = "value",
        freq: str = "B",
    ) -> pd.Series:
        """
        Convert a (realtime) observations dataframe into a daily asof series.

        Rule:
        - effective_col(기본 realtime_start) 있으면 그 날짜부터 value가 "사용 가능"하다고 보고 asof merge.
        - effective_col 없으면 date_col(기본 obs_date) 사용
        - 그것도 없으면 index 사용

        ✅ FIX: index fallback 시 reset_index()가 만들어주는 컬럼명이
        'index'가 아닐 수 있음(예: index.name='obs_date') → '__idx' rename이 실패하던 버그 수정
        """
        if obs_df is None or len(obs_df) == 0:
            idx = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq=freq)
            return pd.Series(index=idx, dtype=float, name=value_col)

        out = obs_df.copy()

        # ---- effective date column 결정 ----
        if effective_col in out.columns:
            eff_col = effective_col
        elif date_col in out.columns:
            eff_col = date_col
        else:
            # index fallback: reset_index가 만든 "index 컬럼" 이름을 찾아서 '__idx'로 통일
            out2 = out.reset_index()

            # reset_index로 추가된 컬럼(=기존 index 컬럼들) 찾기
            added_cols = [c for c in out2.columns if c not in out.columns]
            if not added_cols:
                # 극단 케이스 방어
                added_cols = [out2.columns[0]]

            idx_src = added_cols[0]
            out2 = out2.rename(columns={idx_src: "__idx"})
            eff_col = "__idx"
            out = out2

        # ---- datetime coercion ----
        out[eff_col] = pd.to_datetime(out[eff_col], errors="coerce")

        # ---- value column 1D 강제 ----
        if value_col in out.columns:
            col = out.loc[:, value_col]
            if isinstance(col, pd.DataFrame):  # 중복 컬럼/2D 방어
                col = col.iloc[:, 0]
            out[value_col] = pd.to_numeric(col, errors="coerce")
        else:
            raise RuntimeError(f"asof_daily_from_realtime: '{value_col}' column not found. cols={list(out.columns)}")

        eff = out[[eff_col, value_col]].dropna().copy().sort_values(eff_col)

        # ---- tz normalize ----
        s_ts = pd.to_datetime(start)
        e_ts = pd.to_datetime(end)
        if getattr(s_ts, "tzinfo", None) is not None:
            s_ts = s_ts.tz_localize(None)
        if getattr(e_ts, "tzinfo", None) is not None:
            e_ts = e_ts.tz_localize(None)

        idx = pd.date_range(s_ts, e_ts, freq=freq)

        # ---- asof merge ----
        daily = pd.DataFrame({eff_col: idx})
        merged = pd.merge_asof(
            daily.sort_values(eff_col),
            eff.sort_values(eff_col),
            on=eff_col,
            direction="backward",
        )

        s = merged[value_col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.Series(s.to_numpy(), index=idx, name=value_col)


# -----------------------------
# yfinance fetch (daily)
# -----------------------------
def fetch_yf_daily_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=str(start.date()),
                end=str((end + pd.Timedelta(days=1)).date()),
                progress=False,
            )
            if df is not None and not df.empty:
                break
            time.sleep(2 * (attempt + 1))  # backoff
        except Exception as e:
            print(f"[WARN] yfinance fetch {ticker} failed (attempt {attempt+1}/{max_retries}): {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))

    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]
    close = close.dropna()
    close.name = ticker
    return close


# -----------------------------
# Build DAILY state (Regime + Stress)
# -----------------------------
def build_daily_state(
    fred: FredClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stress_quantile: float = 0.80,
    stress_roll_window: int = 756,
    stress_min_periods: int = 252,
    debug: bool = False,
    debug_n: int = 5,
) -> pd.DataFrame:
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    start = start.tz_localize(None) if getattr(start, "tzinfo", None) else start
    end = end.tz_localize(None) if getattr(end, "tzinfo", None) else end

    daily_idx = pd.date_range(start, end, freq="B")

    def _dbg_df(name: str, df: pd.DataFrame, value_col: str = "value"):
        eprint(f"\n[DEBUG] {name} rows={len(df)} cols={list(df.columns)}")
        if df.empty:
            return
        eprint(df.head(debug_n).to_string(index=False))
        eprint("...")
        eprint(df.tail(debug_n).to_string(index=False))
        if value_col in df.columns:
            eprint(f"[DEBUG] {name} {value_col}.describe()")
            eprint(pd.to_numeric(df[value_col], errors="coerce").describe().to_string())
        if "realtime_start" in df.columns:
            rs = pd.to_datetime(df["realtime_start"], errors="coerce")
            eprint(f"[DEBUG] {name} realtime_start unique={rs.nunique()} min={rs.min()} max={rs.max()}")

    # ---- 1) Fetch monthly macro obs with "release-aware" realtime_start
    def fetch_monthly_with_release(series_id: str) -> pd.DataFrame:
        """
        - value: output_type=1 (latest / revised)  -> value만 신뢰 (realtime_start는 버림)
        - release date: output_type=4 (initial release only) -> earliest realtime_start per obs_date
        - release_start 없는 구간은 obs_date 기반 fallback으로 채움
        """
        latest = fred.fetch_observations(
            series_id,
            start - pd.DateOffset(years=3),
            end,
            output_type=1,
        )
        if latest.empty:
            return latest

        # ✅ output_type=1의 realtime_start/end는 "요청 구간" 의미라서 신뢰하지 않음
        latest["realtime_start"] = pd.NaT

        # ✅ initial release (vintage)로 release_start 만들기
        try:
            initial = fred.fetch_observations(
                series_id,
                start - pd.DateOffset(years=3),
                end,
                output_type=4,
                realtime_start="1776-07-04",
                realtime_end="9999-12-31",
            )
        except Exception as ex:
            if debug:
                eprint(f"[DEBUG] initial-release fetch failed for {series_id}: {ex} (fallback to obs_date-based)")
            initial = pd.DataFrame(columns=["obs_date", "realtime_start"])

        rel = pd.DataFrame(columns=["obs_date", "release_start"])
        if not initial.empty:
            rel = (
                initial.sort_values(["obs_date", "realtime_start"])
                .drop_duplicates("obs_date", keep="first")[["obs_date", "realtime_start"]]
                .rename(columns={"realtime_start": "release_start"})
            )

        out = latest.merge(rel, on="obs_date", how="left")

        # ✅ fallback: release_start가 없으면 "obs_date 다음달 중순"으로 가정
        fallback_release = (
            pd.to_datetime(out["obs_date"])
            + pd.offsets.MonthBegin(1)  # 다음달 1일
            + pd.Timedelta(days=14)     # 다음달 15일
        )

        out["realtime_start"] = pd.to_datetime(out["release_start"], errors="coerce").fillna(fallback_release)

        # (선택) 혹시라도 미래로 튀면 end로 캡
        end_ts = pd.to_datetime(end)
        out["realtime_start"] = out["realtime_start"].where(out["realtime_start"] <= end_ts, end_ts)

        out = out.drop(columns=["release_start"], errors="ignore")
        out = out.sort_values(["obs_date", "realtime_start"]).reset_index(drop=True)
        return out

    cpi_obs = fetch_monthly_with_release(DEFAULT_FRED_SERIES["cpi"])
    core_obs = fetch_monthly_with_release(DEFAULT_FRED_SERIES["core_cpi"])
    unrate_obs = fetch_monthly_with_release(DEFAULT_FRED_SERIES["unrate"])
    orders_obs = fetch_monthly_with_release(DEFAULT_FRED_SERIES["mfg_new_orders"])

    if debug:
        _dbg_df("FRED monthly core_cpi (release-aware)", core_obs)
        _dbg_df("FRED monthly cpi (release-aware)", cpi_obs)
        _dbg_df("FRED monthly unrate (release-aware)", unrate_obs)
        _dbg_df("FRED monthly mfg_new_orders (release-aware)", orders_obs)

    # ---- Derived monthly then mapped to as-of daily by realtime_start
    def derive_monthly_yoy(obs_df: pd.DataFrame) -> pd.DataFrame:
        dfm = (
            obs_df.sort_values("obs_date")
            .drop_duplicates("obs_date", keep="last")
            .set_index("obs_date")
        )
        yoy = pct_change_12m_monthly(dfm["value"])
        out = pd.DataFrame(
            {
                "obs_date": yoy.index,
                "value": yoy.values,
                "realtime_start": dfm.loc[yoy.index, "realtime_start"].values,
            }
        )
        out["obs_date"] = pd.to_datetime(out["obs_date"])
        out["realtime_start"] = pd.to_datetime(out["realtime_start"], errors="coerce")
        out = (
            out.dropna(subset=["value", "realtime_start"])
            .sort_values(["obs_date", "realtime_start"])
            .reset_index(drop=True)
        )
        return out[["obs_date", "value", "realtime_start"]]

    def derive_monthly_diff3(obs_df: pd.DataFrame) -> pd.DataFrame:
        dfm = (
            obs_df.sort_values("obs_date")
            .drop_duplicates("obs_date", keep="last")
            .set_index("obs_date")
        )
        diff3 = dfm["value"].diff(3)
        out = pd.DataFrame(
            {
                "obs_date": diff3.index,
                "value": diff3.values,
                "realtime_start": dfm.loc[diff3.index, "realtime_start"].values,
            }
        )
        out["obs_date"] = pd.to_datetime(out["obs_date"])
        out["realtime_start"] = pd.to_datetime(out["realtime_start"], errors="coerce")
        out = (
            out.dropna(subset=["value", "realtime_start"])
            .sort_values(["obs_date", "realtime_start"])
            .reset_index(drop=True)
        )
        return out[["obs_date", "value", "realtime_start"]]

    core_yoy_obs = derive_monthly_yoy(core_obs)
    orders_yoy_obs = derive_monthly_yoy(orders_obs)
    unrate_chg3_obs = derive_monthly_diff3(unrate_obs)
    orders_mom3_obs = derive_monthly_diff3(orders_obs)

    if debug:
        _dbg_df("Derived core_yoy_obs (YoY%)", core_yoy_obs)
        _dbg_df("Derived orders_yoy_obs (YoY%)", orders_yoy_obs)
        _dbg_df("Derived unrate_chg3_obs (diff3)", unrate_chg3_obs)
        _dbg_df("Derived orders_mom3_obs (diff3)", orders_mom3_obs)

    # ---- rolling median (monthly)
    def add_monthly_rolling_median(obs: pd.DataFrame, window_n: int = 36, min_n: int = 18) -> pd.DataFrame:
        df = obs.copy()
        if "obs_date" in df.columns:
            df = df.set_index("obs_date")
        if "value" not in df.columns:
            raise RuntimeError(f"add_monthly_rolling_median(): 'value' not found. cols={list(df.columns)}")
        s = pd.to_numeric(df["value"], errors="coerce")
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.sort_index()
        med = s.rolling(window_n, min_periods=min_n).median()
        return pd.DataFrame({"value_raw": s, "value": med}, index=s.index)

    core_yoy_med_obs = add_monthly_rolling_median(core_yoy_obs, window_n=36, min_n=18)
    orders_mom_med_obs = add_monthly_rolling_median(orders_mom3_obs, window_n=12, min_n=6)

    # ---- As-of daily
    core_cpi_yoy = fred.asof_daily_from_realtime(core_yoy_obs, start, end)
    core_cpi_yoy.name = "core_cpi_yoy"

    core_cpi_yoy_med = fred.asof_daily_from_realtime(core_yoy_med_obs, start, end)
    core_cpi_yoy_med.name = "core_cpi_yoy_med"

    orders_yoy = fred.asof_daily_from_realtime(orders_yoy_obs, start, end)
    orders_yoy.name = "orders_yoy"

    orders_mom = fred.asof_daily_from_realtime(orders_mom3_obs, start, end)
    orders_mom.name = "orders_mom"

    orders_mom_med = fred.asof_daily_from_realtime(orders_mom_med_obs, start, end)
    orders_mom_med.name = "orders_mom_med"

    unrate_chg_3m = fred.asof_daily_from_realtime(unrate_chg3_obs, start, end)
    unrate_chg_3m.name = "unrate_chg_3m"

    # ---- 2) Daily/weekly market series
    def fred_daily_level(series_id: str) -> pd.Series:
        obs = fred.fetch_observations(series_id, start - pd.DateOffset(days=10), end, output_type=1)
        if obs.empty:
            return pd.Series(index=daily_idx, dtype=float)
        s = obs.sort_values("obs_date").drop_duplicates("obs_date", keep="last").set_index("obs_date")["value"]
        s = s.reindex(pd.date_range(s.index.min(), end, freq="D")).ffill()
        s = s.reindex(daily_idx, method="ffill")
        return s

    dgs2 = fred_daily_level(DEFAULT_FRED_SERIES["dgs2"])
    dgs2.name = "dgs2"

    dgs10 = fred_daily_level(DEFAULT_FRED_SERIES["dgs10"])
    dgs10.name = "dgs10"

    hy_oas = fred_daily_level(DEFAULT_FRED_SERIES["hy_oas"])
    hy_oas.name = "hy_oas"

    stlfsi4_raw = fred_daily_level(DEFAULT_FRED_SERIES["stlfsi4"])
    stlfsi4_raw.name = "stlfsi4"

    # ---- 3) yfinance daily SPX/VIX
    spx_close = fetch_yf_daily_close(YF_TICKERS["spx"], start, end)
    vix_close = fetch_yf_daily_close(YF_TICKERS["vix"], start, end)
    if spx_close.empty:
        raise RuntimeError("yfinance ^GSPC fetch failed.")
    if vix_close.empty:
        raise RuntimeError("yfinance ^VIX fetch failed.")

    spx_close = spx_close.reindex(daily_idx).ffill()
    vix = vix_close.reindex(daily_idx).ffill()
    vix.name = "vix"

    # ---- 4) Stress score (daily)
    dd = spx_close / spx_close.rolling(63, min_periods=20).max() - 1.0
    dd_stress = (-dd).clip(lower=0.0)
    dd_z = robust_zscore(dd_stress, window=252, min_periods=63).fillna(0.0)

    vix_z = robust_zscore(vix, window=252, min_periods=63).fillna(0.0)
    credit_z = robust_zscore(hy_oas, window=252, min_periods=63).fillna(0.0)
    stlfsi_z = robust_zscore(stlfsi4_raw, window=252, min_periods=63).fillna(0.0)

    w_vix, w_dd, w_cr, w_fsi = 0.35, 0.35, 0.20, 0.10

    # ✅ FIX: 상쇄 방지(각 컴포넌트별로 '스트레스(양의 편차)'만 반영)
    vix_pos = vix_z.clip(lower=0.0)
    dd_pos = dd_z.clip(lower=0.0)
    credit_pos = credit_z.clip(lower=0.0)
    stlfsi_pos = stlfsi_z.clip(lower=0.0)

    contrib_vix = w_vix * vix_pos
    contrib_dd = w_dd * dd_pos
    contrib_cr = w_cr * credit_pos
    contrib_fsi = w_fsi * stlfsi_pos

    stress_score = (contrib_vix + contrib_dd + contrib_cr + contrib_fsi).clip(lower=0.0)
    stress_score.name = "stress_score"

    thr = stress_score.rolling(stress_roll_window, min_periods=stress_min_periods).quantile(stress_quantile)
    stress_flag = (stress_score >= thr) | (stress_score >= 1.25)
    stress_flag.name = "stress_flag"

    contrib_df = pd.DataFrame(
        {"vix": contrib_vix, "drawdown": contrib_dd, "credit": contrib_cr, "stlfsi4": contrib_fsi},
        index=daily_idx,
    ).fillna(0.0)

    driver = contrib_df.idxmax(axis=1)
    driver = driver.where(stress_score > 0.1, other="none")
    driver.name = "stress_driver"

    # ---- 5) Regime
    infl_hot = (core_cpi_yoy > core_cpi_yoy_med) | (core_cpi_yoy >= 3.0)
    growth_ok = (orders_yoy >= 0.0) & (orders_mom >= orders_mom_med) & (unrate_chg_3m <= 0.2)

    def label_row(g_ok: bool, i_hot: bool) -> Tuple[int, str]:
        if g_ok and (not i_hot):
            return 1, "Goldilocks"
        if g_ok and i_hot:
            return 2, "Reflation"
        if (not g_ok) and i_hot:
            return 3, "Stagflation"
        return 4, "Recession"

    regime_id_list: List[int] = []
    regime_label_list: List[str] = []
    for d in daily_idx:
        rid, lab = label_row(bool(growth_ok.loc[d]), bool(infl_hot.loc[d]))
        regime_id_list.append(rid)
        regime_label_list.append(lab)

    regime_id = pd.Series(regime_id_list, index=daily_idx, name="regime_id")
    regime_label = pd.Series(regime_label_list, index=daily_idx, name="regime_label")

    yc_10y2y = (dgs10 - dgs2).rename("yc_10y2y")

    # ---- 6) Assemble
    out = pd.DataFrame(index=daily_idx)
    out["date"] = out.index.date
    out["regime_id"] = regime_id.astype(int)
    out["regime_label"] = regime_label.astype(str)

    out["stress_score"] = stress_score
    out["stress_flag"] = stress_flag.astype(bool)
    out["stress_driver"] = driver.astype(str)

    # debug cols (persistable)
    out["core_cpi_yoy"] = core_cpi_yoy
    out["core_cpi_yoy_med"] = core_cpi_yoy_med
    out["orders_yoy"] = orders_yoy
    out["orders_mom"] = orders_mom
    out["orders_mom_med"] = orders_mom_med
    out["unrate_chg_3m"] = unrate_chg_3m
    out["yc_10y2y"] = yc_10y2y
    out["hy_oas"] = hy_oas
    out["stlfsi4"] = stlfsi4_raw
    out["vix"] = vix

    return out


# -----------------------------
# Backtest
# -----------------------------
def backtest_daily(state_df: pd.DataFrame) -> Dict[str, Any]:
    if yf is None:
        return {"error": "yfinance not installed"}

    d0 = pd.to_datetime(state_df["date"].min())
    d1 = pd.to_datetime(state_df["date"].max())

    spx = fetch_yf_daily_close("^GSPC", d0, d1)
    if spx.empty:
        return {"error": "SPX fetch failed"}
    idx = pd.date_range(d0, d1, freq="B")
    spx = spx.reindex(idx).ffill()
    ret = spx.pct_change().fillna(0.0)

    s = state_df.copy()
    s.index = pd.to_datetime(s["date"])
    s = s.reindex(idx).ffill()

    base_weight = {1: 0.70, 2: 0.80, 3: 0.50, 4: 0.40}

    threshold = 1.0
    k = 0.50
    floor = 0.20
    cap = 1.00

    base = s["regime_id"].map(base_weight).astype(float)
    stress_score = pd.to_numeric(s["stress_score"], errors="coerce").fillna(0.0)
    excess = (stress_score - threshold).clip(lower=0.0)
    risk_cut = (1.0 - k * excess).clip(lower=floor, upper=cap)

    w_raw = (base * risk_cut).clip(0.0, 1.0)
    w = w_raw.shift(1).fillna(0.0)

    port_ret = w * ret
    eq = (1.0 + port_ret).cumprod()

    eq_m = eq.resample("ME").last()
    port_ret_m = eq_m.pct_change().dropna()

    stats_d = perf_stats_from_returns(port_ret, freq="D")
    stats_m = perf_stats_from_returns(port_ret_m, freq="M")

    diag = {
        "stress_true_ratio": float((s["stress_flag"].astype(bool)).mean()),
        "start": str(idx.min().date()),
        "end": str(idx.max().date()),
        "n_days": int(len(idx)),
    }

    return {"daily": stats_d, "monthly": stats_m, "diag": diag}


# -----------------------------
# Supabase writer (PostgREST)
# -----------------------------
class SupabaseWriter:
    def __init__(self, url: str, api_key: str, table: str, schema: str = "public"):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.table = table
        self.schema = schema

    def upsert_rows(self, rows: List[Dict[str, Any]], on_conflict: str = "date", chunk: int = 1000) -> None:
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


def make_rows_for_supabase(state_df: pd.DataFrame) -> List[Dict[str, Any]]:
    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out: List[Dict[str, Any]] = []

    for _, row in state_df.iterrows():
        out.append(
            {
                "date": str(row["date"]),
                "regime_id": int(row["regime_id"]),
                "regime_label": str(row["regime_label"]),
                "stress_flag": bool(row["stress_flag"]),
                "stress_score": None if pd.isna(row["stress_score"]) else float(row["stress_score"]),
                "stress_driver": str(row["stress_driver"]) if "stress_driver" in row else None,
                "core_cpi_yoy": None if pd.isna(row["core_cpi_yoy"]) else float(row["core_cpi_yoy"]),
                "core_cpi_yoy_med": None if pd.isna(row.get("core_cpi_yoy_med", np.nan)) else float(row["core_cpi_yoy_med"]),
                "orders_yoy": None if pd.isna(row["orders_yoy"]) else float(row["orders_yoy"]),
                "orders_mom": None if pd.isna(row.get("orders_mom", np.nan)) else float(row["orders_mom"]),
                "orders_mom_med": None if pd.isna(row.get("orders_mom_med", np.nan)) else float(row["orders_mom_med"]),
                "unrate_chg_3m": None if pd.isna(row["unrate_chg_3m"]) else float(row["unrate_chg_3m"]),
                "yc_10y2y": None if pd.isna(row["yc_10y2y"]) else float(row["yc_10y2y"]),
                "hy_oas": None if pd.isna(row["hy_oas"]) else float(row["hy_oas"]),
                "stlfsi4": None if pd.isna(row["stlfsi4"]) else float(row["stlfsi4"]),
                "vix": None if pd.isna(row["vix"]) else float(row["vix"]),
                "updated_at": now_utc,
            }
        )
    return out


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2000-01-01")
    p.add_argument("--end", type=str, default=None, help="default: today(UTC)")
    p.add_argument("--save-supabase", action="store_true")
    p.add_argument("--supabase-table", type=str, default=None)
    p.add_argument("--print-latest", action="store_true")
    p.add_argument("--print-sample", action="store_true")
    p.add_argument("--print-backtest", action="store_true")

    p.add_argument("--stress-quantile", type=float, default=0.80)
    p.add_argument("--stress-roll-window", type=int, default=756)
    p.add_argument("--stress-min-periods", type=int, default=252)

    p.add_argument("--debug", action="store_true", help="print debug diagnostics")
    p.add_argument("--debug-n", type=int, default=5, help="head/tail rows to print for debug tables")
    return p.parse_args()


def main() -> int:
    try:
        import dotenv  # type: ignore
        dotenv.load_dotenv()
    except Exception:
        pass

    args = parse_args()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else pd.Timestamp.now(tz="UTC").normalize()

    start = start.tz_localize(None) if getattr(start, "tzinfo", None) else start
    end = end.tz_localize(None) if getattr(end, "tzinfo", None) else end

    fred_key = env("FRED_API_KEY")
    if not fred_key:
        raise RuntimeError("FRED_API_KEY is required.")

    supa_url = env("SUPABASE_URL")
    supa_key = env("SUPABASE_SERVICE_ROLE_KEY") or env("SUPABASE_API_KEY")
    supa_table = args.supabase_table or env("SUPABASE_TABLE", "macro_state_daily")

    fred = FredClient(api_key=fred_key)

    state = build_daily_state(
        fred=fred,
        start=start,
        end=end,
        stress_quantile=args.stress_quantile,
        stress_roll_window=args.stress_roll_window,
        stress_min_periods=args.stress_min_periods,
        debug=args.debug,
        debug_n=args.debug_n,
    )

    if args.print_latest:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(
            state.tail(1)[
                [
                    "regime_id",
                    "regime_label",
                    "stress_flag",
                    "stress_score",
                    "stress_driver",
                    "core_cpi_yoy",
                    "core_cpi_yoy_med",
                    "orders_yoy",
                    "orders_mom",
                    "orders_mom_med",
                    "unrate_chg_3m",
                    "yc_10y2y",
                    "hy_oas",
                    "stlfsi4",
                    "vix",
                ]
            ]
        )

    if args.print_sample:
        print(state.tail(10)[["regime_id", "regime_label", "stress_flag", "stress_score", "stress_driver"]])

    if args.print_backtest:
        bt = backtest_daily(state)
        print("\n[Backtest] daily:", bt.get("daily"))
        print("[Backtest] monthly:", bt.get("monthly"))
        print("[Backtest] diag:", bt.get("diag"))

    if args.save_supabase:
        if not (supa_url and supa_key):
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_API_KEY) are required.")
        writer = SupabaseWriter(url=supa_url, api_key=supa_key, table=supa_table)
        rows = make_rows_for_supabase(state)
        writer.upsert_rows(rows, on_conflict="date")
        print(f"Upserted {len(rows)} daily rows into {supa_table}.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as ex:
        eprint(f"[ERROR] {ex}")
        raise
