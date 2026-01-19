import os
import time
import requests
import pandas as pd
import yfinance as yf
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
FRED_API_KEY = os.environ["FRED_API_KEY"]

def fred_series(series_id: str, start: str = "2000-01-01") -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series(
        {pd.to_datetime(x["date"]): (None if x["value"] == "." else float(x["value"])) for x in obs},
        name=series_id,
    ).sort_index()
    return s

def yf_close(tickers, start="2000-01-01") -> pd.DataFrame:
    for attempt in range(5):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if df is None or df.empty:
                raise RuntimeError("yfinance empty")
            close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
            if not isinstance(close, pd.DataFrame):
                close = close.to_frame()
            return close.sort_index()
        except Exception as e:
            time.sleep(2 ** attempt)
    raise RuntimeError("yfinance failed")

def main():
    # 1) FRED
    fred_map = {
        "cpi_yoy": "CPIAUCSL",
        "core_yoy": "CPILFESL",
        "core_yoy": "CPILFESL",
        # "pmi": "NAPM",  # 400 Bad Request (likely restricted/discontinued on FRED API)
        "unemp": "UNRATE",
        "y2": "DGS2",
        "y10": "DGS10",
        "credit_spread": "BAMLH0A0HYM2",  # 예시(HY OAS)
    }
    fred_df = pd.concat([fred_series(sid) for sid in fred_map.values()], axis=1)
    fred_df.columns = list(fred_map.keys())

    # 2) yfinance
    yf_map = {
        "dxy": "DX-Y.NYB",
        "oil": "CL=F",
        "gold": "GC=F",
    }
    yf_df = yf_close(list(yf_map.values()))
    yf_df = yf_df.rename(columns={v: k for k, v in yf_map.items()})

    # 3) merge -> 월말(last) + ffill
    daily = pd.concat([fred_df, yf_df], axis=1).sort_index()
    monthly = daily.resample("ME").last().ffill()

    # 4) Supabase upsert (PK=date)
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    payload = []
    for idx, row in monthly.iterrows():
        payload.append({
            "date": idx.date().isoformat(),
            **{c: (None if pd.isna(row[c]) else float(row[c])) for c in monthly.columns},
        })

    # 큰 배치면 쪼개기(예: 500행)
    chunk = 500
    for i in range(0, len(payload), chunk):
        sb.table("macro_monthly").upsert(payload[i:i+chunk]).execute()  # upsert 예시  [oai_citation:2‡Supabase](https://supabase.com/docs/reference/python/upsert?utm_source=chatgpt.com)

if __name__ == "__main__":
    main()