# price_volume_analysis.py
from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# import yfinance as yf  # Uncomment locally if you want live data

SYMBOL   = "CRCL"        # your ticker
INTERVAL = "1m"          # '1m','2m','5m','15m','30m','1h','1d'
PERIOD   = "5d"          # '1d','5d','1mo','3mo','6mo','1y'
TZ       = "America/New_York"
INCLUDE_EXTENDED = False # set True to include pre/post (can be noisy)
N_PRICE_BINS = 40        # price bins for the volume profile
RVOL_LOOKBACK_DAYS = 20  # rolling days used for RVOL on daily bars

# ------------------------ Data Fetch ------------------------
def fetch(symbol: str, interval: str, period: str, tz: str, prepost: bool) -> pd.DataFrame:
    """
    Download OHLCV with yfinance and return tz-aware OHLCV.
    In this chat environment we can’t call yfinance; run this locally.
    Expected return columns: ['Open','High','Low','Close','Volume'].
    """
    raise RuntimeError("fetch() needs yfinance + internet; run locally.")

# ------------------------ Core Features ------------------------
def typical_price(df: pd.DataFrame) -> pd.Series:
    """Typical Price (TP) = (High + Low + Close)/3. A better proxy than Close for price*volume features."""
    return (df["High"] + df["Low"] + df["Close"]) / 3.0

def vwap_by_day(df: pd.DataFrame) -> pd.Series:
    """
    Per-session VWAP using intraday bars:
      VWAP_t = cumulative( TP_t * Vol_t ) / cumulative( Vol_t )  (within each day)
    We group by UTC date (naive) so DST doesn’t split sessions.
    """
    tp = typical_price(df)
    day_key = df.index.tz_convert("UTC").tz_localize(None).date  # stable per-day key
    num = (tp * df["Volume"]).groupby(day_key).cumsum()
    den = df["Volume"].groupby(day_key).cumsum().replace(0, np.nan)
    vwap = num / den
    vwap.name = "VWAP"
    return vwap

def volume_profile(df: pd.DataFrame, bins: int = 40) -> pd.DataFrame:
    """
    Approximate a volume profile by assigning each bar’s Volume to the price bin
    containing its Typical Price. Returns a table of bin mid-price and total volume.
    """
    tp = typical_price(df)
    v  = df["Volume"].astype(float)

    lo, hi = float(tp.min()), float(tp.max())
    # Handle empty/flat data safely
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return pd.DataFrame({"PriceBin": [lo if np.isfinite(lo) else 0.0],
                             "Volume": [float(v.sum())]})

    # Build equal-width price bins and assign each TP to a bin index
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.digitize(tp.values, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    # Sum volume per price bin; use bin midpoints for readability/plotting
    vol_by_bin = np.bincount(idx, weights=v.values, minlength=bins)
    mids = (edges[:-1] + edges[1:]) / 2.0

    prof = pd.DataFrame({"PriceBin": mids, "Volume": vol_by_bin})
    return prof.sort_values("PriceBin", ascending=True, ignore_index=True)

def classify_range_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each bar, measure where its TP sits within that day’s High–Low:
      - RangeFrac in [0,1]: 0=day’s low, 1=day’s high
      - RangeBucket: 'lower'/'middle'/'upper' third (vectorized)
    """
    out = df.copy()
    tp = typical_price(out)

    # Group by day (UTC naive date → robust to DST)
    day_key = out.index.tz_convert("UTC").tz_localize(None).date
    grp = out.groupby(day_key, group_keys=False)

    # .transform broadcasts per-day min/max back to each row (keeps shape)
    day_low  = grp["Low"].transform("min")
    day_high = grp["High"].transform("max")
    rng = (day_high - day_low).replace(0, np.nan)  # avoid div-by-zero

    frac = (tp - day_low) / rng
    out["TP"] = tp
    out["RangeFrac"] = frac.clip(0, 1)

    # Lower/middle/upper third buckets
    buck = np.full(len(out), "middle", dtype=object)
    buck[out["RangeFrac"] < 1/3] = "lower"
    buck[out["RangeFrac"] > 2/3] = "upper"
    out["RangeBucket"] = buck
    return out

def up_down_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy for buying/selling pressure:
      - UpVol  = Volume if Close >= Open, else 0
      - DownVol= Volume if Close <  Open, else 0
    """
    out = df.copy()
    up_mask = (out["Close"] >= out["Open"])
    out["UpVol"] = out["Volume"].where(up_mask, 0)
    out["DownVol"] = out["Volume"].where(~up_mask, 0)
    return out

def rvol_daily(symbol: str, tz: str, lookback: int = 20) -> pd.DataFrame:
    """
    Daily Relative Volume (RVOL) = today_vol / avg_vol(lookback days).
    This implementation uses synthetic data in this notebook; in your local
    environment, fetch daily bars with yfinance and compute the same fields.
    """
    days = pd.date_range(end=pd.Timestamp.now(tz=tz).normalize(), periods=60, freq="D")
    # Synthetic demo series
    close = 150 + np.cumsum(np.random.normal(0, 0.5, size=len(days)))
    base  = 1_000_000 * (1 + 0.2*np.sin(np.linspace(0, 4*np.pi, len(days))))
    vol   = np.clip(base + np.random.normal(0, 120_000, size=len(days)), 100_000, None)

    dfd = pd.DataFrame({"Close": close, "Volume": vol}, index=days)
    dfd["AvgVol"] = dfd["Volume"].rolling(lookback, min_periods=lookback//2).mean()
    dfd["Rvol"] = dfd["Volume"] / dfd["AvgVol"]
    return dfd[["Close","Volume","AvgVol","Rvol"]]

# ------------------------ Self-Test (no internet) ------------------------
def _make_synth_intraday(tz="America/New_York", days=2, freq="5min", start_price=150.0) -> pd.DataFrame:
    """
    Create a small synthetic intraday OHLCV DataFrame for N sessions.
    Useful to test logic when you can’t pull real data.
    """
    # Build two trading sessions (9:30 → 16:00 local time)
    sessions = []
    today = pd.Timestamp.now(tz=tz).normalize()
    for d in range(days):
        day = today - pd.Timedelta(days=(days - 1 - d))
        start = day + pd.Timedelta(hours=9, minutes=30)
        end   = day + pd.Timedelta(hours=16, minutes=0)
        idx = pd.date_range(start, end, freq=freq, tz=tz, inclusive="both")
        sessions.append(idx)
    idx = sessions[0].append(sessions[1]) if days == 2 else sessions[0]

    # Price path and OHLCV
    steps = np.random.normal(0, 0.15, size=len(idx)).cumsum()
    base  = start_price + steps
    close = base + np.random.normal(0, 0.05, size=len(idx))
    openp = np.concatenate([[start_price], close[:-1]])
    high  = np.maximum(openp, close) + np.abs(np.random.normal(0, 0.07, size=len(idx)))
    low   = np.minimum(openp, close) - np.abs(np.random.normal(0, 0.07, size=len(idx)))
    vol   = np.random.lognormal(mean=13.2, sigma=0.3, size=len(idx))

    df = pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)
    return df

if __name__ == "__main__":
    # Self-test with synthetic data (works anywhere)
    intraday = _make_synth_intraday(days=2, freq="5min", start_price=150.0)
    intraday["VWAP"] = vwap_by_day(intraday)
    ud = up_down_volume(intraday)
    enriched = classify_range_fraction(ud)
    prof_all = volume_profile(enriched, bins=N_PRICE_BINS)
    dfd_rvol = rvol_daily(SYMBOL, TZ, RVOL_LOOKBACK_DAYS)

    # Quick prints
    print("\nSynthetic intraday sample:")
    print(enriched.iloc[:10][["Open","High","Low","Close","Volume","VWAP","TP","RangeFrac","RangeBucket"]])

    print("\nVolume Profile (All, top 10 bins):")
    print(prof_all.sort_values("Volume", ascending=False).head(10))

    print("\nSynthetic Daily RVOL (last 5 days):")
    print(dfd_rvol.tail(5))