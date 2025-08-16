# price_volume_analysis.py
"""
Intraday price/volume analysis utilities built on yfinance.

What this script does (high level):
1) Downloads OHLCV data for a single symbol at the requested interval/period.
2) Normalizes the columns (handles MultiIndex and odd column names from yfinance).
3) Computes:
   - Typical Price (TP)
   - Per-session intraday VWAP
   - Price–volume profile (histogram of volume by price bins)
   - “Range fraction” features (where today’s TP sits inside day’s High–Low range)
   - Up/Down volume split (proxy for buying/selling pressure)
   - Daily RVOL (today’s volume vs. rolling average)
4) Prints a compact textual summary and writes CSVs for further analysis.

Notes:
- All intraday timestamps are converted to the provided timezone (`TZ`).
- Daily bars from yfinance are midnight-stamped UTC; we cosmetically shift them to 16:00 ET
  to match market close so your outputs line up with intuition.
"""

from __future__ import annotations

from datetime import datetime
import re
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ------------------------ User-configurable parameters ------------------------
SYMBOL   = "CRCL"        # Ticker to analyze (string)
INTERVAL = "1d"          # One of: '1m','2m','5m','15m','30m','1h','1d'
PERIOD   = "100d"          # One of: '1d','5d','1mo','3mo','6mo','1y'
TZ       = "America/New_York"  # Output timezone for intraday bars
INCLUDE_EXTENDED = False # If True (and intraday), include pre/after-market bars
N_PRICE_BINS = 40        # Number of price bins for the volume profile
RVOL_LOOKBACK_DAYS = 20  # Rolling window used for RVOL on daily bars

# Directory where CSV outputs will be saved
OUT_DIR = Path.cwd() / "volume_indicator_csv"  # change to Path(__file__).parent / "csv" if you prefer script-relative

# ==============================================================================
# Helpers for robust column handling
# ==============================================================================

def _pick_symbol_level(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    If yfinance returns MultiIndex columns (common when requesting multiple tickers),
    reduce to a single level for `symbol`.

    Supports both layouts:
      - Level 0 = symbol, Level 1 = field (Open/High/...)
      - Level 0 = field,  Level 1 = symbol

    If no clear match is found, we fall back to flattening the columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by yfinance.
    symbol : str
        Ticker to select from MultiIndex columns.

    Returns
    -------
    pd.DataFrame
        DataFrame reduced to a single symbol with simple columns.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    lv0 = df.columns.get_level_values(0)
    lv1 = df.columns.get_level_values(1)

    if symbol in lv0:
        return df.xs(symbol, axis=1, level=0)
    if symbol in lv1:
        return df.xs(symbol, axis=1, level=1)

    # Fallback: flatten into single strings like "Open AAPL" or "AAPL Open".
    out = df.copy()
    out.columns = [' '.join([str(p) for p in tup if p]).strip() for tup in df.columns]
    return out


def _coerce_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Force the DataFrame into canonical OHLCV columns:
    ['Open', 'High', 'Low', 'Close', 'Volume'].

    yfinance sometimes returns odd cases:
      - 'Adj Close'
      - lowercase names
      - prefixed with the ticker (e.g., 'AAPL Close')

    We normalize heuristically and error out if we cannot find
    a 1:1 mapping to the 5 columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input with some superset/variant of OHLCV columns.
    symbol : str
        Ticker symbol used to strip any ticker tokens from column names.

    Returns
    -------
    pd.DataFrame
        Exact columns ['Open','High','Low','Close','Volume'] (in that order).

    Raises
    ------
    RuntimeError
        If we cannot identify all 5 desired columns.
    """
    # Fast path: already correct
    if all(c in df.columns for c in ["Open","High","Low","Close","Volume"]):
        return df[["Open","High","Low","Close","Volume"]]

    sym_low = symbol.lower()
    norm = {}
    for c in df.columns:
        # Remove the ticker token and keep only letters to simplify comparisons
        s = re.sub(rf"{re.escape(sym_low)}", "", str(c).lower())
        s = re.sub(r"[^a-z]", "", s)
        norm[c] = s

    # Accepted normalized keys for each target column
    wants = {
        "Open":   {"open"},
        "High":   {"high"},
        "Low":    {"low"},
        "Close":  {"close", "adjclose"},
        "Volume": {"volume", "vol"},
    }

    chosen, used = {}, set()
    for target, keys in wants.items():
        for orig, n in norm.items():
            if orig in used:
                continue
            if n in keys:
                chosen[target] = orig
                used.add(orig)
                break

    if len(chosen) == 5:
        out = df[list(chosen.values())].copy()
        out.columns = ["Open","High","Low","Close","Volume"]
        return out

    raise RuntimeError(
        "Expected OHLCV columns not found after normalization.\n"
        f"Raw columns: {list(df.columns)}\n"
        f"Normalized (first 10 map): {dict(list(norm.items())[:10])}"
    )

# ==============================================================================
# Data Fetch (REAL DATA)
# ==============================================================================

def fetch(symbol: str, interval: str, period: str, tz: str, prepost: bool) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance and normalize columns.

    Parameters
    ----------
    symbol : str
        Ticker to download (e.g., "AAPL").
    interval : str
        yfinance interval string, e.g., "1m", "15m", "1h", "1d".
    period : str
        yfinance period string, e.g., "5d", "1mo", "6mo".
    tz : str
        IANA timezone name for the output index, e.g., "America/New_York".
    prepost : bool
        Include pre/after-market bars for intraday intervals.

    Returns
    -------
    pd.DataFrame
        Normalized OHLCV DataFrame indexed by timezone-aware timestamps.

    Raises
    ------
    RuntimeError
        If no data is returned.
    """
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=True,                                 # Adjust prices for splits/dividends
        prepost=prepost if interval != "1d" else False,   # yfinance ignores pre/post for daily
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} [{period}/{interval}]")

    # Ensure tz-aware index in the requested timezone
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)

    # Handle possible MultiIndex columns and odd names
    df = _pick_symbol_level(df, symbol)

    # Title-case simple names to a consistent format
    if not isinstance(df.columns, pd.MultiIndex):
        df = df.rename(columns={c: str(c).title() for c in df.columns})

    return _coerce_ohlcv(df, symbol)

# ==============================================================================
# Core Features
# ==============================================================================

def typical_price(df: pd.DataFrame) -> pd.Series:
    """
    Typical Price (TP) = (High + Low + Close) / 3

    Rationale: TP is a common proxy for “where trading happened”
    within the bar (more robust than just the close).

    Returns
    -------
    pd.Series
        Typical price per bar (float).
    """
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def vwap_by_day(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP computed per *session* (resets each day).

    VWAP_t = cumulative( TP_t * Vol_t ) / cumulative( Vol_t ), grouped by day.

    Implementation detail:
    - We group by the UTC date (naive date) of each timestamp to avoid DST
      slicing issues; each “trading day” stays intact.

    Returns
    -------
    pd.Series
        Per-bar VWAP values (aligned to df.index) named "VWAP".
    """
    tp = typical_price(df)
    # Stable group key: naive UTC date for each bar
    day_key = df.index.tz_convert("UTC").tz_localize(None).date

    num = (tp * df["Volume"]).groupby(day_key).cumsum()
    den = df["Volume"].groupby(day_key).cumsum().replace(0, np.nan)

    vwap = num / den
    vwap.name = "VWAP"
    return vwap


def volume_profile(df: pd.DataFrame, bins: int = 40) -> pd.DataFrame:
    """
    Aggregate traded volume into price bins based on each bar’s Typical Price.

    For each bar:
      - Find the price bin where its TP falls.
      - Add the bar’s volume to that bin.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ['High','Low','Close','Volume'] columns.
    bins : int
        Number of equally-spaced price bins between min(TP) and max(TP).

    Returns
    -------
    pd.DataFrame
        Columns:
          - PriceBin : float midpoint of the bin
          - Volume   : total volume accumulated into that bin

        Sorted by PriceBin ascending.
    """
    tp = typical_price(df)
    v  = df["Volume"].astype(float)

    lo, hi = float(tp.min()), float(tp.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        # Degenerate case: single price or NaN; return one bin with all volume
        return pd.DataFrame({
            "PriceBin": [lo if np.isfinite(lo) else 0.0],
            "Volume": [float(v.sum())]
        })

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.digitize(tp.values, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    vol_by_bin = np.bincount(idx, weights=v.values, minlength=bins)
    mids = (edges[:-1] + edges[1:]) / 2.0

    prof = pd.DataFrame({"PriceBin": mids, "Volume": vol_by_bin})
    return prof.sort_values("PriceBin", ascending=True, ignore_index=True)


def classify_range_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each bar, compute where its TP sits inside the *day’s* [Low, High] range.

    Definitions (per day):
      RangeFrac in [0, 1]:
        0.0 = at the day’s Low
        1.0 = at the day’s High
      RangeBucket (categorical):
        'lower'  if RangeFrac < 1/3
        'middle' otherwise (default)
        'upper'  if RangeFrac > 2/3

    Returns
    -------
    pd.DataFrame
        Original columns plus:
          - TP
          - RangeFrac
          - RangeBucket (object dtype with values 'lower'/'middle'/'upper')
    """
    out = df.copy()
    tp = typical_price(out)

    day_key = out.index.tz_convert("UTC").tz_localize(None).date
    grp = out.groupby(day_key, group_keys=False)

    day_low  = grp["Low"].transform("min")
    day_high = grp["High"].transform("max")
    rng = (day_high - day_low).replace(0, np.nan)

    frac = (tp - day_low) / rng
    out["TP"] = tp
    out["RangeFrac"] = frac.clip(0, 1)

    buck = np.full(len(out), "middle", dtype=object)
    buck[out["RangeFrac"] < 1/3] = "lower"
    buck[out["RangeFrac"] > 2/3] = "upper"
    out["RangeBucket"] = buck
    return out


def up_down_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split each bar’s volume into “Up” or “Down” buckets as a rough pressure proxy.

    Rule:
      - UpVol   = Volume if Close >= Open
      - DownVol = Volume if Close <  Open

    Returns
    -------
    pd.DataFrame
        Original columns plus UpVol and DownVol.
    """
    out = df.copy()
    up_mask = (out["Close"] >= out["Open"])
    out["UpVol"] = out["Volume"].where(up_mask, 0)
    out["DownVol"] = out["Volume"].where(~up_mask, 0)
    return out


def rvol_daily(
    symbol: str,
    tz: str,
    lookback: int = 20,
    use_adjusted: bool = True
) -> pd.DataFrame:
    """
    Compute daily Relative Volume (RVOL) on daily bars:
      RVOL = TodayVolume / RollingAverageVolume(lookback)

    We download roughly 3× the lookback (min 60 days) to ensure a stable average.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    tz : str
        Timezone for the output index.
    lookback : int
        Rolling window length for average volume.
    use_adjusted : bool
        If True, request auto_adjust=True for prices (does not affect volume).

    Returns
    -------
    pd.DataFrame
        Columns:
          - Close
          - Volume
          - AvgVol (rolling mean over `lookback`)
          - RVOL   (= Volume / AvgVol)
        Index is timezone-aware and cosmetically shifted to 16:00 local if
        the source stamps daily bars at midnight (common in yfinance).
    """
    dfd = yf.download(
        symbol,
        interval="1d",
        period=f"{max(lookback * 3, 60)}d",
        auto_adjust=use_adjusted,
        prepost=False,
        progress=False,
    )
    if dfd.empty:
        return pd.DataFrame()

    dfd.index = pd.to_datetime(dfd.index, utc=True).tz_convert(tz)

    # Handle MultiIndex columns if present
    if isinstance(dfd.columns, pd.MultiIndex):
        if symbol in dfd.columns.get_level_values(0):
            dfd = dfd.xs(symbol, axis=1, level=0)
        elif symbol in dfd.columns.get_level_values(1):
            dfd = dfd.xs(symbol, axis=1, level=1)
        else:
            dfd.columns = [' '.join([str(p) for p in tup if p]).strip() for tup in dfd.columns]

    dfd = dfd.rename(columns={c: str(c).title() for c in dfd.columns})

    if "Volume" not in dfd.columns:
        return pd.DataFrame()

    out = dfd.copy()
    out["AvgVol"] = (
        out["Volume"]
        .rolling(lookback, min_periods=lookback // 2)
        .mean()
        .fillna(0)       # Avoid division by NaN; interpret as 0 until enough history
        .astype(int)     # Friendly integer display; if you prefer floats, drop this
    )
    out["RVOL"] = out["Volume"] / out["AvgVol"].replace(0, np.nan)

    # Cosmetic index shift: many daily series show 00:00; move to 16:00 local for readability
    idx_local = out.index.tz_convert(tz)
    if all((idx_local.hour == 0) & (idx_local.minute == 0)):
        out.index = idx_local.normalize() + pd.Timedelta(hours=16)

    price_col = "Close" if "Close" in out.columns else ("Adj Close" if "Adj Close" in out.columns else None)
    cols = [c for c in [price_col, "Volume", "AvgVol", "RVOL"] if c in out.columns]
    return out[cols].rename(columns={price_col: "Close"}) if cols else out[["Volume","AvgVol","RVOL"]]

# ==============================================================================
# MAIN (REAL DATA)
# ==============================================================================

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1) Pull intraday (or daily) bars for the configured symbol
    df = fetch(SYMBOL, INTERVAL, PERIOD, TZ, INCLUDE_EXTENDED)

    # 2) Per-session intraday VWAP (only meaningful for intraday intervals)
    if INTERVAL.endswith(("m", "h")):
        df["VWAP"] = vwap_by_day(df)
    else:
        df["VWAP"] = np.nan  # Placeholder for daily bars

    # 3) Enrich bars with Up/Down volume and range-fraction features
    df2 = up_down_volume(df)
    df3 = classify_range_fraction(df2)

    # Identify the latest session (by UTC date key) for "today" summaries
    day_key = df3.index.tz_convert("UTC").tz_localize(None).date
    last_day = day_key[-1]
    today = df3[day_key == last_day]

    # Global and last-session volume profiles
    prof_all  = volume_profile(df3, bins=N_PRICE_BINS)
    prof_last = volume_profile(today, bins=max(10, N_PRICE_BINS // 2)) if not today.empty else pd.DataFrame()

    # Volume concentration by 'RangeBucket' for each day
    bucket_vol = df3.groupby([day_key, "RangeBucket"])["Volume"].sum().unstack(fill_value=0)
    bucket_vol["Total"] = bucket_vol.sum(axis=1)
    # Ensure consistent presence of all three buckets
    for k in ("upper", "middle", "lower"):
        if k not in bucket_vol.columns:
            bucket_vol[k] = 0
    bucket_vol["UpperPct"]  = bucket_vol["upper"]  / bucket_vol["Total"].replace(0, np.nan)
    bucket_vol["MiddlePct"] = bucket_vol["middle"] / bucket_vol["Total"].replace(0, np.nan)
    bucket_vol["LowerPct"]  = bucket_vol["lower"]  / bucket_vol["Total"].replace(0, np.nan)

    # Aggregate Up/Down Volume by day (proxy of buy/sell pressure)
    ud_day = df3.groupby(day_key)[["UpVol", "DownVol", "Volume"]].sum()
    ud_day["UpPct"] = ud_day["UpVol"] / ud_day["Volume"].replace(0, np.nan)
    ud_day["DownPct"] = ud_day["DownVol"] / ud_day["Volume"].replace(0, np.nan)

    # 4) Daily RVOL (separate daily download)
    dfd_rvol = rvol_daily(SYMBOL, TZ, RVOL_LOOKBACK_DAYS, use_adjusted=True)

    # ------------------------ Console prints (compact) ------------------------
    pd.options.display.float_format = '{:.2f}'.format
    print(f"\n=== DATA SUMMARY: {SYMBOL} [{PERIOD}/{INTERVAL}] tz={TZ} prepost={INCLUDE_EXTENDED} ===")
    print(df.tail(3)[["Open", "High", "Low", "Close", "Volume", "VWAP"]])

    if not today.empty:
        t_lo, t_hi = float(today["Low"].min()), float(today["High"].max())
        print("\n--- Today’s Range & Concentration ---")
        print(f"Day Low/High: {t_lo:.2f} / {t_hi:.2f}")
        # Show last 3 days of concentration across buckets
        print(bucket_vol.tail(3)[["UpperPct", "MiddlePct", "LowerPct"]])
        upct = float(bucket_vol.iloc[-1]["UpperPct"])
        print(f"Today upper-volume share: {upct:.1%} "
              f"({'bullish' if upct > 0.5 else 'neutral/bearish'})")

    print("\n--- Up/Down Volume by Day (last 5 sessions) ---")
    print(ud_day.tail(5)[["UpVol", "DownVol", "Volume", "UpPct", "DownPct"]])

    print("\n--- Volume Profile (All, top 10 bins) ---")
    print(prof_all.sort_values("Volume", ascending=False).head(10))

    if not prof_last.empty:
        print("\n--- Volume Profile (Last Session, top 5 bins) ---")
        print(prof_last.sort_values("Volume", ascending=False).head(5))

    if not dfd_rvol.empty:
        print("\n--- Daily RVOL (last 90 rows) ---")
        print(dfd_rvol.tail(90))

    # ------------------------ Save CSV outputs ------------------------
    # Filenames include a timestamp to avoid overwriting previous runs.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df3.to_csv(OUT_DIR / f"{SYMBOL}_{INTERVAL}_{PERIOD}_bars_with_features_{ts}.csv")
    prof_all.to_csv(OUT_DIR / f"{SYMBOL}_{INTERVAL}_{PERIOD}_volume_profile_all_{ts}.csv", index=False)
    if not prof_last.empty:
        prof_last.to_csv(OUT_DIR / f"{SYMBOL}_{INTERVAL}_{PERIOD}_volume_profile_last_{ts}.csv", index=False)
    bucket_vol.to_csv(OUT_DIR / f"{SYMBOL}_{INTERVAL}_{PERIOD}_bucket_volume_by_day_{ts}.csv")
    ud_day.to_csv(OUT_DIR / f"{SYMBOL}_{INTERVAL}_{PERIOD}_updown_volume_by_day_{ts}.csv")
    if not dfd_rvol.empty:
        dfd_rvol.to_csv(OUT_DIR / f"{SYMBOL}_daily_rvol_{ts}.csv")

    print(f"\nCSV files saved in: {OUT_DIR}")