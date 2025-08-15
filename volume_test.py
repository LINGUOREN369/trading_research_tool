# price_volume_analysis.py
from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pandas as pd
import yfinance as yf

SYMBOL   = "CRCL"        # your ticker
INTERVAL = "1m"          # '1m','2m','5m','15m','30m','1h','1d'
PERIOD   = "5d"          # '1d','5d','1mo','3mo','6mo','1y'
TZ       = "America/New_York"
INCLUDE_EXTENDED = False # include pre/post for intraday only
N_PRICE_BINS = 40        # price bins for the volume profile
RVOL_LOOKBACK_DAYS = 20  # rolling days used for RVOL on daily bars

# ------------------------ Helpers for robust column handling ------------------------
def _pick_symbol_level(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    If yfinance returns MultiIndex columns, pick the ticker level.
    Supports both orders: (symbol, field) and (field, symbol).
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    lv0 = df.columns.get_level_values(0)
    lv1 = df.columns.get_level_values(1)

    if symbol in lv0:
        return df.xs(symbol, axis=1, level=0)
    if symbol in lv1:
        return df.xs(symbol, axis=1, level=1)

    # Fallback: flatten
    out = df.copy()
    out.columns = [' '.join([str(p) for p in tup if p]).strip() for tup in df.columns]
    return out

def _coerce_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Force DataFrame into ['Open','High','Low','Close','Volume'].
    Handles variations like 'Adj Close', lowercase names, ticker prefixes.
    """
    # Already perfect?
    if all(c in df.columns for c in ["Open","High","Low","Close","Volume"]):
        return df[["Open","High","Low","Close","Volume"]]

    sym_low = symbol.lower()
    norm = {}
    for c in df.columns:
        s = re.sub(rf"{re.escape(sym_low)}", "", str(c).lower())  # strip ticker token
        s = re.sub(r"[^a-z]", "", s)                              # letters only
        norm[c] = s

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
        f"Normalized (first 10): {dict(list(norm.items())[:10])}"
    )

# ------------------------ Data Fetch (REAL DATA) ------------------------
def fetch(symbol: str, interval: str, period: str, tz: str, prepost: bool) -> pd.DataFrame:
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=True,                                 # use adjusted for consistency
        prepost=prepost if interval != "1d" else False,   # daily ignores pre/post
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} [{period}/{interval}]")

    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    df = _pick_symbol_level(df, symbol)

    # Title-case simple names if not MultiIndex
    if not isinstance(df.columns, pd.MultiIndex):
        df = df.rename(columns={c: str(c).title() for c in df.columns})

    return _coerce_ohlcv(df, symbol)

# ------------------------ Core Features ------------------------
def typical_price(df: pd.DataFrame) -> pd.Series:
    """Typical Price (TP) = (High + Low + Close)/3. A better proxy than Close for price*volume features."""
    return (df["High"] + df["Low"] + df["Close"]) / 3.0

def vwap_by_day(df: pd.DataFrame) -> pd.Series:
    """
    Per-session VWAP using intraday bars:
      VWAP_t = cumulative( TP_t * Vol_t ) / cumulative( Vol_t )  (within each day)
    Group by UTC date (naive) so DST 不会拆分交易日。
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
    将每根K线的成交量按其典型价格落入的价格桶进行累加，得到价格-成交量分布。
    返回列：PriceBin（价格桶中点）、Volume（累计成交量）。
    """
    tp = typical_price(df)
    v  = df["Volume"].astype(float)

    lo, hi = float(tp.min()), float(tp.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return pd.DataFrame({"PriceBin": [lo if np.isfinite(lo) else 0.0],
                             "Volume": [float(v.sum())]})

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.digitize(tp.values, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    vol_by_bin = np.bincount(idx, weights=v.values, minlength=bins)
    mids = (edges[:-1] + edges[1:]) / 2.0

    prof = pd.DataFrame({"PriceBin": mids, "Volume": vol_by_bin})
    return prof.sort_values("PriceBin", ascending=True, ignore_index=True)

def classify_range_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每根K线计算其典型价格在当日高低区间中的相对位置：
      - RangeFrac ∈ [0,1]：0=当日最低，1=当日最高
      - RangeBucket ∈ {'lower','middle','upper'}：下/中/上三分位
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
    以收盘价相对开盘价判断上/下压力的成交量分配（代理指标）：
      - UpVol  = Volume if Close >= Open
      - DownVol= Volume if Close <  Open
    """
    out = df.copy()
    up_mask = (out["Close"] >= out["Open"])
    out["UpVol"] = out["Volume"].where(up_mask, 0)
    out["DownVol"] = out["Volume"].where(~up_mask, 0)
    return out

def rvol_daily(symbol: str, tz: str, lookback: int = 20, use_adjusted: bool = True) -> pd.DataFrame:
    """
    日线相对成交量（RVOL）= 当日成交量 / 过去N日平均成交量。
    use_adjusted=True 使用复权价格；False 使用官方未复权收盘价（仅展示，不影响RVOL）。
    """
    dfd = yf.download(
        symbol,
        interval="1d",
        period=f"{max(lookback*3, 60)}d",
        auto_adjust=use_adjusted,
        prepost=False,
        progress=False,
    )
    if dfd.empty:
        return pd.DataFrame()

    dfd.index = pd.to_datetime(dfd.index, utc=True).tz_convert(tz)

    # Handle MultiIndex if present
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
    .rolling(lookback, min_periods=lookback//2)
    .mean()
    .fillna(0)       # replace NaN with 0
    .astype(int)     # safe integer conversion
)
    out["RVOL"]   = out["Volume"] / out["AvgVol"]

    # Cosmetic: daily bars are stamped at midnight; shift to 16:00 ET if desired
    idx_local = out.index.tz_convert(tz)
    if all((idx_local.hour == 0) & (idx_local.minute == 0)):
        out.index = idx_local.normalize() + pd.Timedelta(hours=16)

    price_col = "Close" if "Close" in out.columns else ("Adj Close" if "Adj Close" in out.columns else None)
    cols = [c for c in [price_col, "Volume", "AvgVol", "RVOL"] if c in out.columns]
    return out[cols].rename(columns={price_col: "Close"}) if cols else out[["Volume","AvgVol","RVOL"]]

# ------------------------ MAIN (REAL DATA) ------------------------
if __name__ == "__main__":
    # 1) Pull real intraday bars
    df = fetch(SYMBOL, INTERVAL, PERIOD, TZ, INCLUDE_EXTENDED)

    # 2) Intraday VWAP per session
    if INTERVAL.endswith(("m", "h")):
        df["VWAP"] = vwap_by_day(df)
    else:
        df["VWAP"] = np.nan

    # 3) Enrich and analyze
    df2 = up_down_volume(df)
    df3 = classify_range_fraction(df2)

    # Today’s session
    day_key = df3.index.tz_convert("UTC").tz_localize(None).date
    last_day = day_key[-1]
    today = df3[day_key == last_day]

    # Volume profiles
    prof_all  = volume_profile(df3, bins=N_PRICE_BINS)
    prof_last = volume_profile(today, bins=max(10, N_PRICE_BINS//2)) if not today.empty else pd.DataFrame()

    # Bucketed volume by day
    bucket_vol = df3.groupby([day_key, "RangeBucket"])["Volume"].sum().unstack(fill_value=0)
    bucket_vol["Total"] = bucket_vol.sum(axis=1)
    for k in ("upper","middle","lower"):
        if k not in bucket_vol.columns:
            bucket_vol[k] = 0
    bucket_vol["UpperPct"]  = bucket_vol["upper"]  / bucket_vol["Total"].replace(0, np.nan)
    bucket_vol["MiddlePct"] = bucket_vol["middle"] / bucket_vol["Total"].replace(0, np.nan)
    bucket_vol["LowerPct"]  = bucket_vol["lower"]  / bucket_vol["Total"].replace(0, np.nan)

    # Up/Down volume by day
    ud_day = df3.groupby(day_key)[["UpVol","DownVol","Volume"]].sum()
    ud_day["UpPct"] = ud_day["UpVol"] / ud_day["Volume"].replace(0, np.nan)
    ud_day["DownPct"] = ud_day["DownVol"] / ud_day["Volume"].replace(0, np.nan)

    # Daily RVOL (real)
    dfd_rvol = rvol_daily(SYMBOL, TZ, RVOL_LOOKBACK_DAYS, use_adjusted=True)

    # ----- Prints -----
    pd.options.display.float_format = '{:.2f}'.format
    print(f"\n=== DATA SUMMARY: {SYMBOL} [{PERIOD}/{INTERVAL}] tz={TZ} prepost={INCLUDE_EXTENDED} ===")
    print(df.tail(3)[["Open","High","Low","Close","Volume","VWAP"]])

    if not today.empty:
        t_lo, t_hi = float(today["Low"].min()), float(today["High"].max())
        print("\n--- Today’s Range & Concentration ---")
        print(f"Day Low/High: {t_lo:.2f} / {t_hi:.2f}")
        print(bucket_vol.tail(3)[["UpperPct","MiddlePct","LowerPct"]])
        upct = float(bucket_vol.iloc[-1]["UpperPct"])
        print(f"Today upper-volume share: {upct:.1%} "
              f"({'bullish' if upct > 0.5 else 'neutral/bearish'})")

    print("\n--- Up/Down Volume by Day (last 5 sessions) ---")
    print(ud_day.tail(5)[["UpVol","DownVol","Volume","UpPct","DownPct"]])

    print("\n--- Volume Profile (All, top 10 bins) ---")
    print(prof_all.sort_values("Volume", ascending=False).head(10))

    if not prof_last.empty:
        print("\n--- Volume Profile (Last Session, top 5 bins) ---")
        print(prof_last.sort_values("Volume", ascending=False).head(5))

    if not dfd_rvol.empty:
        print("\n--- Daily RVOL (last 90 rows) ---")
        print(dfd_rvol.tail(90))

    # ----- Save CSVs -----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df3.to_csv(f"{SYMBOL}_{INTERVAL}_{PERIOD}_bars_with_features_{ts}.csv")
    prof_all.to_csv(f"{SYMBOL}_{INTERVAL}_{PERIOD}_volume_profile_all_{ts}.csv", index=False)
    if not prof_last.empty:
        prof_last.to_csv(f"{SYMBOL}_{INTERVAL}_{PERIOD}_volume_profile_last_{ts}.csv", index=False)
    bucket_vol.to_csv(f"{SYMBOL}_{INTERVAL}_{PERIOD}_bucket_volume_by_day_{ts}.csv")
    ud_day.to_csv(f"{SYMBOL}_{INTERVAL}_{PERIOD}_updown_volume_by_day_{ts}.csv")
    if not dfd_rvol.empty:
        dfd_rvol.to_csv(f"{SYMBOL}_daily_rvol_{ts}.csv")

    print("\nCSV files saved in the working directory.")