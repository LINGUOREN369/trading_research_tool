#!/usr/bin/env python3
"""
Fundamental Data Collector (FMP)
--------------------------------
Enter a stock symbol and fetch:
  • Profile (name, sector, market cap, price)
  • Income Statement (annual + quarterly)
  • Balance Sheet (annual + quarterly)
  • Cash Flow (annual + quarterly)
  • Ratios & Key Metrics (TTM)
  • Enterprise Values (for EV-based multiples)
  • Earnings surprises (history) & next earnings date

Outputs:
  • Pretty printed console summary
  • CSV files per statement & period (optional with --export)
  • A compact JSON snapshot with key metrics (optional)

Usage:
  python fundamental_data_collector.py CRCL --export csv json
  python fundamental_data_collector.py AAPL

Requires env var FMP_API_KEY.
"""

import os
import sys
import json
import argparse
import datetime as dt
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

API_KEY = "zFGxDx9DcfEF9RT8TTFpcT4TNJsOWeqW"  # set this in your env
BASE = "https://financialmodelingprep.com/api/v3"
TIMEOUT = 25

# -----------------------------
# HTTP helpers
# -----------------------------

def get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if not API_KEY:
        sys.exit("Error: FMP_API_KEY is not set in environment.")
    params = params.copy() if params else {}
    params["apikey"] = API_KEY
    url = f"{BASE}/{path}"
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"Failed to decode JSON from {url}: {e}")
    return data

# -----------------------------
# FMP endpoints wrappers
# -----------------------------

def profile(symbol: str) -> Dict[str, Any]:
    data = get_json(f"profile/{symbol}")
    return data[0] if isinstance(data, list) and data else {}


def income_statement(symbol: str, period: str) -> List[Dict[str, Any]]:
    assert period in ("annual", "quarter"), "period must be 'annual' or 'quarter'"
    return get_json(f"income-statement/{symbol}", {"period": period}) or []


def balance_sheet(symbol: str, period: str) -> List[Dict[str, Any]]:
    assert period in ("annual", "quarter"), "period must be 'annual' or 'quarter'"
    return get_json(f"balance-sheet-statement/{symbol}", {"period": period}) or []


def cash_flow(symbol: str, period: str) -> List[Dict[str, Any]]:
    assert period in ("annual", "quarter"), "period must be 'annual' or 'quarter'"
    return get_json(f"cash-flow-statement/{symbol}", {"period": period}) or []


def ratios_ttm(symbol: str) -> Dict[str, Any]:
    data = get_json(f"ratios-ttm/{symbol}")
    return data[0] if isinstance(data, list) and data else {}


def key_metrics_ttm(symbol: str) -> Dict[str, Any]:
    data = get_json(f"key-metrics-ttm/{symbol}")
    return data[0] if isinstance(data, list) and data else {}


def enterprise_values(symbol: str, period: str = "annual") -> List[Dict[str, Any]]:
    assert period in ("annual", "quarter"), "period must be 'annual' or 'quarter'"
    return get_json(f"enterprise-values/{symbol}", {"period": period}) or []


def earnings_surprises(symbol: str) -> List[Dict[str, Any]]:
    return get_json(f"earnings-surprises/{symbol}") or []


def next_earnings(symbol: str) -> Optional[Dict[str, Any]]:
    cal = get_json("earning_calendar", {"symbol": symbol})
    return cal[0] if isinstance(cal, list) and cal else None

# -----------------------------
# Computations / utility
# -----------------------------

def pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b in (None, 0):
            return None
        return (a - b) / b * 100.0
    except Exception:
        return None


def latest(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return docs[0] if docs else {}


def df_from(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    # Normalize date column if present
    for col in ("date", "fillingDate", "period"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def compute_snapshot(sym: str) -> Dict[str, Any]:
    prof = profile(sym)
    inc_q = income_statement(sym, "quarter")
    inc_a = income_statement(sym, "annual")
    bs_q = balance_sheet(sym, "quarter")
    cf_q = cash_flow(sym, "quarter")
    ratio = ratios_ttm(sym)
    km_ttm = key_metrics_ttm(sym)
    evs = enterprise_values(sym, "annual")
    nxt = next_earnings(sym)
    esurp = earnings_surprises(sym)

    latest_inc_q = latest(inc_q)
    prev_inc_q = inc_q[1] if len(inc_q) > 1 else {}
    latest_inc_a = latest(inc_a)

    # Growth & margins (quarterly)
    rev_q = latest_inc_q.get("revenue")
    rev_q_prev = prev_inc_q.get("revenue")
    net_q = latest_inc_q.get("netIncome")
    op_q = latest_inc_q.get("operatingIncome")

    q_rev_yoy = pct(rev_q, rev_q_prev)
    q_op_margin = (op_q / rev_q * 100.0) if rev_q else None
    q_net_margin = (net_q / rev_q * 100.0) if rev_q else None

    # Balance sheet items for net cash
    latest_bs_q = latest(bs_q)
    cash = latest_bs_q.get("cashAndCashEquivalents")
    debt = latest_bs_q.get("totalDebt")
    net_cash = (cash - debt) if (cash is not None and debt is not None) else None

    # Cash flow & FCF
    latest_cf_q = latest(cf_q)
    cfo = latest_cf_q.get("netCashProvidedByOperatingActivities")
    capex = latest_cf_q.get("capitalExpenditure")
    fcf = (cfo + capex) if (cfo is not None and capex is not None) else None  # capex is usually negative
    fcf_margin = (fcf / rev_q * 100.0) if (fcf is not None and rev_q) else None

    # Valuation multiples (TTM/EV)
    pe_ttm = ratio.get("priceEarningsRatioTTM")
    ps_ttm = ratio.get("priceToSalesRatioTTM")
    ev_latest = latest(evs)
    ev = ev_latest.get("enterpriseValue") if ev_latest else None

    price = prof.get("price")
    mktcap = prof.get("mktCap")

    snapshot = {
        "symbol": sym,
        "companyName": prof.get("companyName"),
        "sector": prof.get("sector"),
        "industry": prof.get("industry"),
        "price": price,
        "marketCap": mktcap,
        "revenue_q": rev_q,
        "revenue_q_prev": rev_q_prev,
        "q_rev_yoy_pct": q_rev_yoy,
        "operating_margin_q_pct": q_op_margin,
        "net_margin_q_pct": q_net_margin,
        "cash": cash,
        "totalDebt": debt,
        "netCash": net_cash,
        "cfo_q": cfo,
        "capex_q": capex,
        "fcf_q": fcf,
        "fcf_margin_q_pct": fcf_margin,
        "pe_ttm": pe_ttm,
        "ps_ttm": ps_ttm,
        "enterpriseValue": ev,
        "nextEarnings": nxt.get("date") if nxt else None,
        "surprises": esurp[:6],  # last ~6 prints
        "asOf": dt.datetime.utcnow().isoformat() + "Z",
    }
    return snapshot


def print_summary(s: Dict[str, Any]) -> None:
    def fmt(x, d="–"):
        if x is None:
            return d
        if isinstance(x, float):
            # large currency values
            return f"{x:,.2f}"
        return x

    print("\n=== Snapshot ===")
    print(f"Symbol:           {s.get('symbol')}")
    print(f"Name / Sector:    {s.get('companyName')} / {s.get('sector')} ({s.get('industry')})")
    print(f"Price / Mkt Cap:  {fmt(s.get('price'))} / {fmt(s.get('marketCap'))}")
    print("--- Growth & Margins (Quarterly) ---")
    print(f"Revenue (Q):      {fmt(s.get('revenue_q'))}  | YoY: {fmt(s.get('q_rev_yoy_pct'))}%")
    print(f"Op Margin (Q):    {fmt(s.get('operating_margin_q_pct'))}%")
    print(f"Net Margin (Q):   {fmt(s.get('net_margin_q_pct'))}%")
    print("--- Cash & FCF ---")
    print(f"Cash / Debt:      {fmt(s.get('cash'))} / {fmt(s.get('totalDebt'))}")
    print(f"Net Cash:         {fmt(s.get('netCash'))}")
    print(f"CFO / Capex (Q):  {fmt(s.get('cfo_q'))} / {fmt(s.get('capex_q'))}")
    print(f"FCF (Q):          {fmt(s.get('fcf_q'))}  | FCF Margin: {fmt(s.get('fcf_margin_q_pct'))}%")
    print("--- Valuation ---")
    print(f"P/E (TTM):        {fmt(s.get('pe_ttm'))}")
    print(f"P/S (TTM):        {fmt(s.get('ps_ttm'))}")
    print(f"Enterprise Value: {fmt(s.get('enterpriseValue'))}")
    print("--- Earnings ---")
    print(f"Next Earnings:    {s.get('nextEarnings') or 'n/a'}")
    if s.get("surprises"):
        last = s["surprises"][0]
        beat = (last.get("surprise") or 0) if isinstance(last, dict) else None
        print("Last Surprise:   ", fmt(beat))
    print()


# -----------------------------
# Export helpers
# -----------------------------

def export_statements(symbol: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for period in ("annual", "quarter"):
        inc = df_from(income_statement(symbol, period))
        bs = df_from(balance_sheet(symbol, period))
        cf = df_from(cash_flow(symbol, period))
        if not inc.empty:
            inc.to_csv(os.path.join(outdir, f"{symbol}_income_{period}.csv"), index=False)
        if not bs.empty:
            bs.to_csv(os.path.join(outdir, f"{symbol}_balance_{period}.csv"), index=False)
        if not cf.empty:
            cf.to_csv(os.path.join(outdir, f"{symbol}_cashflow_{period}.csv"), index=False)


def export_snapshot_json(symbol: str, snapshot: Dict[str, Any], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{symbol}_snapshot.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch fundamental data & compute key metrics from FMP.")
    parser.add_argument("symbol", help="Ticker symbol, e.g., CRCL")
    parser.add_argument("--export", nargs="*", choices=["csv", "json"], default=[],
                        help="Export statements to CSV and/or a compact JSON snapshot")
    parser.add_argument("--out", default="fundamentals_out", help="Output directory for exports")
    args = parser.parse_args()

    sym = args.symbol.upper()

    try:
        snap = compute_snapshot(sym)
    except requests.HTTPError as e:
        sys.exit(f"HTTP error: {e}")
    except Exception as e:
        sys.exit(f"Error: {e}")

    print_summary(snap)

    if args.export:
        if "csv" in args.export:
            export_statements(sym, args.out)
        if "json" in args.export:
            export_snapshot_json(sym, snap, args.out)
        print(f"Exports written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()