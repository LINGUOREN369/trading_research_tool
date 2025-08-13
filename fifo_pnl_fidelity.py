import pandas as pd
import yfinance as yf
from collections import deque

def fifo_pnl_with_live_price(csv_path: str, symbol: str, expected_price: float | None = None):
    """
    Calculate FIFO realized/unrealized P&L for a given stock symbol from the Fidelity report.
    Uses trade prices from CSV for realized P&L and fetches current market price via Yahoo Finance.

    Call the function with:
    fifo_pnl_with_live_price("path/to/your/csv_file.csv", "stock_symbol", expected_price=200.0)
    
    Parameters
    ----------
    csv_path : str
        Path to your CSV file with trade history.
    symbol : str
        Stock ticker symbol (e.g., "CRCL").
    expected_price : float | None
        Optional scenario price per share to evaluate potential unrealized P&L. If provided,
        additional fields are returned showing P&L at this price. If None, the scenario fields
        mirror the live-price fields.

    Returns
    -------
    dict
        Summary with open quantity, avg cost, realized P&L, unrealized P&L, total P&L at live price,
        and scenario values at `expected_price` when supplied:
        {
            'symbol', 'current_price', 'open_qty', 'avg_cost', 'market_value',
            'realized_pl', 'unrealized_pl', 'total_pl',
            'expected_price', 'scenario_market_value', 'scenario_unrealized_pl', 'scenario_total_pl'
        }
    """
    # ---------------- Load CSV ---------------- #
    df = pd.read_csv(csv_path)

    # Normalize date column name
    if "Run Date" in df.columns:
        df = df.rename(columns={"Run Date": "Date"})
    if "Price ($)" in df.columns:
        df = df.rename(columns={"Price ($)": "Price"})

    # Filter for symbol
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Symbol"].astype(str).str.upper() == symbol.upper()].copy()

    # Make sells negative if Action exists
    if "Action" in df.columns and df["Quantity"].ge(0).all():
        sell_mask = df["Action"].astype(str).str.contains("sell", case=False, na=False)
        df.loc[sell_mask, "Quantity"] *= -1

    # Ensure numeric
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------- FIFO Calculation ---------------- #
    fifo_lots = deque()
    realized_pl = 0.0

    for _, row in df.iterrows():
        qty = float(row["Quantity"])
        price = float(row["Price"])

        if qty > 0:
            # Buy
            fifo_lots.append([qty, price])
        elif qty < 0:
            # Sell
            sell_left = -qty
            while sell_left > 0 and fifo_lots:
                lot_qty, lot_price = fifo_lots[0]
                take = min(lot_qty, sell_left)
                realized_pl += (price - lot_price) * take
                lot_qty -= take
                sell_left -= take
                if lot_qty == 0:
                    fifo_lots.popleft()
                else:
                    fifo_lots[0][0] = lot_qty

    # ---------------- Fetch Live Price ---------------- #
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d", interval="1m")
    if not hist.empty and "Close" in hist:
        current_price = float(hist["Close"].iloc[-1])
    else:
        # Fallback to last daily close
        current_price = float(ticker.history(period="5d")["Close"].iloc[-1])

    # ---------------- Calculate Open Position Stats (live) ---------------- #
    open_qty = sum(q for q, _ in fifo_lots)
    open_cost = sum(q * p for q, p in fifo_lots)
    avg_cost = open_cost / open_qty if open_qty else 0.0

    market_value = open_qty * current_price
    unrealized_pl = market_value - open_cost
    total_pl = realized_pl + unrealized_pl

    # ---------------- Scenario metrics at expected_price ---------------- #
    scenario_price = expected_price if expected_price is not None else current_price
    scenario_market_value = open_qty * scenario_price
    scenario_unrealized_pl = scenario_market_value - open_cost
    scenario_total_pl = realized_pl + scenario_unrealized_pl

    print(f"\n--- Based on current price: ${current_price:.2f} ---")
    print(f"Symbol: {symbol.upper()}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Open Quantity: {open_qty}")
    print(f"Average Cost: ${avg_cost:.2f}")
    print(f"Market Value: ${market_value:.2f}")
    print(f"Realized P&L: ${realized_pl:.2f}")
    print(f"Unrealized P&L: ${unrealized_pl:.2f}")
    print(f"Total P&L: ${total_pl:.2f}")
    if expected_price is not None:
        print(f"\n--- Scenario at Expected Price: ${expected_price:.2f} ---")
        print(f"Scenario Market Value: ${scenario_market_value:.2f}")
        print(f"Scenario Realized P&L: ${realized_pl:.2f}")
        print(f"Scenario Unrealized P&L: ${scenario_unrealized_pl:.2f}")
        print(f"Scenario Total P&L: ${scenario_total_pl:.2f}")

    print("\n")

    return {
        "symbol": symbol.upper(),
        "current_price": current_price,
        "open_qty": open_qty,
        "avg_cost": avg_cost,
        "market_value": market_value,
        "realized_pl": realized_pl,
        "unrealized_pl": unrealized_pl,
        "total_pl": total_pl,
        # Scenario fields
        "expected_price": expected_price,
        "scenario_market_value": scenario_market_value,
        "scenario_unrealized_pl": scenario_unrealized_pl,
        "scenario_total_pl": scenario_total_pl,
    }
    
    