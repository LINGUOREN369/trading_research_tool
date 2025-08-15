def RSI_calculator(data, period=14):
    close = data["Close"].astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def MACD_calculator(data, short_window=12, long_window=26, signal_window=9):
    exp1 = data["Close"].ewm(span=short_window, adjust=False).mean()
    exp2 = data["Close"].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def momentum_calculator(data, window=14):
    momentum = data["Close"].diff(periods=window)
    return momentum

