import ta

def compute_indicators(df):
    return {
        "rsi": ta.momentum.RSIIndicator(df["Close"]).rsi().iloc[-1],
        "macd": ta.trend.MACD(df["Close"]).macd().iloc[-1],
        "sma_50": df["Close"].rolling(50).mean().iloc[-1],
        "sma_200": df["Close"].rolling(200).mean().iloc[-1]
    }