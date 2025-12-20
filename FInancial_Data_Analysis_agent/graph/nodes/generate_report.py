def generate_report(state):
    ind = state["indicators"]

    signal = "neutral"
    if ind["rsi"] < 30:
        signal = "oversold"
    elif ind["rsi"] > 70:
        signal = "overbought"

    state["report"] = {
        "ticker": state["ticker"],
        "signal": signal,
        "indicators": ind,
        "summary": (
            f"RSI suggests {signal} conditions. "
            "MACD and moving averages provide trend context."
        )
    }
    return state
