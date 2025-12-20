def validate_input(state):
    if not state["ticker"].isalpha():
        state["error"] = "Invalid ticker format"
    if state["period"] not in {"1mo", "3mo", "6mo", "1y", "5y"}:
        state["error"] = "Invalid period"
    return state