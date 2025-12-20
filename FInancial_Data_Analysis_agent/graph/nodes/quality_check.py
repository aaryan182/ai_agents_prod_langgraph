def quality_check(state):
    df = state["raw_data"]

    if df is None or len(df) < 50:
        state["error"] = "Insufficient data for analysis"
    return state