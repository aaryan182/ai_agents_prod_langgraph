def validate_output(state):
    report = state["report"]

    required = {"ticker", "signal", "indicators", "summary"}
    if not report or not required.issubset(report):
        state["error"] = "Invalid report structure"
    return state
