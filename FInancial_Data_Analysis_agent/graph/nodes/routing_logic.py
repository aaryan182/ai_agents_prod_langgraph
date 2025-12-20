MAX_RETRIES = 2

def route_after_fetch(state):
    if state["error"] and state["retries"] <= MAX_RETRIES:
        return "fetch"
    if state["error"]:
        return "end"
    return "quality"
