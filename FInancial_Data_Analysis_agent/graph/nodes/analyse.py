from utils.indicators import compute_indicators

def analyze(state):
    state["indicators"] = compute_indicators(state["raw_data"])
    return state
