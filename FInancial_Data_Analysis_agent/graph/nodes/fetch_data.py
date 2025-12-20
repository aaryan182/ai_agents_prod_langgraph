from tools.market_data import fetch_stock_data

def fetch_data(state):
    try:
        state["raw_data"] = fetch_stock_data(
            state["ticker"], state["period"]
        )
        state["error"] = ""
    except Exception as e:
        state["error"] = str(e)
        state["retries"] += 1
    return state