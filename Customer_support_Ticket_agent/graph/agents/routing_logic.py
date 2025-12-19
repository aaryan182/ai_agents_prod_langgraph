CONFIDENCE_THRESHOLD = 0.75

def route_ticket(state):
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "human"

    return state["category"]