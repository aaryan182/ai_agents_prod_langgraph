from utils.scoring import calculate_confidence

def confidence_node(state):
    errors = 0 if state["validation_errors"] is None else 1
    state["confidence"] = calculate_confidence(errors)
    return state
