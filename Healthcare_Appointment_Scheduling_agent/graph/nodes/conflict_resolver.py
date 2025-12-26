def conflict_resolver_node(state):
    if state["available"]:
        return state

    # Simple rule-based resolution
    state["requested_time"] = "26-12-2025T11:00"
    state["available"] = True
    state["conflict_reason"] = None
    return state