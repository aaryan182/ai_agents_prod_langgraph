def router_node(state):
    if state["confidence"] >= 0.9:
        state["route"] = "auto_approve"
        state["final_status"] = "Processed automatically"
    else:
        state["route"] = "human_review"
    return state
