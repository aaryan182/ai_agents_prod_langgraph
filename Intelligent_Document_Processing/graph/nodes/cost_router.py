def cost_router_node(state):
    if len(state["text"]) < 1500:
        state["model_tier"] = "cheap"
    else:
        state["model_tier"] = "expensive"
    return state