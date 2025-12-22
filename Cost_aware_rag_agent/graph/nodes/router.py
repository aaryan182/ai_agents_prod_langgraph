MAX_CHEAP_COST = 0.005

def router_node(state):
    if state["estimated_cost"] <= MAX_CHEAP_COST:
        state["route"] = "cheap"
    else:
        state['route'] = 'expensive'
    return state