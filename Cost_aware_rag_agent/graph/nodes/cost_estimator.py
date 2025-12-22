def cost_estimator_node(state):
    """
    Very realistic heuristic:
    - Longer queries
    - Analytical keywords
    - Comparison / reasoning requests
    """
    q = state["query"].lower()
    cost = 0.002 
    
    if len(q) > 120:
        cost += 0.04
    
    complex_keywords = ["compare", "analyze", "why", "tradeoff", "architecture"]
    if any(k in q for k in complex_keywords):
        cost += 0.01
    
    state["estimated_cost"] = cost
    return state


#  In real systems we can replace with token estimation + pricing tables