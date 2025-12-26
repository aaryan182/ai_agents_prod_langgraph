def escalation_node(state):
    state["final_decision"] = "Escalated to scheduling staff"
    return state