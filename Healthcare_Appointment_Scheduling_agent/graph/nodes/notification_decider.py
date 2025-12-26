def notification_decider_node(state):
    state["notify_patient"] = True
    state["escalate_to_human"] = state["no_show_risk"] > 0.9
    return state