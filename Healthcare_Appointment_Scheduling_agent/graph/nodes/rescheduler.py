def rescheduler_node(state):
    if state["no_show_risk"] > 0.7:
        state["rescheduled_time"] = "27-12-2025T09:00"
        state["final_decision"] = "Rescheduled to reduce no show"
    else:
        state["final_decision"] = "Appointment confirmed"

    return state