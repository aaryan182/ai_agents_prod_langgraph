def human_review_agent(state):
    state["response"] = (
        "This ticket requires human review. "
        "A support representative will contact you shortly."
    )
    state["escalated"] = True
    return state