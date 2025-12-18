def gap_analysis_node(state):
    state["missing_skills"] = list(
        set(state["jd_skills"]) - set(state["resume_skills"])
    )
    return state