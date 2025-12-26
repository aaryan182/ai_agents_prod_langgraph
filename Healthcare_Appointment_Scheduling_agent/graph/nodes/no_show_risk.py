from tools.patient_history import get_no_show_count
from utils.scoring import calculate_no_show_risk
from utils.audit import audit

def no_show_risk_node(state):
    history = get_no_show_count(state["patient_id"])
    state["no_show_risk"] = calculate_no_show_risk(history)
    audit("NO_SHOW_RISK_COMPUTED", state)
    return state