from policies.hippa import validate_request
from utils.audit import audit

def intake_node(state):
    validate_request(state['request'])
    state['requested_time'] = state['request']['appointment_time']
    state['doctor_id'] = state['request']['doctor_id']
    audit("INTAKE_VALIDATED", state)
    return state