from tools.calendar import is_slot_available
from utils.audit import audit

def availability_check_node(state):
    state['available'] = is_slot_available(
        state['doctor_id'],
        state['requested_time']
    )
    
    if not state['available']:
        state['conflict_reason'] = 'doctor unavailable'
        
    audit("AVAILABILITY_CHECKED", state)
    return state