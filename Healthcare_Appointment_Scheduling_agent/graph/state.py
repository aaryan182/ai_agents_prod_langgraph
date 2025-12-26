from typing import TypedDict, Optional

class AppointmentState(TypedDict):
    patient_id: str
    request: dict
    
    requested_time: str
    doctor_id: str
    
    available: bool
    conflict_reason: Optional[str]
    
    no_show_risk: float
    rescheduled_time: Optional[str]
    
    notify_patient: bool
    escalate_to_human:bool
    
    final_decision: str