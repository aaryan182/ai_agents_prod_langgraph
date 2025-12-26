def audit(event: str, state):
    print(f"[AUDIT] {event} | patient={state['patient_id']}")