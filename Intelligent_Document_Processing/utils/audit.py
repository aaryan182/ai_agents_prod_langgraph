import datetime

def audit(event: str, state):
    ts = datetime.datetime.utcnow().isoformat()
    print(f"[AUDIT] {ts} | {event} | doc={state['document_id']}")
    