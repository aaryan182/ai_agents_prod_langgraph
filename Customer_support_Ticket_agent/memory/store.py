_ticket_db={}

def get_ticket_history(ticket_id: str):
    return _ticket_db.get(ticket_id, [])

def save_ticket_message(ticket_id: str, message: str):
    history = _ticket_db.get(ticket_id, [])
    history.append(message)
    _ticket_db[ticket_id] = history