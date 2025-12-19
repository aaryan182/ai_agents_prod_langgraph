from typing import TypedDict, List

class TicketState(TypedDict):
    ticket_id: str
    message: str
    category: str
    confidence: float
    history: List[str]
    response: str
    resolved: bool
    escalated: bool
