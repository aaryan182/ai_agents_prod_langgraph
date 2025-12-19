import json
from llm.client import call_llm
from graph.state import TicketState
from memory.store import get_ticket_history


def classifier_agent(state: TicketState):
    history = get_ticket_history(state["ticket_id"])
    
    prompt = f"""
    Classify the support ticket into one category:
    - technical
    - billing
    - general
    
    Also provide confidence score(0-1).
    
    Ticket history: 
    {history}
    
    Current message: 
    {state['message']}
    
    Return JSON:
    {{"category": "...", "confidence": 0.0}}
    """
    response = call_llm(prompt)
    if response is None:
        raise ValueError("call_llm returned None")
    result = json.loads(response)
    state['category'] = result.get('category')
    state['confidence'] = result.get('confidence')
    return state
    