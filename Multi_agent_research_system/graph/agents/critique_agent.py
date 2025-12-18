from llm.client import call_llm
from graph.state import ResearchState

def critique_agent(state: ResearchState):
    prompt = f"""
    Critically evaluate the following summary.
    
    Check for: 
    - Missing perspectives
    - Bias
    - Weak reasoning
    - Unclear claims
    
    Provide short critique and improvement suggestions.
    
    Summary: 
    {state['summary']}
    """
    state['critique'] = call_llm(prompt)
    return state