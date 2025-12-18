from llm.client import call_llm
from graph.state import ResearchState

def summarizer_agent(state: ResearchState):
    prompt= f"""
    Summarize the following extracted facts into 5-6 concise sentences.
    
    Facts: 
    {state['extracted_facts']}
    """
    state['summary'] = call_llm(prompt)
    return state