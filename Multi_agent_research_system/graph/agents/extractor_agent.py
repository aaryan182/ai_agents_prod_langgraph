from llm.client import call_llm
from graph.state import ResearchState

def extractor_agent(state: ResearchState):
    prompt = f"""
    Extract factual points, statistics and key claims from the following content.
    Return bullent points only.
    
    Content: 
    {state['raw_search_results']}
    """
    state['extracted_facts'] = call_llm(prompt)
    return state