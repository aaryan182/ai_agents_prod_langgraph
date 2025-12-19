from llm.client import call_llm

def tech_agent(state):
    prompt = f"""
    You are a technical support agent.
    Solve this issue clearly.
    
    Issue: 
    {state['message']}
    """
    state['response'] = call_llm(prompt)
    state['resolved'] = True
    return state