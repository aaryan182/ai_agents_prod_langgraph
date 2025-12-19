from llm.client import call_llm

def general_agent(state):
    prompt = f"""
    You are a customer support agent.
    
    Issue: 
    {state['message']}
    """
    state['response'] = call_llm(prompt)
    state['resolved'] = True
    return state