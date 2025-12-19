from llm.client import call_llm

def billing_agent(state):
    prompt = f"""
    You are a billing support agent.
    Issue:
    {state['message']}
    """
    state['response'] = call_llm(prompt)
    state['resolved'] = True
    return state