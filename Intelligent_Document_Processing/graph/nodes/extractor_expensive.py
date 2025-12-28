from llm.client import call_llm

def extractor_expensive_node(state):
    prompt = f"""
Carefully extract ALL required structured fields from this {state['doc_type']}.

{state['text']}
Return strict JSON.
"""
    state["extracted_data"] = eval(
        call_llm(prompt, model="gpt-4.1")
    )
    return state
