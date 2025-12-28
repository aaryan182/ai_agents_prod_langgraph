from llm.client import call_llm

def extractor_cheap_node(state):
    prompt = f"""
Extract key fields from this {state['doc_type']}:

{state['text']}
Return JSON only.
"""
    state["extracted_data"] = eval(call_llm(prompt))
    return state