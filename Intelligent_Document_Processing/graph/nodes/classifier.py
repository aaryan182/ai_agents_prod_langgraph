from llm.client import call_llm
from utils.audit import audit

def classifier_node(state):
    prompt = f"""
Classify document type:
invoice, contract, medical_record, insurance_claim

Text:
{state['text'][:1000]}
"""
    state["doc_type"] = call_llm(prompt).strip()
    audit("DOCUMENT_CLASSIFIED", state)
    return state