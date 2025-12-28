from tools.text_parser import extract_text
from utils.audit import audit

def intake_node(state):
    state['text'] = extract_text(state['raw_document'])
    audit("DOCUMENT_INGESTED", state)
    return state