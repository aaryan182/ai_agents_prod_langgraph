from llm.client import call_llm
from rag.retriever import retrieve_context

def cheap_rag_node(state):
    ctx = retrieve_context(state["query"])
    prompt = f"""
    Answer using the context below concisely.
    
    Context: 
    {ctx}
    
    Question: 
    {state['query']}
    """
    state["retrieved_context"] = ctx
    state['answer'] = call_llm(
        prompt,
        model="gpt-4.1-mini"
    )
    return state