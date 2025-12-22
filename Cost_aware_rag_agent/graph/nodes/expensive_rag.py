from llm.client import call_llm
from rag.retriever import retrieve_context

def expensive_rag_node(state):
    ctx = retrieve_context(state["query"])
    prompt = f"""
Provide a detailed, well-reasoned answer using the context.

Context:
{ctx}

Question:
{state['query']}
"""
    state["retrieved_context"] = ctx
    state["answer"] = call_llm(
        prompt,
        model="gpt-4.1",  
    )
    return state