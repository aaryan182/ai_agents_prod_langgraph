from llm.client import call_llm

CONFIDENCE_THRESHOLD = 0.75

def validator_node(state):
    prompt = f"""
Evaluate the quality of this answer on a scale 0 to 1.
Consider correctness, completeness, and clarity.

Answer:
{state['answer']}

Return ONLY a number.
"""
    score = call_llm(prompt, model="gpt-4.1-mini")

    try:
        state["confidence"] = float(score)
    except:
        state["confidence"] = 0.0

    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        state["final_answer"] = state["answer"]

    return state