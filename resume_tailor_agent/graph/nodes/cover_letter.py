from llm.client import call_llm

def cover_letter_node(state):
    prompt = f"""
Write a personalized cover letter based on:

Job Description:
{state['jd_text']}

Refined Resume:
{state['refined_bullets']}
"""
    state["cover_letter"] = call_llm(prompt)
    return state