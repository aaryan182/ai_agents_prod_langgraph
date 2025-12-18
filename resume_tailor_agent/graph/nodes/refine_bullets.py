from llm.client import call_llm

def refine_bullets_node(state):
    prompt = f"""
Rewrite resume bullet points to:
- Match job description
- Include missing skills where honest
- Be ATS friendly

Missing Skills:
{state['missing_skills']}

Resume:
{state['resume_text']}
"""
    state["refined_bullets"] = call_llm(prompt)
    return state