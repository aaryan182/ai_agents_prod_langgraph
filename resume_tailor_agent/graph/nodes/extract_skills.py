from llm.client import call_llm

def extract_skills_node(state):
    prompt = f"""
    Extract technical skills as a JSON list.
    
    Resume:
    {state['resume_text']}
    
    Job Description: 
    {state['jd_text']}
    """
    skills = call_llm(prompt)
    if skills:
        state['resume_skills'] = eval(skills.split("\n")[0])
        state['jd_skills'] = eval(skills.split("\n")[0])
    return state