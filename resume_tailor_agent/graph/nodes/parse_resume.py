from tools.pdf_parser import parse_pdf

def parse_resume_node(state, resume_path, jd_path):
    state["resume_text"] = parse_pdf(resume_path)
    state["jd_text"] = parse_pdf(jd_path)
    return state