from tools.pdf_exporter import export_pdf

def export_node(state):
    content = (
        "TAILORED RESUME\n\n"
        + state["refined_bullets"]
        + "\n\nCOVER LETTER\n\n"
        + state["cover_letter"]
    )
    state["final_pdf_path"] = export_pdf(content, "tailored_resume.pdf")
    return state