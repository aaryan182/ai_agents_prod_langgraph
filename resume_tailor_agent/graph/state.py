from typing import TypedDict, List

class ResumeState(TypedDict):
    resume_text: str
    jd_text: str
    resume_skills: List[str]
    jd_skills: List[str]
    missing_skills: List[str]
    refined_bullets: str
    cover_letter: str
    final_pdf_path: str