from langgraph.graph import StateGraph, END
from graph.state import ResumeState
from graph.nodes.parse_resume import parse_resume_node
from graph.nodes.extract_skills import extract_skills_node
from graph.nodes.gap_analysis import gap_analysis_node
from graph.nodes.refine_bullets import refine_bullets_node
from graph.nodes.cover_letter import cover_letter_node
from graph.nodes.export_node import export_node

def build_graph():
    g = StateGraph(ResumeState)

    g.add_node("parse", parse_resume_node)
    g.add_node("skills", extract_skills_node)
    g.add_node("gap", gap_analysis_node)
    g.add_node("refine", refine_bullets_node)
    g.add_node("cover", cover_letter_node)
    g.add_node("export", export_node)

    g.set_entry_point("parse")

    g.add_edge("parse", "skills")
    g.add_edge("skills", "gap")
    g.add_edge("gap", "refine")
    g.add_edge("refine", "cover")
    g.add_edge("cover", "export")
    g.add_edge("export", END)

    return g.compile()