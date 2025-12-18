from langgraph.graph import StateGraph, END
from graph.state import ResearchState

from graph.agents.search_agent import search_agent
from graph.agents.extractor_agent import extractor_agent
from graph.agents.summarizer_agent import summarizer_agent
from graph.agents.critique_agent import critique_agent
from graph.agents.report_agent import report_agent

def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("search", search_agent)
    g.add_node("extract", extractor_agent)
    g.add_node("summarize", summarizer_agent)
    g.add_node("critique", critique_agent)
    g.add_node("report", report_agent)

    g.set_entry_point("search")

    g.add_edge("search", "extract")
    g.add_edge("extract", "summarize")
    g.add_edge("summarize", "critique")
    g.add_edge("critique", "report")
    g.add_edge("report", END)

    return g.compile()