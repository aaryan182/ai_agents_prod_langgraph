from langgraph.graph import StateGraph, END
from graph.state import FinanceState

from graph.nodes.validate_input import validate_input
from graph.nodes.fetch_data import fetch_data
from graph.nodes.quality_check import quality_check
from graph.nodes.analyse import analyze
from graph.nodes.generate_report import generate_report
from graph.nodes.validate_output import validate_output
from graph.nodes.routing_logic import route_after_fetch

def build_graph():
    g = StateGraph(FinanceState)

    g.add_node("validate", validate_input)
    g.add_node("fetch", fetch_data)
    g.add_node("quality", quality_check)
    g.add_node("analyze", analyze)
    g.add_node("report", generate_report)
    g.add_node("validate_output", validate_output)

    g.set_entry_point("validate")

    g.add_edge("validate", "fetch")

    g.add_conditional_edges(
        "fetch",
        route_after_fetch,
        {
            "fetch": "fetch",
            "quality": "quality",
            "end": END
        }
    )

    g.add_edge("quality", "analyze")
    g.add_edge("analyze", "report")
    g.add_edge("report", "validate_output")
    g.add_edge("validate_output", END)

    return g.compile()
