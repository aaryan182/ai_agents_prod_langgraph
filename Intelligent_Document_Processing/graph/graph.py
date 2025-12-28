from langgraph.graph import StateGraph, END
from graph.state import DocumentState

from graph.nodes.intake import intake_node
from graph.nodes.classifier import classifier_node
from graph.nodes.cost_router import cost_router_node
from graph.nodes.extractor_cheap import extractor_cheap_node
from graph.nodes.extractor_expensive import extractor_expensive_node
from graph.nodes.validator import validator_node
from graph.nodes.confidence import confidence_node
from graph.nodes.router import router_node
from graph.nodes.human_review import human_review_node

def build_graph():
    g = StateGraph(DocumentState)

    g.add_node("intake", intake_node)
    g.add_node("classify", classifier_node)
    g.add_node("cost", cost_router_node)
    g.add_node("cheap", extractor_cheap_node)
    g.add_node("expensive", extractor_expensive_node)
    g.add_node("validate", validator_node)
    g.add_node("confidence", confidence_node)
    g.add_node("route", router_node)
    g.add_node("human", human_review_node)

    g.set_entry_point("intake")

    g.add_edge("intake", "classify")
    g.add_edge("classify", "cost")

    g.add_conditional_edges(
        "cost",
        lambda s: s["model_tier"],
        {
            "cheap": "cheap",
            "expensive": "expensive",
        }
    )

    g.add_edge("cheap", "validate")
    g.add_edge("expensive", "validate")
    g.add_edge("validate", "confidence")
    g.add_edge("confidence", "route")

    g.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {
            "human_review": "human",
            "auto_approve": END,
        }
    )

    g.add_edge("human", END)

    return g.compile()
