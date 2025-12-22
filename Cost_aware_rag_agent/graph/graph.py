from langgraph.graph import StateGraph, END
from graph.state import RAGState

from graph.nodes.cost_estimator import cost_estimator_node
from graph.nodes.router import router_node
from graph.nodes.cheap_rag import cheap_rag_node
from graph.nodes.expensive_rag import expensive_rag_node
from graph.nodes.validator import validator_node
from utils.scoring import validation_router

def build_graph():
    g = StateGraph(RAGState)

    g.add_node("cost", cost_estimator_node)
    g.add_node("route", router_node)
    g.add_node("cheap", cheap_rag_node)
    g.add_node("expensive", expensive_rag_node)
    g.add_node("validate", validator_node)

    g.set_entry_point("cost")

    g.add_edge("cost", "route")

    g.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {
            "cheap": "cheap",
            "expensive": "expensive",
        }
    )

    g.add_edge("cheap", "validate")
    g.add_edge("expensive", "validate")

    g.add_conditional_edges(
        "validate",
        validation_router,
        {
            "expensive": "expensive",
            "end": END,
        }
    )

    return g.compile()
