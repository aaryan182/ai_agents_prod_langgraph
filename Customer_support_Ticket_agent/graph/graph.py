from langgraph.graph import StateGraph, END
from graph.state import TicketState

from graph.agents.classifier_agent import classifier_agent
from graph.agents.tech_agent import tech_agent
from graph.agents.billing_agent import billing_agent
from graph.agents.general_agent import general_agent
from graph.agents.human_review_agent import human_review_agent
from memory.store import save_ticket_message
from graph.agents.routing_logic import route_ticket

def build_graph():
    g = StateGraph(TicketState)

    g.add_node("classify", classifier_agent)
    g.add_node("technical", tech_agent)
    g.add_node("billing", billing_agent)
    g.add_node("general", general_agent)
    g.add_node("human", human_review_agent)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        route_ticket,
        {
            "technical": "technical",
            "billing": "billing",
            "general": "general",
            "human": "human"
        }
    )

    g.add_edge("technical", END)
    g.add_edge("billing", END)
    g.add_edge("general", END)
    g.add_edge("human", END)

    return g.compile()