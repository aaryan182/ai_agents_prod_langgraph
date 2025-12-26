from langgraph.graph import StateGraph, END
from graph.state import AppointmentState

from graph.nodes.intake import intake_node
from graph.nodes.availability_check import availability_check_node
from graph.nodes.conflict_resolver import conflict_resolver_node
from graph.nodes.no_show_risk import no_show_risk_node
from graph.nodes.rescheduler import rescheduler_node
from graph.nodes.notification_decider import notification_decider_node
from graph.nodes.escalation import escalation_node

def build_graph():
    g = StateGraph(AppointmentState)

    g.add_node("intake", intake_node)
    g.add_node("availability", availability_check_node)
    g.add_node("resolve_conflict", conflict_resolver_node)
    g.add_node("risk", no_show_risk_node)
    g.add_node("reschedule", rescheduler_node)
    g.add_node("notify", notification_decider_node)
    g.add_node("escalate", escalation_node)

    g.set_entry_point("intake")

    g.add_edge("intake", "availability")
    g.add_edge("availability", "resolve_conflict")
    g.add_edge("resolve_conflict", "risk")
    g.add_edge("risk", "reschedule")
    g.add_edge("reschedule", "notify")

    g.add_conditional_edges(
        "notify",
        lambda s: "escalate" if s["escalate_to_human"] else END,
        {
            "escalate": "escalate",
            END: END
        }
    )

    g.add_edge("escalate", END)

    return g.compile()
