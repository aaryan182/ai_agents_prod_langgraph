from graph.graph import build_graph
from memory.store import save_ticket_message

agent = build_graph()

ticket_id = "TICKET-001"

message = "My payment failed but money was deducted"

save_ticket_message(ticket_id, message)

result = agent.invoke({
    "ticket_id": ticket_id,
    "message": message,
    "resolved": False,
    "escalated": False
})

print("\n=== SUPPORT RESPONSE ===\n")
print(result["response"])