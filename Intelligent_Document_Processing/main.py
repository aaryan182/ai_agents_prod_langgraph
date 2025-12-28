from graph.graph import build_graph

agent = build_graph()

result = agent.invoke({
    "document_id": "DOC_001",
    "raw_document": b"Invoice #123 Amount $450 Date 2025-09-01"
})

print("\nFINAL STATUS:")
print(result["final_status"])
print("CONFIDENCE:", result["confidence"])
