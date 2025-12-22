from graph.graph import build_graph

agent = build_graph()

result = agent.invoke({
    "query": "Compare Redis vs PostgreSQL for vector search",
    "retries": 0
})

print("\n=== FINAL ANSWER ===\n")
print(result["final_answer"])
print("\nConfidence:", result["confidence"])
print("Estimated cost:", result["estimated_cost"])