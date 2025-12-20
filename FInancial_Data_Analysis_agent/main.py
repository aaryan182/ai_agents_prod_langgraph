from graph.graph import build_graph

agent = build_graph()

result = agent.invoke({
    "ticker": "AAPL",
    "period": "6mo",
    "retries": 0
})

print("\n=== FINANCIAL REPORT ===\n")
print(result["report"])
