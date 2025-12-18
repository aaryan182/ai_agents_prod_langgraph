from graph.graph import build_graph

agent = build_graph()

result = agent.invoke({
    "query": "Impact of AI on software engineering productivity"
})

print("\n===== FINAL RESEARCH REPORT =====\n")
print(result["final_report"])