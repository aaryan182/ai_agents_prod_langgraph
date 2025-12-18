from graph.graph import build_graph

agent = build_graph()

result = agent.invoke({
    "resume_path": "resume.pdf",
    "jd_path": "job_description.pdf"
})

print("PDF Generated:", result["final_pdf_path"])