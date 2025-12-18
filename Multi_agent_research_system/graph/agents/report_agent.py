from llm.client import call_llm
from graph.state import ResearchState

def report_agent(state: ResearchState):
    prompt = f"""
Generate a professional research report using:

Summary:
{state['summary']}

Critique:
{state['critique']}

The report should include:
- Title
- Executive summary
- Key findings
- Limitations
- Conclusion
"""
    state["final_report"] = call_llm(prompt)
    return state