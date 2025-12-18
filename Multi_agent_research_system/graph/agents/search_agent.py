from tools.search_tool import web_search
from graph.state import ResearchState

def search_agent(state: ResearchState):
    state["raw_search_results"] = web_search(state['query'])
    return state