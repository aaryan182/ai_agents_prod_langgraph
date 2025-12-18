from typing import TypedDict, Optional

class ResearchState(TypedDict):
    query: str
    raw_search_results: str
    extracted_facts: Optional[str]
    summary: Optional[str]
    critique: Optional[str]
    final_report: Optional[str]