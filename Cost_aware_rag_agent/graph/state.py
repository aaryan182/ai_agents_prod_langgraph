from typing import TypedDict, Literal

class RAGState(TypedDict, total = False):
    query: str
    estimated_cost: float
    router: str
    retrieved_context: str
    answer: str
    confidence: float
    retries: int
    final_answer: str
    route: Literal["cheap", "expensive"]
    retries: int