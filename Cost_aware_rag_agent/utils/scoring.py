"""
Centralized scoring, validation and routing logic.

Why this file exists:
- Keeps business logic OUT of graph wiring
- Makes retry & confidence policies reusable
- Allows easy tuning without touching agents
"""

from typing import Literal
from graph.state import RAGState

CONFIDENCE_THRESHOLD = 0.75
MAX_RETRIES = 1

# -----------------------------
# VALIDATION ROUTER
# -----------------------------

def validation_router(state: RAGState) -> Literal["expensive", "end"]:
    """
    Decide what to do AFTER answer validation.

    Returns:
    - "end"        → accept answer, stop graph
    - "expensive" → re-route to expensive model

    This function mutates state safely when retrying.
    """

    # Case 1: Answer is good enough → finish
    if state.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD:
        return "end"

    # Case 2: Cheap model failed, retry with expensive model
    if (
        state.get("route") == "cheap"
        and state.get("retries", 0) < MAX_RETRIES
    ):
        state["route"] = "expensive"
        state["retries"] = state.get("retries", 0) + 1
        return "expensive"

    # Case 3: Already expensive OR retries exhausted
    return "end"


# -----------------------------
# OPTIONAL: CONFIDENCE NORMALIZER
# -----------------------------

def normalize_confidence(raw_score: float) -> float:
    """
    Guardrail to keep confidence in [0, 1].
    Prevents bad LLM outputs from breaking routing.
    """
    try:
        return max(0.0, min(1.0, float(raw_score)))
    except Exception:
        return 0.0
