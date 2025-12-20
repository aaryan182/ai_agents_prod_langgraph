from typing import TypeDict, Dict, Any

class FinanceState(TypeDict):
    ticker: str
    period: str
    raw_data: Any
    indicators: Dict[str, float]
    report: Dict[str, Any]
    error: str
    retries: int
    