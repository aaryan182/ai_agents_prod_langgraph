from typing import TypedDict, Optional, Dict, Any

class DocumentState(TypedDict):
    document_id: str
    raw_document: bytes
    
    text: str
    doc_type: str
    
    model_tier: str
    
    extracted_data: Dict[str, Any]
    validation_errors: Optional[str]
    
    confidence: float
    
    route: str
    final_status: str