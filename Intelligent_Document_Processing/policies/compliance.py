REQUIRED_FIELDS = {
    "invoice": {"invoice_number", "amount", "date"},
    "contract": {"party_a", "party_b", "effective_date"},
}

def validate_required_fields(doc_type: str, data: dict):
    required = REQUIRED_FIELDS.get(doc_type, set())
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")