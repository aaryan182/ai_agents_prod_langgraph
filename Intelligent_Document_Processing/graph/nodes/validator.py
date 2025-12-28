from policies.compliance import validate_required_fields

def validator_node(state):
    try:
        validate_required_fields(
            state["doc_type"],
            state["extracted_data"]
        )
        state["validation_errors"] = None
    except Exception as e:
        state["validation_errors"] = str(e)
    return state
