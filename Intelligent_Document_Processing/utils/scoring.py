def calculate_confidence(errors: int) -> float:
    if errors == 0:
        return 0.96
    if errors == 1:
        return 0.85
    return 0.6