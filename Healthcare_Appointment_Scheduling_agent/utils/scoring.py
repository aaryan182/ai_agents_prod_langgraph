def calculate_no_show_risk(previous_no_shows: int) -> float:
    if previous_no_shows >= 3:
        return 0.85
    if previous_no_shows == 2:
        return 0.6
    return 0.2