ALLOWED_FIELDS = {
    "appointment_time",
    "doctor_id",
    "department",
}

def validate_request(data: dict):
    for key in data:
        if key not in ALLOWED_FIELDS:
            raise ValueError("HIPAA violation: unauthorized field")