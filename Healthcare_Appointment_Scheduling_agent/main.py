"""
Standalone runner for Healthcare Appointment Scheduling Agent.

In production:
- This file is invoked by workers / APIs
- Here it allows local testing & demos
"""

from graph.graph import build_graph

def run_example():
    agent = build_graph()

    result = agent.invoke({
        "patient_id": "PATIENT_123",
        "request": {
            "appointment_time": "26-12-2025T10:00",
            "doctor_id": "DR_42",
            "department": "cardiology"
        },
        "notify_patient": False,
        "escalate_to_human": False,
    })

    print("\n=== FINAL DECISION ===\n")
    print(result["final_decision"])

if __name__ == "__main__":
    run_example()
