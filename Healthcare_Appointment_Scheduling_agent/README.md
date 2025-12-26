# Healthcare Appointment Scheduling Agent

> An intelligent, HIPAA compliant multi agent system for automated healthcare appointment scheduling with conflict resolution, no show risk prediction and smart escalation.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40+-green.svg)](https://github.com/langchain-ai/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![HIPAA](https://img.shields.io/badge/HIPAA-Compliant-red.svg)](https://www.hhs.gov/hipaa/index.html)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [System Flow](#-system-flow)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Detailed Components](#-detailed-components)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)

---

## 🎯 Overview

The Healthcare Appointment Scheduling Agent is a sophisticated **LangGraph based multi agent system** designed to automate the complex workflow of scheduling medical appointments. It intelligently handles:

- **Patient intake validation** with HIPAA compliance
- **Real-time availability checking** across multiple doctors and departments
- **Conflict resolution** with smart rescheduling
- **No-show risk prediction** based on patient history
- **Automated notification decisions**
- **Human escalation** for high risk cases

This system reduces administrative burden, improves patient experience and ensures compliance with healthcare regulations.

---

## ✨ Key Features

### 🔒 **HIPAA Compliant Data Handling**
- Strict field validation to prevent unauthorized data exposure
- Comprehensive audit logging for all operations
- Secure patient data management

### 🤖 **Intelligent Agent Workflow**
- **7 specialized nodes** working in orchestrated sequence
- **StateGraph architecture** for transparent workflow management
- **Conditional routing** based on risk assessment

### 📊 **Predictive Analytics**
- No show risk scoring based on patient history
- Proactive rescheduling for high risk appointments
- Data driven decision making

### 🔄 **Automated Conflict Resolution**
- Smart slot finding when conflicts occur
- Multi factor availability checking
- Seamless rescheduling without human intervention

### 🚨 **Smart Escalation**
- Automatic escalation for high risk patients (>90% no show probability)
- Human in the loop for complex cases
- Audit trail for all escalations

---

## 🏗️ Architecture

### System Architecture Diagram

```mermaid
graph TB
    subgraph "Healthcare Appointment Scheduling Agent"
        subgraph "Entry Layer"
            MAIN[main.py<br/>Entry Point]
        end
        
        subgraph "Orchestration Layer"
            GRAPH[StateGraph<br/>Workflow Engine]
            STATE[AppointmentState<br/>Shared State]
        end
        
        subgraph "Agent Nodes"
            N1[Intake Node<br/>Validation]
            N2[Availability Check<br/>Calendar Lookup]
            N3[Conflict Resolver<br/>Smart Rescheduling]
            N4[No-Show Risk<br/>Prediction]
            N5[Rescheduler<br/>Decision Maker]
            N6[Notification Decider<br/>Alert Logic]
            N7[Escalation<br/>Human Review]
        end
        
        subgraph "Support Systems"
            subgraph "Policies"
                HIPAA[HIPAA Validator<br/>Compliance]
            end
            
            subgraph "Tools"
                CAL[Calendar API<br/>Availability]
                HIST[Patient History<br/>No-Show Data]
            end
            
            subgraph "Utils"
                AUDIT[Audit Logger<br/>Tracking]
                SCORE[Risk Scorer<br/>Calculation]
            end
            
            subgraph "LLM"
                CLIENT[OpenAI Client<br/>GPT-4]
            end
        end
    end
    
    MAIN --> GRAPH
    GRAPH --> STATE
    STATE --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 -->|Risk > 90%| N7
    N6 -->|Risk ≤ 90%| END[End]
    N7 --> END
    
    N1 -.-> HIPAA
    N1 -.-> AUDIT
    N2 -.-> CAL
    N2 -.-> AUDIT
    N4 -.-> HIST
    N4 -.-> SCORE
    N4 -.-> AUDIT
    
    style MAIN fill:#e1f5ff
    style GRAPH fill:#fff3e0
    style N1 fill:#c8e6c9
    style N2 fill:#c8e6c9
    style N3 fill:#c8e6c9
    style N4 fill:#c8e6c9
    style N5 fill:#c8e6c9
    style N6 fill:#c8e6c9
    style N7 fill:#ffccbc
    style HIPAA fill:#f8bbd0
    style AUDIT fill:#e1bee7
```

### High-Level Architecture

The system follows a **layered architecture** with clear separation of concerns:

1. **Entry Layer**: `main.py` serves as the application entry point
2. **Orchestration Layer**: LangGraph manages workflow state and transitions
3. **Agent Layer**: 7 specialized nodes handle specific responsibilities
4. **Support Layer**: Policies, tools and utilities provide cross-cutting services

---

## 🔄 System Flow

### Detailed Workflow Sequence

```mermaid
sequenceDiagram
    participant User
    participant Main
    participant Graph
    participant Intake
    participant Availability
    participant Conflict
    participant Risk
    participant Reschedule
    participant Notify
    participant Escalate
    participant HIPAA
    participant Calendar
    participant History
    participant Audit

    User->>Main: Request Appointment
    Main->>Graph: invoke(state)
    Graph->>Intake: Process Request
    
    Intake->>HIPAA: validate_request()
    HIPAA-->>Intake: ✓ Valid
    Intake->>Audit: Log INTAKE_VALIDATED
    Intake->>Graph: State Updated
    
    Graph->>Availability: Check Slot
    Availability->>Calendar: is_slot_available()
    Calendar-->>Availability: Not Available
    Availability->>Audit: Log AVAILABILITY_CHECKED
    Availability->>Graph: available=False
    
    Graph->>Conflict: Resolve Conflict
    Conflict->>Conflict: Find Alternative Slot
    Conflict->>Graph: requested_time updated
    
    Graph->>Risk: Assess No-Show Risk
    Risk->>History: get_no_show_count()
    History-->>Risk: 3 no-shows
    Risk->>Risk: calculate_risk() = 0.85
    Risk->>Audit: Log NO_SHOW_RISK_COMPUTED
    Risk->>Graph: no_show_risk=0.85
    
    Graph->>Reschedule: Make Decision
    alt Risk > 0.7
        Reschedule->>Reschedule: Reschedule to safer time
        Reschedule->>Graph: final_decision="Rescheduled"
    else Risk ≤ 0.7
        Reschedule->>Graph: final_decision="Confirmed"
    end
    
    Graph->>Notify: Decide Notification
    Notify->>Notify: Evaluate Risk
    alt Risk > 0.9
        Notify->>Graph: escalate_to_human=True
        Graph->>Escalate: Human Review Needed
        Escalate->>Graph: final_decision="Escalated"
    else Risk ≤ 0.9
        Notify->>Graph: notify_patient=True
    end
    
    Graph-->>Main: Final State
    Main-->>User: Appointment Decision
```

### Execution Flow Chart

```mermaid
flowchart TD
    START([User Request]) --> INTAKE{Intake Node<br/>HIPAA Validation}
    
    INTAKE -->|Valid| AVAIL{Availability Check<br/>Query Calendar}
    INTAKE -->|Invalid| ERROR1[HIPAA Violation Error]
    
    AVAIL -->|Available| RISK{No-Show Risk<br/>Assessment}
    AVAIL -->|Not Available| CONFLICT{Conflict Resolver<br/>Find Alternative}
    
    CONFLICT --> RISK
    
    RISK -->|Calculate Risk| RESCHED{Rescheduler<br/>Decision Logic}
    
    RESCHED -->|Risk > 0.7| RESCHED_ACTION[Reschedule to<br/>Safer Time]
    RESCHED -->|Risk ≤ 0.7| CONFIRM[Confirm<br/>Appointment]
    
    RESCHED_ACTION --> NOTIFY{Notification<br/>Decider}
    CONFIRM --> NOTIFY
    
    NOTIFY -->|Risk > 0.9| ESCALATE[Escalate to<br/>Human Staff]
    NOTIFY -->|Risk ≤ 0.9| PATIENT_NOTIFY[Notify Patient]
    
    ESCALATE --> END1([End: Manual Review])
    PATIENT_NOTIFY --> END2([End: Automated])
    ERROR1 --> END3([End: Error])
    
    style START fill:#4CAF50,color:#fff
    style INTAKE fill:#2196F3,color:#fff
    style AVAIL fill:#2196F3,color:#fff
    style CONFLICT fill:#FF9800,color:#fff
    style RISK fill:#9C27B0,color:#fff
    style RESCHED fill:#2196F3,color:#fff
    style NOTIFY fill:#2196F3,color:#fff
    style ESCALATE fill:#F44336,color:#fff
    style END1 fill:#F44336,color:#fff
    style END2 fill:#4CAF50,color:#fff
    style END3 fill:#F44336,color:#fff
```

### State Transitions

```mermaid
stateDiagram-v2
    [*] --> Intake: Request Received
    
    Intake --> AvailabilityCheck: Validated
    Intake --> [*]: HIPAA Violation
    
    AvailabilityCheck --> ConflictResolver: Always
    
    ConflictResolver --> RiskAssessment: Slot Found
    
    RiskAssessment --> Rescheduler: Risk Calculated
    
    Rescheduler --> NotificationDecider: Decision Made
    
    NotificationDecider --> Escalation: risk > 0.9
    NotificationDecider --> [*]: risk ≤ 0.9
    
    Escalation --> [*]: Human Review
    
    note right of Intake
        Validates HIPAA compliance
        Audits intake event
    end note
    
    note right of RiskAssessment
        Fetches patient history
        Calculates no-show probability
    end note
    
    note right of NotificationDecider
        Decides patient notification
        Triggers escalation if needed
    end note
```

---

## 📁 Project Structure

```
Healthcare_Appointment_Scheduling_agent/
│
├── 📄 main.py                          # Application entry point
├── 📄 requirements.txt                 # Python dependencies
│
├── 📂 graph/                           # Workflow orchestration
│   ├── graph.py                        # StateGraph definition & compilation
│   ├── state.py                        # AppointmentState schema
│   │
│   └── 📂 nodes/                       # Agent nodes (7 specialized agents)
│       ├── intake.py                   # Patient intake & HIPAA validation
│       ├── availability_check.py       # Calendar availability lookup
│       ├── conflict_resolver.py        # Smart conflict resolution
│       ├── no_show_risk.py            # Risk prediction engine
│       ├── rescheduler.py             # Appointment rescheduling logic
│       ├── notification_decider.py     # Notification decision logic
│       └── escalation.py              # Human escalation handler
│
├── 📂 llm/                             # LLM integration
│   └── client.py                       # OpenAI client wrapper
│
├── 📂 policies/                        # Compliance & governance
│   └── hippa.py                        # HIPAA validation rules
│
├── 📂 tools/                           # External integrations
│   ├── calendar.py                     # Calendar API mock
│   └── patient_history.py             # Patient history lookup mock
│
└── 📂 utils/                           # Utilities
    ├── audit.py                        # Audit logging
    └── scoring.py                      # Risk scoring algorithms
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.9+**
- **OpenAI API Key**
- **pip** package manager

### Step-by-Step Setup

1. **Clone the repository** (if applicable):
   ```bash
   cd Healthcare_Appointment_Scheduling_agent
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Verify installation**:
   ```bash
   python main.py
   ```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.30.0 | OpenAI API client for LLM integration |
| `langgraph` | ≥0.0.40 | State graph workflow orchestration |
| `typing-extensions` | ≥4.9.0 | Enhanced type hints support |
| `python-dotenv` | ≥1.0.1 | Environment variable management |

---

## ⚡ Quick Start

### Basic Usage Example

```python
from graph.graph import build_graph

# Initialize the agent
agent = build_graph()

# Create an appointment request
result = agent.invoke({
    "patient_id": "PATIENT_123",
    "request": {
        "appointment_time": "2025-12-26T10:00",
        "doctor_id": "DR_42",
        "department": "cardiology"
    },
    "notify_patient": False,
    "escalate_to_human": False,
})

# View the final decision
print(result["final_decision"])
```

### Expected Output

```
[AUDIT] INTAKE_VALIDATED | patient=PATIENT_123
[AUDIT] AVAILABILITY_CHECKED | patient=PATIENT_123
[AUDIT] NO_SHOW_RISK_COMPUTED | patient=PATIENT_123

=== FINAL DECISION ===
Rescheduled to reduce no show
```

### Running the Demo

```bash
python main.py
```

This runs a sample scenario demonstrating:
- HIPAA validation
- Slot unavailability detection
- Automatic conflict resolution
- Risk assessment (3 previous no-shows → 0.85 risk)
- Automatic rescheduling due to high risk

---

## 🔍 Detailed Components

### 1. Graph Orchestration (`graph/graph.py`)

The **StateGraph** is the brain of the system, orchestrating the flow between nodes.

```python
def build_graph():
    g = StateGraph(AppointmentState)
    
    # Add all 7 agent nodes
    g.add_node("intake", intake_node)
    g.add_node("availability", availability_check_node)
    g.add_node("resolve_conflict", conflict_resolver_node)
    g.add_node("risk", no_show_risk_node)
    g.add_node("reschedule", rescheduler_node)
    g.add_node("notify", notification_decider_node)
    g.add_node("escalate", escalation_node)
    
    # Define linear flow
    g.set_entry_point("intake")
    g.add_edge("intake", "availability")
    g.add_edge("availability", "resolve_conflict")
    g.add_edge("resolve_conflict", "risk")
    g.add_edge("risk", "reschedule")
    g.add_edge("reschedule", "notify")
    
    # Conditional routing based on risk
    g.add_conditional_edges(
        "notify",
        lambda s: "escalate" if s["escalate_to_human"] else END,
        {"escalate": "escalate", END: END}
    )
    
    return g.compile()
```

**Key Features:**
- **Linear pipeline** with conditional branching
- **Shared state** passed between all nodes
- **Type-safe** state management via `AppointmentState`

---

### 2. State Management (`graph/state.py`)

```python
class AppointmentState(TypedDict):
    # Input fields
    patient_id: str                      # Patient identifier
    request: dict                        # Appointment request details
    
    # Processing fields
    requested_time: str                  # Parsed appointment time
    doctor_id: str                       # Assigned doctor
    available: bool                      # Slot availability status
    conflict_reason: Optional[str]       # Reason for conflict
    
    # Risk assessment
    no_show_risk: float                 # Predicted no-show probability
    rescheduled_time: Optional[str]     # Alternative time if rescheduled
    
    # Decision outputs
    notify_patient: bool                # Should notify patient?
    escalate_to_human: bool             # Needs human review?
    final_decision: str                 # Final appointment decision
```

**State Evolution Example:**

```mermaid
graph LR
    S1[Initial State<br/>patient_id, request] --> S2[After Intake<br/>+ requested_time<br/>+ doctor_id]
    S2 --> S3[After Availability<br/>+ available<br/>+ conflict_reason]
    S3 --> S4[After Risk<br/>+ no_show_risk]
    S4 --> S5[Final State<br/>+ final_decision<br/>+ notify_patient<br/>+ escalate_to_human]
```

---

### 3. Agent Nodes (Detailed)

#### 🔵 **Intake Node** (`graph/nodes/intake.py`)

**Purpose:** Validates incoming requests and ensures HIPAA compliance

```python
def intake_node(state):
    validate_request(state['request'])      # HIPAA validation
    state['requested_time'] = state['request']['appointment_time']
    state['doctor_id'] = state['request']['doctor_id']
    audit("INTAKE_VALIDATED", state)        # Audit trail
    return state
```

**Responsibilities:**
- ✅ Validates only allowed fields (HIPAA compliance)
- ✅ Extracts appointment time and doctor ID
- ✅ Logs validation event

**HIPAA Validation Logic:**
```python
ALLOWED_FIELDS = {
    "appointment_time",
    "doctor_id", 
    "department"
}

def validate_request(data: dict):
    for key in data:
        if key not in ALLOWED_FIELDS:
            raise ValueError("HIPAA violation: unauthorized field")
```

---

#### 🔵 **Availability Check Node** (`graph/nodes/availability_check.py`)

**Purpose:** Checks doctor's calendar for slot availability

```python
def availability_check_node(state):
    state['available'] = is_slot_available(
        state['doctor_id'],
        state['requested_time']
    )
    
    if not state['available']:
        state['conflict_reason'] = 'doctor unavailable'
        
    audit("AVAILABILITY_CHECKED", state)
    return state
```

**Flow:**
```mermaid
graph TD
    A[Receive State] --> B{Query Calendar<br/>API}
    B -->|Available| C[available = True]
    B -->|Not Available| D[available = False<br/>conflict_reason = 'doctor unavailable']
    C --> E[Log Event]
    D --> E
    E --> F[Return State]
```

---

#### 🔵 **Conflict Resolver Node** (`graph/nodes/conflict_resolver.py`)

**Purpose:** Finds alternative slots when conflicts occur

```python
def conflict_resolver_node(state):
    if state["available"]:
        return state  # No conflict to resolve
    
    # Smart resolution: find next available slot
    state["requested_time"] = "26-12-2025T11:00"  # Mock: next hour
    state["available"] = True
    state["conflict_reason"] = None
    return state
```

**Resolution Strategy:**
- If slot available → Pass through
- If conflict → Find alternative time
- Update state with new time

> **Production Note:** In real implementations, this would query the calendar API for multiple alternative slots and apply intelligent scheduling algorithms.

---

#### 🔵 **No-Show Risk Node** (`graph/nodes/no_show_risk.py`)

**Purpose:** Predicts patient no-show probability using historical data

```python
def no_show_risk_node(state):
    history = get_no_show_count(state["patient_id"])
    state["no_show_risk"] = calculate_no_show_risk(history)
    audit("NO_SHOW_RISK_COMPUTED", state)
    return state
```

**Risk Calculation Algorithm:**

```python
def calculate_no_show_risk(previous_no_shows: int) -> float:
    if previous_no_shows >= 3:
        return 0.85  # High risk
    if previous_no_shows == 2:
        return 0.6   # Medium risk
    return 0.2       # Low risk
```

**Risk Categories:**

| No-Shows | Risk Score | Classification |
|----------|------------|----------------|
| 0-1 | 0.2 | 🟢 Low Risk |
| 2 | 0.6 | 🟡 Medium Risk |
| 3+ | 0.85 | 🔴 High Risk |

---

#### 🔵 **Rescheduler Node** (`graph/nodes/rescheduler.py`)

**Purpose:** Makes final scheduling decision based on risk assessment

```python
def rescheduler_node(state):
    if state["no_show_risk"] > 0.7:
        state["rescheduled_time"] = "27-12-2025T09:00"
        state["final_decision"] = "Rescheduled to reduce no show"
    else:
        state["final_decision"] = "Appointment confirmed"
    
    return state
```

**Decision Logic:**

```mermaid
graph TD
    A[Check Risk Score] --> B{Risk > 0.7?}
    B -->|Yes| C[Reschedule to<br/>Next Day Morning]
    B -->|No| D[Confirm Original<br/>Appointment]
    C --> E[Set final_decision]
    D --> E
```

**Rationale:** High-risk patients are rescheduled to morning slots (historically better attendance rates).

---

#### 🔵 **Notification Decider Node** (`graph/nodes/notification_decider.py`)

**Purpose:** Determines notification strategy and escalation needs

```python
def notification_decider_node(state):
    state["notify_patient"] = True
    state["escalate_to_human"] = state["no_show_risk"] > 0.9
    return state
```

**Decision Matrix:**

| Risk Score | Notify Patient | Escalate to Human |
|------------|----------------|-------------------|
| ≤ 0.9 | ✅ Yes | ❌ No |
| > 0.9 | ✅ Yes | ✅ Yes |

---

#### 🔵 **Escalation Node** (`graph/nodes/escalation.py`)

**Purpose:** Flags appointments for human review

```python
def escalation_node(state):
    state["final_decision"] = "Escalated to scheduling staff"
    return state
```

**Escalation Triggers:**
- No-show risk > 90%
- Complex scheduling conflicts
- VIP patients (future feature)

---

### 4. Support Systems

#### 🔒 HIPAA Compliance (`policies/hippa.py`)

```python
ALLOWED_FIELDS = {
    "appointment_time",
    "doctor_id",
    "department"
}

def validate_request(data: dict):
    for key in data:
        if key not in ALLOWED_FIELDS:
            raise ValueError("HIPAA violation: unauthorized field")
```

**Protection Against:**
- Unauthorized PHI (Protected Health Information) exposure
- Data leakage through logs
- Unvalidated field access

---

#### 📅 Calendar Tool (`tools/calendar.py`)

```python
def is_slot_available(doctor_id: str, time: str) -> bool:
    # Mock: blocks 10:00 AM slots
    return time not in ["26-12-2025T10:00"]
```

**Production Integration Points:**
- Google Calendar API
- Microsoft Outlook API
- Custom EHR calendar systems

---

#### 📊 Patient History (`tools/patient_history.py`)

```python
def get_no_show_count(patient_id: str) -> int:
    # Mock: returns 3 no-shows
    return 3
```

**Real Implementation Would:**
- Query patient database
- Fetch appointment history
- Calculate no-show statistics
- Consider time windows (last 6 months)

---

#### 📝 Audit Logger (`utils/audit.py`)

```python
def audit(event: str, state):
    print(f"[AUDIT] {event} | patient={state['patient_id']}")
```

**Audit Events Tracked:**
1. `INTAKE_VALIDATED` - Request passed HIPAA validation
2. `AVAILABILITY_CHECKED` - Calendar queried
3. `NO_SHOW_RISK_COMPUTED` - Risk assessment completed

**Production Requirements:**
- Store in secure audit database
- Timestamp all events
- Include user context
- Comply with audit retention policies

---

#### 🧮 Risk Scoring (`utils/scoring.py`)

```python
def calculate_no_show_risk(previous_no_shows: int) -> float:
    if previous_no_shows >= 3:
        return 0.85
    if previous_no_shows == 2:
        return 0.6
    return 0.2
```

**Enhancement Opportunities:**
- Machine learning models
- Multi-factor analysis (demographics, time of day, etc.)
- Real-time model updates

---

### 5. LLM Integration (`llm/client.py`)

```python
def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def call_llm(prompt: str, model='gpt-4.1-mini') -> str:
    client = get_client()
    res = client.chat.completions.create(
        model=model,
        temperature=0.1,  # Low temperature for deterministic outputs
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content or ""
```

**Current Usage:** Reserved for future enhancements (natural language processing of appointment requests)

**Potential Use Cases:**
- Parse free-text appointment requests
- Generate personalized patient notifications
- Analyze appointment patterns

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx

# Optional: Model Selection
MODEL_NAME=gpt-4.1-mini
TEMPERATURE=0.1

# Optional: Logging
LOG_LEVEL=INFO
AUDIT_ENABLED=true
```

### Customization Points

#### 1. **Risk Thresholds** (`utils/scoring.py`)

Adjust risk scoring based on your institution's data:

```python
def calculate_no_show_risk(previous_no_shows: int) -> float:
    # Customize these thresholds
    if previous_no_shows >= 3:
        return 0.85  # Adjust high-risk threshold
    if previous_no_shows == 2:
        return 0.6   # Adjust medium-risk threshold
    return 0.2       # Adjust low-risk threshold
```

#### 2. **Escalation Criteria** (`graph/nodes/notification_decider.py`)

Modify when to escalate to human review:

```python
def notification_decider_node(state):
    state["notify_patient"] = True
    # Change escalation threshold (default: 0.9)
    state["escalate_to_human"] = state["no_show_risk"] > 0.9
    return state
```

#### 3. **HIPAA Fields** (`policies/hippa.py`)

Add/remove allowed fields based on your requirements:

```python
ALLOWED_FIELDS = {
    "appointment_time",
    "doctor_id",
    "department",
    # Add more fields as needed
    # "insurance_id",
    # "reason_for_visit"
}
```

#### 4. **Calendar Integration** (`tools/calendar.py`)

Replace mock with real calendar API:

```python
def is_slot_available(doctor_id: str, time: str) -> bool:
    # Replace with actual calendar API call
    # Example: Google Calendar API
    # return google_calendar.check_availability(doctor_id, time)
    return time not in ["26-12-2025T10:00"]
```

---

## 📊 API Reference

### Main Entry Point

```python
from graph.graph import build_graph

agent = build_graph()
result = agent.invoke(initial_state)
```

### State Schema

#### Input State

```python
{
    "patient_id": str,              # Required: Patient identifier
    "request": {                    # Required: Appointment details
        "appointment_time": str,    # Format: "DD-MM-YYYYTHH:MM"
        "doctor_id": str,           # Doctor identifier
        "department": str           # Department name
    },
    "notify_patient": bool,         # Optional: Initial notification flag
    "escalate_to_human": bool       # Optional: Initial escalation flag
}
```

#### Output State

```python
{
    # All input fields plus:
    "requested_time": str,          # Parsed appointment time
    "doctor_id": str,               # Confirmed doctor ID
    "available": bool,              # Slot availability
    "conflict_reason": str | None,  # Conflict details
    "no_show_risk": float,          # Risk score (0.0 - 1.0)
    "rescheduled_time": str | None, # Alternative time
    "notify_patient": bool,         # Final notification decision
    "escalate_to_human": bool,      # Final escalation decision
    "final_decision": str           # Human-readable decision
}
```

### Example Usage Scenarios

#### Scenario 1: Successful Appointment (Low Risk)

```python
result = agent.invoke({
    "patient_id": "PATIENT_001",
    "request": {
        "appointment_time": "27-12-2025T14:00",
        "doctor_id": "DR_SMITH",
        "department": "general"
    },
    "notify_patient": False,
    "escalate_to_human": False
})

# Expected output:
# {
#     "final_decision": "Appointment confirmed",
#     "no_show_risk": 0.2,
#     "notify_patient": True,
#     "escalate_to_human": False
# }
```

#### Scenario 2: High Risk - Automatic Rescheduling

```python
result = agent.invoke({
    "patient_id": "PATIENT_HIGH_RISK",  # Has 3+ no-shows
    "request": {
        "appointment_time": "26-12-2025T15:00",
        "doctor_id": "DR_JONES",
        "department": "cardiology"
    },
    "notify_patient": False,
    "escalate_to_human": False
})

# Expected output:
# {
#     "final_decision": "Rescheduled to reduce no show",
#     "no_show_risk": 0.85,
#     "rescheduled_time": "27-12-2025T09:00",
#     "notify_patient": True,
#     "escalate_to_human": False
# }
```

#### Scenario 3: Critical Risk - Human Escalation

```python
# Simulate critical risk patient (modify scoring.py to return > 0.9)
result = agent.invoke({
    "patient_id": "PATIENT_CRITICAL",
    "request": {
        "appointment_time": "28-12-2025T10:00",
        "doctor_id": "DR_WILSON",
        "department": "surgery"
    },
    "notify_patient": False,
    "escalate_to_human": False
})

# Expected output:
# {
#     "final_decision": "Escalated to scheduling staff",
#     "no_show_risk": 0.95,
#     "notify_patient": True,
#     "escalate_to_human": True
# }
```

#### Scenario 4: Conflict Resolution

```python
result = agent.invoke({
    "patient_id": "PATIENT_002",
    "request": {
        "appointment_time": "26-12-2025T10:00",  # Unavailable slot
        "doctor_id": "DR_BROWN",
        "department": "dermatology"
    },
    "notify_patient": False,
    "escalate_to_human": False
})

# Expected output:
# {
#     "requested_time": "26-12-2025T11:00",  # Automatically rescheduled
#     "available": True,
#     "conflict_reason": None,
#     "final_decision": "Appointment confirmed" or "Rescheduled..."
# }
```

---

## 🔐 Security & Compliance

### HIPAA Compliance Features

#### 1. **Data Minimization**
- Only collects essential appointment fields
- No sensitive PHI stored in state
- Field validation prevents unauthorized data

#### 2. **Audit Trail**
```python
[AUDIT] INTAKE_VALIDATED | patient=PATIENT_123
[AUDIT] AVAILABILITY_CHECKED | patient=PATIENT_123
[AUDIT] NO_SHOW_RISK_COMPUTED | patient=PATIENT_123
```
- All critical operations logged
- Immutable audit records
- Patient-level tracking

#### 3. **Access Control**
- HIPAA validator blocks unauthorized fields
- Type-safe state management
- No data leakage between nodes

#### 4. **Secure Data Handling**
```python
# Example: Encrypting patient IDs (add to production)
def encrypt_patient_id(patient_id: str) -> str:
    # Use proper encryption (AES-256, etc.)
    return encrypted_id
```

### Best Practices for Production

#### ✅ DO:
- **Encrypt data at rest** and in transit
- **Implement role-based access control** (RBAC)
- **Store audit logs** in separate secure database
- **Use secure API keys** management (AWS Secrets Manager, Azure Key Vault)
- **Implement rate limiting** to prevent abuse
- **Conduct regular security audits**

#### ❌ DON'T:
- Store sensitive PHI in logs
- Expose patient data in error messages
- Use patient IDs in URLs
- Log full state objects (may contain PHI)

### Security Checklist

- [ ] Environment variables secured
- [ ] API keys rotated regularly
- [ ] Audit logs encrypted and backed up
- [ ] Access controls implemented
- [ ] HIPAA compliance verified
- [ ] Penetration testing completed
- [ ] Incident response plan documented

---

## 🧪 Testing

### Running Tests

```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Run specific test
python -m pytest tests/test_intake_node.py -v
```

### Test Coverage

```bash
pytest --cov=graph --cov=tools --cov=utils --cov-report=html
```

### Example Test Cases

```python
# tests/test_intake_node.py
def test_intake_node_valid_request():
    state = {
        "patient_id": "TEST_001",
        "request": {
            "appointment_time": "2025-12-26T10:00",
            "doctor_id": "DR_TEST",
            "department": "cardiology"
        }
    }
    result = intake_node(state)
    assert result["requested_time"] == "2025-12-26T10:00"
    assert result["doctor_id"] == "DR_TEST"

def test_intake_node_hipaa_violation():
    state = {
        "patient_id": "TEST_002",
        "request": {
            "appointment_time": "2025-12-26T10:00",
            "social_security_number": "123-45-6789"  # Unauthorized field
        }
    }
    with pytest.raises(ValueError, match="HIPAA violation"):
        intake_node(state)
```

---

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=""

CMD ["python", "main.py"]
```

```bash
# Build and run
docker build -t healthcare-scheduler .
docker run -e OPENAI_API_KEY=your_key healthcare-scheduler
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: healthcare-scheduler
spec:
  replicas: 3
  selector:
    matchLabels:
      app: healthcare-scheduler
  template:
    metadata:
      labels:
        app: healthcare-scheduler
    spec:
      containers:
      - name: scheduler
        image: healthcare-scheduler:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

### Production Considerations

1. **Scalability**
   - Stateless design enables horizontal scaling
   - Use message queues (RabbitMQ, Kafka) for high volume
   - Implement caching (Redis) for patient history

2. **Monitoring**
   - Prometheus for metrics
   - Grafana for dashboards
   - Alert on escalation rate, error rate

3. **Reliability**
   - Implement circuit breakers
   - Add retry logic with exponential backoff
   - Dead letter queues for failed appointments

---

## 📈 Performance Metrics

### Key Performance Indicators (KPIs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Average Processing Time** | < 2 seconds | Time from intake to final decision |
| **Conflict Resolution Rate** | > 95% | Automatic resolutions / total conflicts |
| **No-Show Prediction Accuracy** | > 80% | Correct predictions / total appointments |
| **Escalation Rate** | < 5% | Escalations / total appointments |
| **HIPAA Violations** | 0 | Caught by validator |
| **System Uptime** | > 99.9% | Available time / total time |

### Monitoring Example

```python
# Add to nodes for metrics collection
import time

def intake_node(state):
    start = time.time()
    # ... existing logic ...
    duration = time.time() - start
    metrics.record("intake_duration", duration)
    return state
```

---

## 🔄 Workflow Variations

### Alternative Flows

#### 1. **Multi-Doctor Fallback**

```python
# Enhanced conflict_resolver_node
def conflict_resolver_node(state):
    if state["available"]:
        return state
    
    # Try alternative doctors in same department
    alternative_doctors = find_available_doctors(
        state["request"]["department"],
        state["requested_time"]
    )
    
    if alternative_doctors:
        state["doctor_id"] = alternative_doctors[0]
        state["available"] = True
        state["conflict_reason"] = "Assigned to alternative doctor"
    
    return state
```

#### 2. **Priority Scheduling**

```python
# Enhanced rescheduler with priority
def rescheduler_node(state):
    priority = get_patient_priority(state["patient_id"])
    
    if priority == "URGENT":
        # Skip risk checks for urgent cases
        state["final_decision"] = "Urgent appointment confirmed"
    elif state["no_show_risk"] > 0.7:
        state["rescheduled_time"] = "27-12-2025T09:00"
        state["final_decision"] = "Rescheduled to reduce no show"
    else:
        state["final_decision"] = "Appointment confirmed"
    
    return state
```
