# Real-World Use Case: Healthcare Chatbot
Let us put everything together in a realistic scenario. We will build a healthcare chatbot that has four guardrail layers working together.

## Healthcare-Specific Content Filter
First, a domain-specific input filter that blocks harmful or off-topic requests:

```python
from typing import Any
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, hook_config
)
from langchain.agents.middleware import (
    PIIMiddleware, HumanInTheLoopMiddleware
)
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

class HealthcareSafetyFilter(AgentMiddleware):
    """Block non-medical or harmful requests in a healthcare context."""

    BLOCKED_TOPICS = [
        "drug synthesis", "self-harm", "suicide method",
        "weapon", "hack"
    ]

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_msg = state["messages"][0]
        if first_msg.type != "human":
            return None

        content = first_msg.content.lower()
        for topic in self.BLOCKED_TOPICS:
            if topic in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I'm a healthcare assistant and can only help "
                            "with medical questions, appointments, and "
                            "health information. If you're in crisis, "
                            "please call 112 or your local emergency number."
                        )
                    }],
                    "jump_to": "end"
                }
        return None
```

## Medical Output Validator
An output guardrail that ensures every response includes an appropriate medical disclaimer:

```python
class MedicalOutputValidator(AgentMiddleware):
    """Ensure all responses include appropriate medical disclaimers."""

    DISCLAIMER = (
        "\n\nThis is general health information, not medical advice. "
        "Please consult a qualified healthcare professional."
    )

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Add disclaimer if not already present
        if "medical advice" not in last_message.content.lower():
            last_message.content += self.DISCLAIMER

        return None
```
## Healthcare Tools

```python
@tool
def search_symptoms(symptoms: str) -> str:
    """Search for information about medical symptoms."""
    return (
        f"Symptom information for: {symptoms}. "
        "Please consult a doctor for diagnosis."
    )

@tool
def book_appointment(patient_name: str, date: str, doctor: str) -> str:
    """Book a medical appointment."""
    return (
        f"Appointment booked for {patient_name} "
        f"with Dr. {doctor} on {date}"
    )

@tool
def get_medication_info(medication: str) -> str:
    """Get information about a medication."""
    return (
        f"General info about {medication}. "
        "Always follow your doctor's prescription."
    )
```
## Assembling the Healthcare Chatbot
```python
healthcare_bot = create_agent(
    model="gpt-4o",
    tools=[search_symptoms, book_appointment, get_medication_info],
    middleware=[
        # Guardrail 1: Block harmful/off-topic requests
        HealthcareSafetyFilter(),

        # Guardrail 2: Redact patient PII from inputs
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware(
            "credit_card", strategy="mask", apply_to_input=True
        ),

        # Guardrail 3: Require approval before booking appointments
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_appointment": True,
                "search_symptoms": False,
                "get_medication_info": False,
            }
        ),

        # Guardrail 4: Add medical disclaimer to all outputs
        MedicalOutputValidator(),
    ],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "You are a helpful healthcare assistant. "
        "You can search for symptoms, medication information, "
        "and help book appointments. Always be empathetic and "
        "remind users to consult a doctor for diagnosis."
    ),
)

print("Healthcare chatbot with full guardrail stack created!")
```
## Testing the Healthcare Chatbot

Test 1: Safe medical query

```python
config_t1 = {"configurable": {"thread_id": "healthcare_session_t1"}}

result = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": "What are symptoms of Type 2 Diabetes?"}]},
    config=config_t1
)
print(result["messages"][-1].content)
```
The agent answers the medical question normally, and the MedicalOutputValidator automatically appends the medical disclaimer.

Test 2: Query with PII (email gets redacted)

```python
result = healthcare_bot.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "My email is patient123@gmail.com. "
            "What can I take for a headache?"
        )
    }]},
    config=config_t1
)
print("=== PII Redaction Test ===")
print(result["messages"][-1].content)
```

The patient’s email is redacted before the model ever sees it. The medical question is answered normally with a disclaimer appended.

Test 3: Off-topic / harmful request (gets blocked)

```python
result = healthcare_bot.invoke({
    "messages": [{"role": "user", "content": "How do I synthesize drugs at home?"}]
},
    config=config_t1
)
print("=== Blocked Request ===")
print(result["messages"][-1].content)
```

The HealthcareSafetyFilter catches “drug synthesis” and returns a safe fallback message with emergency contact information. The request never reaches the LLM.

Test 4: Appointment booking (requires human approval)

```python
config = {"configurable": {"thread_id": "healthcare_session_001"}}

result = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": "Book me an appointment with Dr. Sharma on March 15"}]},
    config=config
)
print("=== Appointment Booking -- Awaiting Approval ===")

# Approve

from langgraph.types import Command

approved = healthcare_bot.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
print("=== After Approval ===")
print(approved["messages"][-1].content)
```

The appointment booking pauses for human approval. Only after the human approves does the booking execute. This prevents the AI from making unauthorized appointments.