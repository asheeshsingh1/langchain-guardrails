from typing import Any
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, hook_config
)
from langchain.agents.middleware import (
    PIIMiddleware, HumanInTheLoopMiddleware
)
from langchain_google_genai import ChatGoogleGenerativeAI
from guardrail.step8.healthcare_tools import (book_appointment,search_symptoms,get_medication_info)
from guardrail.step8.output_validator import MedicalOutputValidator
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

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

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

healthcare_bot = create_agent(
    model=model,
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