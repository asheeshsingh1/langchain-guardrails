from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware, HumanInTheLoopMiddleware
)
from langchain_google_genai import ChatGoogleGenerativeAI
from guardrail.step6.custom_guardrail_output import SafetyGuardrailMiddleware
from guardrail.step5.custom_guardrail_input import ContentFilterMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results: {query}"

@tool
def send_email_tool(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Full layered guardrail stack
production_agent = create_agent(
    model=model,
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware"]
        ),

        # Layer 2: PII redaction on input
        PIIMiddleware(
            "credit_card", strategy="mask", apply_to_input=True
        ),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": True,
                "search_tool": False,
            }
        ),

        # Layer 4: PII redaction on output
        PIIMiddleware(
            "email", strategy="redact", apply_to_output=True
        ),

        # Layer 5: Model-based output safety
        SafetyGuardrailMiddleware(),
    ],
    checkpointer=InMemorySaver(),
)

print("Production-grade agent with 5-layer guardrails created!")