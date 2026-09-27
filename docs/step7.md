# Layered / Combined Guardrails
In production, you never use just one guardrail. You stack them. LangChain makes this trivial -- just add multiple middleware to the `middleware=[]` array. They execute in order, building layered protection:

```
User Input
    |
[Layer 1] ContentFilterMiddleware     -- Deterministic input filter
    |
[Layer 2] PIIMiddleware (input)       -- PII redaction on input
    |
[Layer 3] HumanInTheLoopMiddleware    -- Approval for sensitive tools
    |
[Layer 4] PIIMiddleware (output)      -- PII redaction on output
    |
[Layer 5] SafetyGuardrailMiddleware   -- Model-based output safety
    |
User Response
```
Here is the code:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware, HumanInTheLoopMiddleware
)
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

# Full layered guardrail stack
production_agent = create_agent(
    model="gpt-4o",
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
```
Five layers of protection. A request has to pass through all of them to reach the user. This is defense in depth -- if one layer misses something, the next one catches it.