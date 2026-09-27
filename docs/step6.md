# Custom After-Agent Guardrail (Output Safety)
Input filtering is half the battle. You also need to validate what comes out of your agent. The after_agent() hook runs after the agent produces a response but before the user sees it.

This is where model-based guardrails shine. We use a cheap, fast model (GPT-4o-mini) as a “safety judge” to evaluate the main agent’s output:

```python
from typing import Any
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, hook_config
)
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    Model-based guardrail: Use an LLM to evaluate response safety.
    Runs AFTER the agent produces a response, before it reaches the user.
    """

    def __init__(self):
        super().__init__()
        # Use a smaller, cheaper model for the safety check
        self.safety_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a lightweight model as the safety judge
        safety_prompt = f"""Evaluate if this AI response is safe \
and appropriate for users.
Respond with only 'SAFE' or 'UNSAFE'.

Response to evaluate:
{last_message.content}"""

        result = self.safety_model.invoke(
            [{"role": "user", "content": safety_prompt}]
        )

        if "UNSAFE" in result.content.upper():
            print("Output flagged as UNSAFE -- replacing with safe fallback")
            last_message.content = (
                "I'm unable to provide that response. "
                "Please rephrase your request or contact support."
            )

        return None


@tool
def general_tool(query: str) -> str:
    """A general purpose tool."""
    return f"Tool result: {query}"


safe_agent = create_agent(
    model="gpt-4o",
    tools=[general_tool],
    middleware=[SafetyGuardrailMiddleware()],
)
```
The pattern here is powerful: your main agent uses a capable (and expensive) model like GPT-4o, while the safety check uses a cheap, fast model like GPT-4o-mini. The safety model only needs to answer a simple yes/no question, so it does not need to be the most capable model.

```python
# Test output safety check
result = safe_agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like today?"}]
})
print("Response:")
print(result["messages"][-1].content)
```
If the safety model flags the output as unsafe, the response is replaced with a safe fallback message. The user never sees the problematic content.