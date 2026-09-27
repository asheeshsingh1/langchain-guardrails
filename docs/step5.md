# Custom Before-Agent Guardrail (Input Filter)
The built-in middleware covers common cases, but real-world applications need custom logic. LangChain’s middleware system lets you create custom guardrails using before_agent() and after_agent() hooks.

The before_agent() hook runs before any LLM processing begins. This is perfect for keyword/content filtering, authentication checks, rate limiting, or blocking specific categories of requests.

```python
from typing import Any
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, hook_config
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()

class ContentFilterMiddleware(AgentMiddleware):
    """
    Deterministic guardrail: Block requests containing banned keywords.
    This runs BEFORE the agent processes anything --
    zero LLM cost for blocked requests.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                print(f"Blocked -- keyword detected: '{keyword}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I cannot process requests containing "
                            "inappropriate content. "
                            "Please rephrase your request."
                        )
                    }],
                    "jump_to": "end"
                }
        return None


@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Create agent with content filter
filtered_agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=[
                "hack", "exploit", "malware", "jailbreak", "bypass"
            ]
        ),
    ],
)
```
Let us break down what is happening in this custom middleware:

`AgentMiddleware` is the base class for all custom guardrails. You extend it and override the hooks you need.

`@hook_config(can_jump_to=["end"])` tells the middleware system that this hook is allowed to short-circuit execution by jumping directly to the end. Without this, you could not block requests.

`before_agent()` receives the current state (including all messages) and can either return `None` (let execution continue) or return a new state with `"jump_to": "end"` (block execution and return immediately).

The beauty of this pattern is that blocked requests never touch the LLM. Zero tokens consumed, zero cost incurred.

## Testing the Content Filter
```python
# Test 1: Safe request -- should pass through
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": "What is machine learning?"}]
})
print("Safe request response:")
print(result["messages"][-1].content)
```

```python
# Test 2: Unsafe request -- should be blocked
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a server?"}]
})
print("Unsafe request response:")
print(result["messages"][-1].content)
```

The safe request passes through to the LLM normally. The unsafe request gets caught by the keyword filter and returns a canned response without ever hitting the model.