# Built-in PII Detection Middleware
LangChain provides built-in PIIMiddleware for detecting and handling Personally Identifiable Information. It supports multiple PII types -- email, credit card, IP address, MAC address, URL, and API keys -- and multiple strategies for handling them.

- redact -- Replaces with [REDACTED_EMAIL]
- mask -- Replaces with ****-****-****-1234
- hash -- Replaces with a hash like a8f5f167...
- block -- Raises an exception, stopping execution entirely

Here is how to set it up:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def customer_lookup(query: str) -> str:
    """Look up customer information."""
    return f"Customer record found for query: {query}"

# Create agent with PII Middleware
agent = create_agent(
    model="gpt-4o",
    tools=[customer_lookup],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

print("Agent with PII middleware created successfully!")
```
Three layers of PII protection in one agent. Emails get redacted, credit cards get masked, and API keys trigger a hard block. Let us test each scenario.

Test 1: PII Redaction in Action

```python
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "My email is john.doe@example.com and my card is "
            "5105-1051-0510-5100. Can you help me?"
        )
    }]
})

print("=== Agent Response ===")
print(result["messages"][-1].content)
```
The agent receives the message with the email replaced by [REDACTED_EMAIL] and the credit card replaced by ****-****-****-5100. The actual PII never reaches the model, the logs, or any downstream system.

Test 2: API Key Blocking
```python
try:
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Here is my key: sk-abcdefghijklmnopqrstuvwxyz123456"
        }]
    })
except Exception as e:
    print(f"Blocked as expected: {e}")
```

When an API key is detected, the `block` strategy raises an exception immediately. The request never reaches the model. This is critical for preventing accidental secret leakage in production systems.