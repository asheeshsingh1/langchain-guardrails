# Two Approaches to Guardrails
Before building anything, you need to understand the two fundamental approaches to guardrails.

## Deterministic Guardrails
These are rule-based checks: regex patterns, keyword matching, explicit logic. They are fast, predictable, and cost-effective -- but they can miss nuanced violations.

```python
import re

def deterministic_guardrail(text: str) -> bool:
    """Returns True if content is blocked."""
    banned_keywords = ["hack", "exploit", "malware", "bomb"]
    return any(kw in text.lower() for kw in banned_keywords)

test_inputs = [
    "How do I hack into a database?",
    "What is the capital of France?",
    "Explain how malware spreads",
]

print("=== Deterministic Guardrail Demo ===")
for inp in test_inputs:
    blocked = deterministic_guardrail(inp)
    status = "BLOCKED" if blocked else "ALLOWED"
    print(f"{status}: {inp}")
```
Output:

```bash
BLOCKED: How do I hack into a database?
ALLOWED: What is the capital of France?
BLOCKED: Explain how malware spreads
```
Fast and cheap -- but notice it would also block “Explain how companies protect against malware,” which is a perfectly legitimate question. Keyword matching has no understanding of intent.

## Model-Based Guardrails
These use an LLM or classifier for semantic understanding. They catch subtle and nuanced issues that keyword matching misses -- but they are slower and more expensive.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

test_inputs = [
    "How do I hack into a database?",
    "What is the capital of France?",
    "Explain how malware spreads",
]


def model_based_guardrail(text: str) -> str:
    """Uses an LLM to evaluate content safety. Returns ALLOWED or BLOCKED."""

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
    )

    prompt = f"""
        You are a content safety classifier.

        Classify the following user input as either:
        - ALLOWED: safe and benign
        - BLOCKED: requests for harmful, malicious, or dangerous instructions

        Reply with exactly one word:
        ALLOWED
        or
        BLOCKED

        User input:
        {text}
    """

    result = model.invoke(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    verdict = result.content

    # Handle structured content
    if isinstance(verdict, list):
        verdict = "".join(
            block.get("text", "")
            for block in verdict
            if isinstance(block, dict)
        )

    verdict = verdict.strip().upper()

    if verdict not in {"ALLOWED", "BLOCKED"}:
        raise ValueError(f"Unexpected guardrail response: {verdict}")

    return verdict


print("=== Model-Based Guardrail Demo ===")

for inp in test_inputs:
    verdict = model_based_guardrail(inp)
    print(f"{verdict}: {inp}")
```

Output:

```bash
BLOCKED: How do I hack into a database?
ALLOWED: What is the capital of France?
ALLOWED: Explain how malware spreads
```

The model-based approach can understand intent. It knows that “Explain how malware spreads” in an educational context might be safe, while “How do I create malware” is not. This nuance is impossible with keyword matching alone.

The golden rule: use deterministic guardrails first (cheap, fast), then model-based guardrails second (expensive, thorough). This way, obviously bad requests get caught early without wasting LLM calls.