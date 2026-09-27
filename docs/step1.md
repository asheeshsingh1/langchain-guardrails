# What are Guardrails and why do they matter?
Guardrails are safety mechanisms that validate and filter content at key points in your agent’s execution. In LangChain, they are implemented as middleware that intercepts execution at three levels:

Before the agent starts (input guardrails) -- Block harmful requests, detect PII, enforce authentication, or apply rate limiting before any LLM processing happens. This saves cost because blocked requests never hit your model.

After the agent completes (output guardrails) -- Validate the final response before the user sees it. Check for safety, add compliance disclaimers, remove sensitive information that slipped through, or enforce quality standards.

Around model and tool calls -- Intercept specific tool calls to require human approval, redact PII from tool inputs/outputs, or apply business rules to specific operations.

Common use cases include PII leakage prevention (redacting emails and credit cards before logging), prompt injection blocking (detecting adversarial inputs), harmful content filtering (blocking dangerous requests), business rule enforcement (requiring approval for financial operations), and output quality validation (ensuring responses meet safety standards).


### Setup
Before we start, configure your environment:

```python

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
```