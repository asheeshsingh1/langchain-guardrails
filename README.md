# LangChain Guardrails

A practical implementation of **AI agent guardrails using LangChain and LangGraph**. This project demonstrates how to protect agent workflows using layered input/output validation, PII protection, human approval, and model-based safety checks.

## Guardrails Covered

* **Content filtering** — Block unsafe or restricted inputs.
* **PII protection** — Redact or mask sensitive information.
* **Human-in-the-loop** — Require approval for sensitive tool calls.
* **Output guardrails** — Validate and filter model responses.
* **Layered guardrails** — Combine multiple protections in an agent workflow.

## Project Structure

```text
guardrail/
├── step5/    # Input guardrails
├── step6/    # Output guardrails
└── step7/    # Layered guardrails
```

## Setup

```bash
git clone <repository-url>
cd langchain-guardrails

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Add your Gemini API key to `.env`:

```env
GOOGLE_API_KEY=your_api_key
```

## Run

Run examples as Python modules:

```bash
python -m guardrail.step5.custom_guardrail_input
python -m guardrail.step6.custom_guardrail_output
python -m guardrail.step7.layered_guardrail
```

## Tech Stack

**Python · LangChain · LangGraph · Gemini · PII Middleware · Human-in-the-Loop**
