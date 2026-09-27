## Intro
Your LLM agent works. It answers questions, calls tools, and impresses stakeholders. But what happens when a user asks it to “hack into a database”? What happens when it accidentally leaks a customer’s credit card number in a log? What happens when it confidently gives dangerous medical advice without a disclaimer?

This is why guardrails exist. And if you are building production AI systems, they are not optional.

In this crash course, we will cover everything you need to know about implementing guardrails in LangChain agents -- from simple keyword filters to production-grade layered middleware stacks. By the end, you will have built a fully guarded healthcare chatbot with PII detection, content filtering, human-in-the-loop approval, and output safety validation.

Here is what we will cover:

- [What are Guardrails and why do they matter?](./step1.md#what-are-guardrails-and-why-do-they-matter)
- [Two approaches](./step2.md#two-approaches-to-guardrails)
    - [Deterministic](./step2.md#deterministic-guardrails)
    - [Model-based](./step2.md#model-based-guardrails)
- [Built-in: PII Detection Middleware](./step3.md#built-in-pii-detection-middleware)
- [Built-in: Human-in-the-Loop Middleware](./step4.md#built-in-human-in-the-loop-middleware)
    - [Approval Flow](./step4.md#approval-flow)
    - [Rejection Flow](./step4.md#rejection-flow)
- [Custom: Before-Agent Guardrail (Input filtering)](./step5.md#custom-before-agent-guardrail-input-filter)
- [Custom: After-Agent Guardrail (output safety)](./step6.md#custom-after-agent-guardrail-output-safety)
- [Layered / Combined Guardrails](./step7.md#layered--combined-guardrails)
- [Real-World Use Case: Healthcare Chatbot](./step8.md#real-world-use-case-healthcare-chatbot)
    - [Healthcare-Specific Content Filter](./step8.md#healthcare-specific-content-filter)
    - [Medical Output Validator](./step8.md#medical-output-validator)
    - [Healthcare Tools](./step8.md#healthcare-tools)
    - [Assembling the Healthcare Chatbot](./step8.md#assembling-the-healthcare-chatbot)
    - [Testing the Healthcare Chatbot](./step8.md#testing-the-healthcare-chatbot)