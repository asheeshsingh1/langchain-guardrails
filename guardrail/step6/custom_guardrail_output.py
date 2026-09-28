from typing import Any
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, hook_config
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    Model-based guardrail: Use an LLM to evaluate response safety.
    Runs AFTER the agent produces a response, before it reaches the user.
    """

    def __init__(self):
        super().__init__()
        # Use a smaller, cheaper model for the safety check
        self.safety_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

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
            {last_message.content}
        """

        result = self.safety_model.invoke(
            [{"role": "user", "content": safety_prompt}]
        )

        safety_result = extract_text(result.content).strip().upper()

        print("%%%%%%%%%%%%%%%%%%%%",safety_result,"%%%%%%%%%%%%%%%%%%%%")

        if safety_result == "UNSAFE":
            print("Output flagged as UNSAFE -- replacing with safe fallback")
            last_message.content = (
                "I'm unable to provide that response. "
                "Please rephrase your request or contact support."
            )

        return None


def extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content)

@tool
def general_tool(query: str) -> str:
    """A general purpose tool."""
    return f"Tool result: {query}"

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

safe_agent = create_agent(
    model=model,
    tools=[general_tool],
    middleware=[SafetyGuardrailMiddleware()],
)

# Test output safety check
result = safe_agent.invoke({
    "messages": [{"role": "user", "content": "how to make bombs?"}]
})
final_response = result["messages"][-1].content

print("Response:")
print(extract_text(final_response))