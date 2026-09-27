# Built-in Human-in-the-Loop Middleware
Some operations are too sensitive for an AI to execute autonomously. Sending emails, deleting database records, processing financial transactions -- these need a human to approve.

LangChain’s `HumanInTheLoopMiddleware` pauses agent execution before sensitive tool calls and waits for human `approval`. It requires a `checkpointer` for state persistence across the interrupt.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject: {subject}"

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from the database."""
    return f"Deleted records from {table} where {condition}"

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Create agent with HITL middleware
hitl_agent = create_agent(
    model=model,
    tools=[search_web, send_email, delete_records],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,       # Require approval
                "delete_records": True,    # Require approval
                "search_web": False,       # Auto-approve
            }
        ),
    ],
    checkpointer=InMemorySaver(),  # Required for state persistence
)
```
The `interrupt_on` dictionary is the key configuration. You specify exactly which tools need human approval and which can auto-execute. Web searches are harmless -- let them through. Emails and database deletions? Those need a human in the loop.

## Approval Flow
```python
from hitl import hitl_agent
from langgraph.types import Command

# Step 1: Invoke -- agent will pause before send_email
config = {"configurable": {"thread_id": "session_001"}}

result = hitl_agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to team@company.com about the Q4 results"}]},
    config=config
)
print(result)
print("=== Agent paused -- awaiting human approval ===")

## Step 2: Human reviews and APPROVES
approved_result = hitl_agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config  # Same thread_id resumes the paused session
)

print("=== Approved! Final response ===")
print(approved_result["messages"][-1].content)
```
Using the same thread_id, the human sends an approve command, and the agent resumes execution from where it paused.

## Rejection Flow
```python
# Alternative -- Human REJECTS
from langgraph.types import Command
from hitl import hitl_agent

# Alternative -- Human REJECTS
config2 = {"configurable": {"thread_id": "session_002"}}

result = hitl_agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all records from the users table where active=false"}]},
    config=config2
)

print("=== Rejected! Mid response ===")
print(result)

rejected_result = hitl_agent.invoke(
    Command(resume={"decisions": [{"type": "reject", "reason": "Too risky, needs DBA review"}]}),
    config=config2
)

print("=== Rejected! Final response ===")
print(rejected_result["messages"][-1].content)
```
When a human rejects, the agent receives the rejection reason and responds accordingly. The dangerous operation never executes.