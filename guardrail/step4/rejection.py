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