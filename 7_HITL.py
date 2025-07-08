from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition, ToolExecutor
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import re
import os

load_dotenv()
print("LangSmith Project:", os.getenv("LANGSMITH_PROJECT"))

# In-memory store for conversation history
memory = MemorySaver()

# ---- 1. State Definition ----
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---- 2. Thread-specific Expense Log ----
# This remains a global variable for simplicity in this example.
expense_log = {}


@tool
def log_expense(entry: str, thread_id: str = "default") -> str:
    """Logs an expense to the system. This is a sensitive tool that requires approval."""
    matches = re.findall(r"\$?(\d+(?:\.\d{1,2})?)\s*(?:on|for)?\s*(\w+)", entry.lower())
    if not matches:
        return "No valid expenses found to log."

    session = expense_log.setdefault(thread_id, [])
    for amount, category in matches:
        session.append((category.lower(), float(amount)))
    return f"Successfully logged {len(matches)} expense(s)."

@tool
def get_total(category: str = "", thread_id: str = "default") -> str:
    """Returns total spending for a session (optionally by category)."""
    session = expense_log.get(thread_id, [])
    if not session:
        return "No expenses logged for this session."
    total = sum(amt for cat, amt in session if not category or cat == category.lower())
    return f"Total spent{' on ' + category if category else ''}: ${total:.2f}"

# ---- 3. LLM + Tools ----
tools = [log_expense, get_total]
tool_executor = ToolExecutor(tools)

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

def assistant_node(state: State) -> dict:
    """The assistant node, which calls the LLM."""
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

def tool_node(state: State) -> dict:
    """The tool node, which executes tools and returns the results."""
    # The ToolExecutor is called here to run the tools
    result = tool_executor.invoke(state["messages"][-1])
    # The result is wrapped in a ToolMessage to be sent back to the assistant
    tool_call_id = state["messages"][-1].tool_calls[0]['id']
    return {"messages": [ToolMessage(content=str(result), tool_call_id=tool_call_id)]}


# ---- 4. Graph Construction ----
builder = StateGraph(State)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile(checkpointer=memory)


# ---- 5. Corrected Execution Logic ----
from langsmith import traceable

@traceable
def call_graph():
    # Use a unique thread_id for each run to ensure a clean state
    config = {"configurable": {"thread_id": "4"}}

    # Step 1: Ask to log an expense
    msg = "I spent $45 on groceries and $20 on fuel"
    state = graph.invoke({"messages": [{"role": "user", "content": msg}]}, config=config)

    # Check if the assistant wants to call a tool
    if state["messages"][-1].tool_calls:
        tool_call = state["messages"][-1].tool_calls[0]
        # We manually intercept the call to our sensitive 'log_expense' tool
        if tool_call['name'] == 'log_expense':
            print(f"🤖 Awaiting approval: Do you want to log the expense: \"{tool_call['args']['entry']}\"?")
            user_input = input("✅ Approve? (yes/no): ").strip().lower()

            if user_input == "yes":
                print("...Approval received, resuming graph to log expense...")
                # The graph's state already has the pending tool call.
                # Invoking with 'None' resumes execution, running the ToolNode.
                state = graph.invoke(None, config=config)
            else:
                print("...Request denied. Informing assistant...")
                # We must provide a ToolMessage to resolve the pending tool call,
                # informing the assistant that the request was denied.
                state = graph.invoke(
                    {"messages": [ToolMessage(
                        content="User denied request. Do not log the expense.",
                        tool_call_id=tool_call['id']
                    )]},
                    config=config
                )

    print("🧾 Final:", state["messages"][-1].content)
    print("-" * 30)

    # Step 2: Check the total spending
    msg = "What is my total spending so far?"
    state = graph.invoke({"messages": [{"role": "user", "content": msg}]}, config=config)
    print("✅ Total:", state["messages"][-1].content)


if __name__ == "__main__":
    call_graph()