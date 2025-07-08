from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
# For LangSmith tracing
from langsmith import traceable
import re
import os

load_dotenv()
print("LangSmith Project:", os.getenv("LANGSMITH_PROJECT"))  # DEBUG

memory = MemorySaver()

# 1. State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Thread-specific in-memory expense log
expense_log = {}  # thread_id -> list of (category, amount)

@tool
def log_expense(entry: str, thread_id: str = "default") -> str:
    """
    Extracts amount and category from text and logs the expense for a session.
    """
    matches = re.findall(r"\$?(\d+(?:\.\d{1,2})?)\s*(?:on|for)?\s*(\w+)", entry.lower())
    if not matches:
        return "Could not parse any expenses."

    session = expense_log.setdefault(thread_id, [])
    for amount, category in matches:
        session.append((category.lower(), float(amount)))

    return f"Logged {len(matches)} expense(s)."

@tool
def get_total(category: str = "", thread_id: str = "default") -> str:
    """
    Returns total spending (optionally per category) for a session.
    """
    session = expense_log.get(thread_id, [])
    if not session:
        return "No expenses logged for this session."

    if category:
        total = sum(amt for cat, amt in session if cat == category.lower())
        return f"Total spent on {category}: ${total:.2f}"
    else:
        total = sum(amt for _, amt in session)
        return f"Total spent so far: ${total:.2f}"
    
tools = [log_expense, get_total]

# 3. Model and Tool Binding
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

def assistant(state: State) -> State:
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

# 4. Build the LangGraph
builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition, {
    "tools": "tools",
    "__end__": END
})
builder.add_edge("tools", "assistant")

graph = builder.compile(checkpointer=memory)

@traceable
def call_graph():
    # 5. Testing: Multi-user expenses
    config1 = { "configurable": { "thread_id": "1" } }
    config2 = { "configurable": { "thread_id": "2" } }

    # 🧾 User 1 logs expenses
    msg = "I spent $45 on groceries and $20 on fuel"
    state = graph.invoke({"messages": [{"role": "user", "content": msg}]}, config=config1)
    print("User1:", state["messages"][-1].content)

    # 🧾 User 2 logs different expenses
    msg = "I spent $100 on books and $50 on stationery"
    state = graph.invoke({"messages": [{"role": "user", "content": msg}]}, config=config2)
    print("User2:", state["messages"][-1].content)

    # 💰 User 1 checks total
    msg = "What is my total spending so far?"
    state = graph.invoke({"messages": [{"role": "user", "content": msg}]}, config=config1)
    print("User1:", state["messages"][-1].content)

    # ➕ User 2 adds more and checks again
    msg = "Add $30 on snacks and tell me total again"
    state = graph.invoke({"messages": [{"role": "user", "content": msg}]}, config=config2)
    print("User2:", state["messages"][-1].content)
call_graph()    
