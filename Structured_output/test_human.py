import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict

# ✅ Load environment and LLM
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ Define the lookup tool
@tool
def lookup(query: str) -> str:
    """Use this tool to get the capital of a country. Example: 'What is the capital of France?'"""
    db = {
        "capital of france": "Paris",
        "capital of india": "New Delhi",
        "capital of germany": "Berlin",
        "capital of japan": "Tokyo"
    }
    return db.get(query.lower(), "Unknown")

tools = [lookup]
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# ✅ LangGraph state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ✅ AI reasoning node
def ai_node(state: AgentState):
    print("🧠 AI Reasoning...")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ✅ Tool pause with human-in-the-loop
memory = MemorySaver()
graph = StateGraph(AgentState)
graph.add_node("ai", ai_node)
graph.add_node("tool_exec", ToolNode(tools=tools))
graph.add_edge(START, "ai")
graph.add_conditional_edges("ai", tools_condition, {
    "tools": "tool_exec",
    "__end__": END
})
graph.add_edge("tool_exec", "ai")

# ✅ Compile graph with human interruption
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["tool_exec"]
)

# ✅ Initial prompt + system message
config = {"configurable": {"thread_id": "42"}}
system_message = (
    "Use the 'lookup' tool to answer any country capital question. "
    "Do not answer directly — always call the tool."
)

# ✅ Step 1: Initial question
user_input = "What is the capital of France?"
events = app.stream({"messages": [("system", system_message), ("user", user_input)]}, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# ✅ Step 2: Inject tool result manually (France)
from langchain_core.messages import ToolMessage
snapshot = app.get_state(config)
last_msg = snapshot.values["messages"][-1]

if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    tool_call = last_msg.tool_calls[0]
    tool_id = tool_call["id"]
    tool_args = tool_call["args"]

    print(f"\n🔍 Tool Call Detected: {tool_call}")
    print(f"🧑‍💻 Approving tool → {tool_args}")

    result = lookup.invoke(tool_args)

    new_msgs = [
        ToolMessage(content=result, tool_call_id=tool_id),
        AIMessage(content=result + "\nDo you want to know another capital?")
    ]
    app.update_state(config, {"messages": new_msgs})
else:
    print("ℹ️ No tool call to handle manually.")

# ✅ Step 3: Follow-up interaction
follow_up = input("👤 You: ")  # User reply like “Germany” or “No”
events = app.stream({"messages": [("user", follow_up)]}, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# ✅ Step 4: Inject second result manually (if applicable)
snapshot = app.get_state(config)
last_msg = snapshot.values["messages"][-1]

if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    tool_call = last_msg.tool_calls[0]
    tool_id = tool_call["id"]
    tool_args = tool_call["args"]

    print(f"\n🔍 Tool Call Detected: {tool_call}")
    print(f"🧑‍💻 Approving tool → {tool_args}")

    result = lookup.invoke(tool_args)

    new_msgs = [
        ToolMessage(content=result, tool_call_id=tool_id),
        AIMessage(content=result + "\nHappy to help! 👋")
    ]
    app.update_state(config, {"messages": new_msgs})
else:
    print("ℹ️ No second tool call. Conversation likely ended.")
