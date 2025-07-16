# === Imports ===
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


@tool
def search(query: str) -> str:
    """Only handles weather for San Francisco. Else let Gemini handle it."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    raise ValueError("Unknown location") 

tools = [search]
tool_node = ToolNode(tools)
llm_with_tool = llm.bind_tools(tools)

class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

def router_function(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# === Build LangGraph Workflow ===
memory = MemorySaver()
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", router_function, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent") 

app = workflow.compile(checkpointer=memory)

from langchain_core.messages import HumanMessage

input1 = {"messages": [HumanMessage(content="What is the weather in New York?")]}
config = {"configurable": {"thread_id": "session-1"}}

print("🧠 Streaming Execution:")
for step in app.stream(input1, config=config, stream_mode="values"):
    print("Step Output:\n------------")
    for msg in step["messages"]:
        msg.pretty_print()

print("\n🗃️ Memory contents:")
print(memory.get(config))
