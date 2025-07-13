import os
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from serpapi import GoogleSearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
# The code snippet `SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")` is retrieving the value of the
# environment variable named "SERPAPI_API_KEY" using the `os.getenv()` function and storing it in the
# variable `SERPAPI_API_KEY`.
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

@tool
def search(query: str) -> str:
    """Search the web using Google SerpAPI and return the top snippet."""
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num": 1
    }
    try:
        result = GoogleSearch(params).get_dict()
        organic_results = result.get("organic_results", [])
        if organic_results and "snippet" in organic_results[0]:
            return organic_results[0]["snippet"]
        else:
            return "No relevant search results found."
    except Exception as e:
        return f"Search failed due to error: {str(e)}"

# ✅ Bind LLM to tools
tools = [search]
llm_with_tools = llm.bind_tools(tools)

# ✅ Define LangGraph State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ✅ Reasoning Node
def ai_assistant(state: AgentState):
    print("🧠 Reasoning step called")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ✅ Manual tool pause node
def wait_for_tool_response(state: AgentState):
    print("🔧 Waiting for manual tool input...")
    return {"messages": state["messages"]}

# ✅ Build LangGraph
memory = MemorySaver()
graph = StateGraph(AgentState)
graph.add_node("ai_assistant", ai_assistant)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_edge(START, "ai_assistant")

graph.add_conditional_edges("ai_assistant", tools_condition, {
    "tools": "tools",
    "__end__": END
})

graph.add_edge("tools", "ai_assistant")

# ✅ Compile the graph
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

# ✅ Step 1: Initial user query
config = {"configurable": {"thread_id": "1"}}
user_input = "What is the latest news in New Delhi?"
events = app.stream({"messages": [("user", user_input)]}, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# ✅ Step 2: Inject tool output only if a tool was requested
from langchain_core.messages import ToolMessage, AIMessage
snapshot = app.get_state(config)
last_msg = snapshot.values["messages"][-1]

if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    tool_call = last_msg.tool_calls[0]
    tool_call_id = tool_call["id"]
    tool_args = tool_call["args"]
    tool_result = search.invoke(tool_args)

    new_messages = [
        ToolMessage(content=tool_result, tool_call_id=tool_call_id),
        AIMessage(content=tool_result),
    ]
    app.update_state(config, {"messages": new_messages})
else:
    print("ℹ️ No tool call found in the last message. Skipping tool execution.")

# ✅ Step 3: Follow-up query
follow_up = "What about the weather there?"
events = app.stream({"messages": [("user", follow_up)]}, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
