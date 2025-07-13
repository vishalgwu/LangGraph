import os
import json
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from serpapi import GoogleSearch

# ✅ Load API key
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

# ✅ Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


# ✅ Define tools
@tool
def multiplication(a: int, b: int) -> int:
    
    """  Multiplication of two numbers. """
    return a * b

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
        return result["organic_results"][0]["snippet"]
    except Exception:
        return "Search failed or no results found."

# ✅ Fix: tools must be a list
tools = [search, multiplication]
tool_mapping = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

# ✅ LangGraph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ✅ LLM step
def invoke_model(state: AgentState):
    print("🧠 Reasoning step called")
    last_msg = state['messages'][-1]
    ai_msg = model_with_tools.invoke(state["messages"])

    return {"messages": state["messages"] + [ai_msg]}

def invoke_tool(state: AgentState):
    print("🔧 Tool invoked")
    tool_call = state["messages"][-1].additional_kwargs.get("tool_calls", [])[0]

    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])

    if tool_name == "search":
        user_input = input("⚠️ Expensive web search. Continue? (y/n): ")
        if user_input.lower() != "y":
            raise Exception("Search aborted by user.")

    # Run tool
    tool_result = tool_mapping[tool_name].invoke(tool_args)

    # Wrap tool result in an `AIMessage` so it's readable in the final messages
    from langchain_core.messages import AIMessage
    return {
        "messages": state["messages"] + [AIMessage(content=str(tool_result))]
    }


def router(state: AgentState):
    tool_calls = state["messages"][-1].additional_kwargs.get("tool_calls", [])
    return "tool" if tool_calls else "end"

graph = StateGraph(AgentState)
graph.add_node("ai_assistant", invoke_model)
graph.add_node("tool", invoke_tool)
graph.add_conditional_edges("ai_assistant", router, {"tool": "tool", "end": END})
graph.add_edge("tool", "ai_assistant")
graph.set_entry_point("ai_assistant")
app = graph.compile()

input_state = {
    "messages": [HumanMessage(content="What is 2 times the age of Narendra Modi?")]
}
output = app.invoke(input_state)


print("\n📨 Final Messages:")
for msg in output["messages"]:
    msg.pretty_print()