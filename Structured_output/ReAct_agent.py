import os
from dotenv import load_dotenv
from typing import List
from pydantic import Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from serpapi import GoogleSearch
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

@tool
def multiple(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def addition(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def division(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        return float('inf')
    return a / b

@tool
def minus(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

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

tools = [search, addition, multiple, division, minus]
llm_with_tools = llm.bind_tools(tools)

system_prompt = SystemMessage(content="You are a helpful assistant that can do math and web search using tools.")

class ReActState(MessagesState):
    tools: list = Field(default_factory=lambda: tools)

def reasoner(state: ReActState):
    print("🧠 Reasoning step called")
    response = llm_with_tools.invoke([system_prompt] + state["messages"])
    return {
        "messages": state["messages"] + [response],  
        "tools": state["tools"]  
    }

workflow = StateGraph(ReActState)
workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", ToolNode(tools))  

workflow.set_entry_point("reasoner")

workflow.add_conditional_edges(
    "reasoner",
    tools_condition,
    {
        "tools": "tools",
        "final": END,
        "__end__": END
    }
)

workflow.add_edge("tools", "reasoner")  

graph = workflow.compile()

input_messages = [HumanMessage(content="What is 2 times the age of Donald trump? ")]
result = graph.invoke({
    "messages": input_messages,
    "tools": tools
})

print("\n📨 Final Messages:")
for m in result["messages"]:
    m.pretty_print()
