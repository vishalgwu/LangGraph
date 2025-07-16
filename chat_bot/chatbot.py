from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from serpapi import GoogleSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

@tool
def serpapi_search(query: str) -> str:
    """Google search using SerpAPI."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY  
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    if "error" in results:
        return f"Search error: {results['error']}"

    if results.get("organic_results"):
        first = results["organic_results"][0]
        return f"{first.get('title', '')}: {first.get('link', '')}\n{first.get('snippet', '')}"
    return "No results found."


class chatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")  

    def call_tool(self):
        tools = [serpapi_search]
        self.tool_node = ToolNode(tools=tools)
        self.llm_with_tool = self.llm.bind_tools(tools)

    def call_model(self, state: MessagesState):
        messages = state["messages"]
        response = self.llm_with_tool.invoke(messages)
        return {"messages": [response]}

    def router_function(self, state: MessagesState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.router_function, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        self.app = workflow.compile()
        return self.app


if __name__ == "__main__":
    mybot = chatbot()
    workflow = mybot()
    response = workflow.invoke({"messages": ["Who is the current president of the india?"]})
    print(response["messages"][-1].content)
