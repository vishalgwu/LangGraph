from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Tool using SerpAPI
@tool
def city_details(search: str) -> str:
    """Web search to find details of the city using SerpAPI."""
    params = {
        "q": search,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
    }
    search_result = GoogleSearch(params).get_dict()
    try:
        snippet = search_result['organic_results'][0]['snippet']
    except (KeyError, IndexError):
        snippet = "No relevant results found."
    return snippet

# Bind tools
tools = [city_details]
model_with_tools = llm.bind_tools(tools)

# Define structured output schema
class citydetails(BaseModel):
    """This will give response to user"""
    state_name: str = Field(description="Name of the state")
    city_name: str = Field(description="Name of the city")
    country_name: str = Field(description="Name of the country")
    capital_name: str = Field(description="Name of the capital of the city")

# Define agent state
class agent_state(MessagesState):
    final_response: citydetails

# Bind structured output model
model_with_str_output = llm.with_structured_output(citydetails)

# Node: Call LLM with tools
from langchain_core.messages import ToolMessage

def call_model(state: agent_state):
    print(f"Step 01 - input to call_model: {state}")
    
    # Run model to get AI message with tool call
    ai_message = model_with_tools.invoke(state['messages'])
    print(f"Step 02 - response from call_model: {ai_message}")
    
    messages = [ai_message]

    # If tool was called, execute it and add ToolMessage
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
        
        # Find and run matching tool
        for tool in tools:
            if tool.name == tool_name:
                tool_result = tool.invoke(tool_args)
                tool_msg = ToolMessage(tool_call_id=tool_id, content=tool_result)
                messages.append(tool_msg)
    
    return {"messages": messages}



def should_continue(state: agent_state):
    message = state['messages']
    last_message = message[-1]
    if not getattr(last_message, "tool_call", None):
        return "respond"
    else:
        return "continue"

# Node: Respond with structured output
from langchain_core.messages import ToolMessage

def respond(state: agent_state):
    print(f"Step 03 - state in respond: {state}")
    
    # Extract the last ToolMessage (tool response)
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_output = msg.content
            break
    else:
        tool_output = "No tool output found."

    print(f"Using tool output for structured parsing:\n{tool_output}\n")

    # Send the tool output (not user prompt) to the structured output model
    response = model_with_str_output.invoke([HumanMessage(content=tool_output)])

    print(f"Step 04 - structured response: {response}")
    return {"final_response": response}


# Build graph
workflow = StateGraph(agent_state)
workflow.add_node("llm", call_model)
workflow.add_node("tool", ToolNode(tools))
workflow.add_node("respond", respond)
workflow.set_entry_point("llm")
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "tool",
        "respond": "respond",
    },
)
workflow.add_edge("tool", "llm")
workflow.add_edge("respond", END)

# Compile and run
graph = workflow.compile()
result = graph.invoke(
    input={"messages": [HumanMessage(content="Tell me about the city details Jaipur  ")]}
)['final_response']

print("\n✅ Final Structured Response:\n", result)
