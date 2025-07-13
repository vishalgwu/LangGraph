import os
from typing import Sequence
from typing import Annotated
import operator
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from serpapi import GoogleSearch
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Test basic LLM functionality
result = llm.invoke("Hi")
print("Basic LLM test:", result.content)

@tool
def multiply(first_number: int, second_number: int) -> int:
    """Multiply two integer numbers"""
    logger.info(f"Multiplying {first_number} * {second_number}")
    return first_number * second_number

@tool
def search(query: str) -> str:
    """Search the web using Google SerpAPI and return the top snippet."""
    logger.info(f"Searching for: {query}")
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num": 1
    }
    try:
        result = GoogleSearch(params).get_dict()
        organic_results = result.get("organic_results", [])
        if organic_results:
            first = organic_results[0]
            snippet = first.get("snippet")
            title = first.get("title")
            link = first.get("link")
            if snippet:
                return f"{title}: {snippet}\n{link}"
            else:
                return f"No snippet found, but got title: {title}\n{link}"
        else:
            return "No relevant search results found."
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return f"Search failed due to error: {str(e)}"
print(" for web search just print 'Y' or 'N" )
@tool
def add(first_number: int, second_number: int) -> int:
    """Add two integer numbers"""
    logger.info(f"Adding {first_number} + {second_number}")
    return first_number + second_number

# Test individual tools
print("Testing multiply tool:", multiply.invoke({"first_number": 10, "second_number": 20}))
print("Testing search tool:", search.invoke({"query": "who is the president of united state now?"}))

# Set up tools
tools = [search, multiply, add]
model_with_tools = llm.bind_tools(tools)
tool_mapping = {tool.name: tool for tool in tools}

print("Available tools:", list(tool_mapping.keys()))

# Test tool invocation
msg = model_with_tools.invoke("Multiply 5 and 7.")
print("LLM Tool Invocation Response:", msg)

if hasattr(msg, "tool_calls") and msg.tool_calls:
    tool_details = msg.tool_calls
    print("\n🔧 Tool Function Name:", tool_details[0]["name"])  
    print("🧾 Raw JSON Arguments:", tool_details[0]["args"])
    print("✅ Parsed Arguments (as dict):", tool_details[0]["args"])
else:
    print("\n❌ No tool calls were triggered by the LLM.")

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def invoke_model(state: AgentState):
    """
    Invoke the model with tools and handle cases where no tools are called.
    """
    messages = state['messages']
    logger.info(f"Invoking model with {len(messages)} messages")
    
    try:
        response = model_with_tools.invoke(messages)
        logger.info(f"Model response: tool_calls={hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
        
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            # If no tool is called, return the response as-is
            if response.content:
                logger.info("No tool calls, returning content response")
                return {"messages": [response]}
            else:
                # If no content and no tool calls, ask for clarification
                explanation = llm.invoke(f"I couldn't perform that action. Original request: {messages[-1].content}")
                return {"messages": [AIMessage(content=explanation.content)]}
        
        logger.info(f"Tool calls detected: {len(response.tool_calls)}")
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"Error in invoke_model: {str(e)}")
        return {"messages": [AIMessage(content=f"Error processing request: {str(e)}")]}

def invoke_tool(state: AgentState):
    """
    Invoke tools with comprehensive error handling and human-in-the-loop for search.
    """
    logger.info("Starting tool invocation")
    
    try:
        # Get the last message
        last_message = state['messages'][-1]
        
        # Check if the message has tool_calls attribute
        if not hasattr(last_message, 'tool_calls'):
            logger.error("Message has no tool_calls attribute")
            return {"messages": [AIMessage(content="No tool calls found in the message.")]}
        
        # Check if tool_calls is None or empty
        if not last_message.tool_calls:
            logger.error("Message tool_calls is None or empty")
            return {"messages": [AIMessage(content="No tool calls were found in the message.")]}
        
        # Get the first tool call
        tool_call = last_message.tool_calls[0]
        logger.info(f"Processing tool call: {tool_call}")
        
        # Validate tool call structure
        if not isinstance(tool_call, dict) or 'name' not in tool_call or 'args' not in tool_call:
            logger.error(f"Invalid tool call structure: {tool_call}")
            return {"messages": [AIMessage(content="Invalid tool call structure.")]}
        
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        logger.info(f"🛠️ Selected tool: {tool_name}")
        logger.info(f"📥 Tool Arguments: {tool_args}")
        
        # Check if tool exists
        if tool_name not in tool_mapping:
            available_tools = ", ".join(tool_mapping.keys())
            error_msg = f"Tool '{tool_name}' not found. Available tools: {available_tools}"
            logger.error(error_msg)
            return {"messages": [AIMessage(content=error_msg)]}
        
        # Human-in-the-loop for search operations
        if tool_name == "search":
            response = input(f"[y/n] Proceed with web search for '{tool_args.get('query', 'unknown query')}'? ")
            if response.lower().strip() != "y":
                logger.info("Search tool aborted by user")
                return {"messages": [AIMessage(content="Search operation cancelled by user.")]}
        
        # Execute the tool
        try:
            tool_function = tool_mapping[tool_name]
            tool_output = tool_function.invoke(tool_args)
            logger.info(f"📤 Tool Output: {tool_output}")
            
            # Return the result as a ToolMessage for proper LangGraph flow
            return {"messages": [ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call.get('id', 'unknown'),
                name=tool_name
            )]}
            
        except Exception as e:
            error_msg = f"❌ Tool execution failed: {str(e)}"
            logger.error(error_msg)
            return {"messages": [AIMessage(content=error_msg)]}
    
    except Exception as e:
        logger.error(f"Critical error in invoke_tool: {str(e)}")
        return {"messages": [AIMessage(content=f"Critical error in tool execution: {str(e)}")]}

def router(state):
    """
    Route the conversation based on the last message type.
    """
    last_msg = state['messages'][-1]
    logger.info(f"Router processing message type: {type(last_msg).__name__}")
    
    # Check if message has tool_calls attribute and it's not empty
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        logger.info("Routing to tool execution")
        return "tool"
    # Check if it's a ToolMessage (result from tool execution)
    elif isinstance(last_msg, ToolMessage):
        logger.info("Tool execution completed, routing to AI assistant")
        return "ai_assistant"
    # Check if message has content (regular response)
    elif hasattr(last_msg, "content") and last_msg.content:
        logger.info("Routing to end")
        return "end"
    else:
        # If we get here, something unexpected happened
        logger.warning("Unexpected message type, routing to end")
        return "end"

def should_continue(state):
    """
    Determine whether to continue the conversation or end it.
    """
    last_msg = state['messages'][-1]
    
    # If the last message is a ToolMessage, continue to let AI respond
    if isinstance(last_msg, ToolMessage):
        return "ai_assistant"
    # If AI made tool calls, execute them
    elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool"
    # Otherwise, end the conversation
    else:
        return "end"

# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("ai_assistant", invoke_model)
graph.add_node("tool", invoke_tool)

# Add edges
graph.add_conditional_edges(
    "ai_assistant", 
    should_continue, 
    {
        "tool": "tool",
        "end": END
    }
)

# After tool execution, go back to AI assistant to formulate response
graph.add_edge("tool", "ai_assistant")

# Set entry point
graph.set_entry_point("ai_assistant")

# Compile the graph
app = graph.compile()

# Test the agent
print("\n" + "="*50)
print("Testing the LangGraph Agent")
print("="*50)

def test_agent(query):
    """Test the agent with a given query"""
    print(f"\n🔍 Testing query: {query}")
    print("-" * 40)
    
    try:
        for i, s in enumerate(app.stream({"messages": [HumanMessage(content=query)]})):
            print(f"Step {i+1}: {list(s.keys())[0]}")
            message = list(s.values())[0]['messages'][0]
            
            if hasattr(message, 'content') and message.content:
                print(f"Content: {message.content}")
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
            if isinstance(message, ToolMessage):
                print(f"Tool result: {message.content}")
            
            print("-" * 20)
    
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        logger.error(f"Error in test_agent: {str(e)}")

# Run tests
if __name__ == "__main__":
    # Test cases
    test_queries = [
        "Calculate the addition of 25 and 35",
        "What is 12 multiplied by 8?",
        "Search for the latest news about AI",  # This will prompt for human approval
        "Hello, how are you?"  # This should not trigger tools
    ]
    
    for query in test_queries:
        test_agent(query)
        print("\n" + "="*50)