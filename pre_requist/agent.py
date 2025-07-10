"""
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Define your LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Define tools (example)
tools = [
    DuckDuckGoSearchRun()
    # add more tools if needed
]

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "hello how are you?"})
print(response)
"""

# ReAct Agent- 

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Tool
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        description="Search the web for recent sports news and results",
        func=search.run,
    )
]

# Use built-in ReAct prompt with correct placeholders
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Query
response = agent_executor.invoke(
    {"input": "How is the 1st and 2nd winner of F1 racing in 2023?"}
)

print(response)
