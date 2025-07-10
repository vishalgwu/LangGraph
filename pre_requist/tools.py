import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY=os.getenv("SERPAPI_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"]= GROQ_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]=LANGCHAIN_PROJECT

#print(LANGCHAIN_PROJECT)
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
result= llm.invoke(" what is the full name of elon musk?")
#print(result.content)

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_api = WikipediaAPIWrapper()

wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

#print(wiki_tool.run({ "query": "Agentic AI" }))

from langchain_community.tools import YouTubeSearchTool
youtube_search = YouTubeSearchTool()
#print(youtube_search.run("B BK vinies"))

# CUstom tools 
#1)

from langchain.agents import tool
@tool
def get_addition( num1: int, num2: int) -> float :
    """ summation of two numbers """
    return num1+num2

#print(get_addition.run({"num1":4,"num2":5}))

#Agents 

from langchain.agents import AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent

wiki_agent_tool=load_tools(["wikipedia"],llm=llm)
wiki_agent = initialize_agent(
    wiki_agent_tool,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
#wiki_agent.run(" what is open ai and who creata is ? ")
from serpapi import GoogleSearch

from langchain import hub
prompt=hub.pull("hwchase17/openai-functions-agent")
prompt
tool=load_tools(["wikipedia"],llm=llm)

from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tool, prompt)
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
agent_executor.invoke({"input": "hello how are you?"})

from langchain_community.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
result=search.run("weather in SF")
print(result)



