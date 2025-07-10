from dotenv import load_dotenv
import os
from langchain_community.utilities.tavily import TavilySearchAPIWrapper

load_dotenv()

# Tavily key
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# set up
tavily_tool = TavilySearchAPIWrapper()

# test
results = tavily_tool.results("latest news in India")

print(results)
