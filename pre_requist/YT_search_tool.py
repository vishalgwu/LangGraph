from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearchResults

tool= YouTubeSearchTool()

print(tool.description)

print(tool.run("Samay Raina"))
