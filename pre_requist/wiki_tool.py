from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=300)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)


print(tool.description)
print({'query':'langchain'})

print(tool.run('langchain'))


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field


api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)


class WikiInputs(BaseModel):
    query: str = Field(description="query to look up in Wikipedia, should be 3 or fewer words")

tool = WikipediaQueryRun(
    name="wiki-tool",
    description="look up things in wikipedia",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)

print(tool.name)  
print(tool.description)

print(tool.run("langchain"))
