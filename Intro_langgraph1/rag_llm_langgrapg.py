from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

from functools import partial
from langchain_community.document_loaders import TextLoader

utf8_loader = partial(TextLoader, encoding="utf-8")

loader2 = DirectoryLoader("../data", glob="data2.txt", loader_cls=utf8_loader)
docs2 = loader2.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
new_docs2 = text_splitter.split_documents(documents=docs2)

db2 = Chroma.from_documents(new_docs2, embeddings)
retriever2 = db2.as_retriever(search_kwargs={"k": 3})


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="Selected Topic")
    Reasoning: str = Field(description="Reasoning behind topic selection")

parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

def function_1(state):
    message = state["messages"]
    question = message[-1].content if hasattr(message[-1], "content") else str(message[-1])

    template = """
    Your task is to classify the given user query into one of the following categories: [India, Not Related]. 
    Only respond with the category name and nothing else.

    User query: {question}
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    response = chain.invoke({"question": question})
    return {"messages": [response.Topic]}

def router(state):
    last = state["messages"][-1]
    if "India" in last:
        return "RAG Call"
    return "LLM Call"


def function_2(state):
    question = state["messages"][0].content if hasattr(state["messages"][0], "content") else str(state["messages"][0])
    template = """Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever2, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke(question)
    return {"messages": [result]}


def function_3(state):
    question = state["messages"][0].content if hasattr(state["messages"][0], "content") else str(state["messages"][0])
    complete_query = f"Answer the following question with your knowledge: {question}"
    response = llm.invoke(complete_query)
    return {"messages": [response.content]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", function_1)
workflow.add_node("RAG", function_2)
workflow.add_node("LLM", function_3)
workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", router, {
    "RAG Call": "RAG",
    "LLM Call": "LLM"
})

workflow.add_edge("RAG", END)
workflow.add_edge("LLM", END)

app = workflow.compile()


from langchain_core.messages import HumanMessage
inputs = {"messages": [HumanMessage(content="Tell me about India's Industrial Growth")]}
output = app.invoke(inputs)

print("\n✅ Final Output:\n", output["messages"][-1])
