

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal, Sequence, TypedDict
from langchain import hub
from langchain_community import embeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field  
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from serpapi import GoogleSearch
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GROQ_API_KEY"]= GROQ_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]=LANGCHAIN_PROJECT
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo") 
result= llm.invoke(" hello ")
print(result)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_vector = embedding_model.embed_query("LangChain RAG example")
url_list =[ "https://lilianweng.github.io/posts/2023-06-23-agent/#scientific-discovery-agent",
       "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
       "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/",
       "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
       ]
web_loader = WebBaseLoader(url_list)
docs = web_loader.load()

print(f"\n✅ Loaded {len(docs)} document(s).")
print(f"Preview: {docs[0].page_content[:]}...")

list_of_docs= [item for sublist in docs for item in sublist]
#print(list_of_docs)

result=WebBaseLoader(url_list).load()
#print(result)
result1= WebBaseLoader(url_list).load()[0].metadata
#print(result1)

result2=WebBaseLoader(url_list).load()[0].metadata["description"]
#print(result2)


docs = [WebBaseLoader(url).load() for url in url_list]

docs_list=[item for sublist in docs for item in sublist]
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=25)
doc_splits=text_splitter.split_documents(docs_list)
doc_splits
print(doc_splits)


dataset= Chroma.from_documents(documents=doc_splits, collection_name="rag-chrome",embedding=embedding_model)

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=dataset.as_retriever(),
    name="vector_retriever",
    description="Retrieve documents from RAG Chroma DB",
)

tools = [retriever_tool]
retrieve = ToolNode(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def AI_AGNET(state: AgentState): 
    print(" --- AGENT MESSAGE -----")
    messages = state['messages']  # ✅ correct key
    
    if len(messages) > 1:
        last_message = messages[-1]
        question = last_message.content
        prompt = PromptTemplate(
            template="""You are a helpful assistant. Here is the question: {question}""",
            input_variables=["question"]
        )
        chain = prompt | llm
        response = chain.invoke({"question": question})
        return {"messages": [response]}
    else:
        llm_with_tool = llm.bind_tools(tools)
        response = llm_with_tool.invoke(messages)  # ✅ already HumanMessage list
        return {"messages": [response]}
    
class grade_score(BaseModel):
    binary_score:str=Field(description="Relevance score 'yes' or 'no'")

def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    llm_with_structure_op = llm.with_structured_output(grade_score)

    prompt = PromptTemplate(
        template="""You are a grader deciding if a document is relevant to a user’s question.
        Here is the document: {context}
        Here is the user’s question: {question}
        If the document talks about or contains information related to the user’s question, mark it as relevant. 
        Give a 'yes' or 'no' answer to show if the document is relevant to the question.""",
        input_variables=["context", "question"]
    )
    chain = prompt | llm_with_structure_op

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score.lower() == "yes":
        return "generate"
    else:
        return "rewrite" 
    
    
hub.pull("rlm/rag-prompt").pretty_print()

def generate(state:AgentState):
    print("---GENERATE---")
    messages = state["messages"]

    question = messages[0].content
    
    last_message = messages[-1]
    docs = last_message.content
    
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = prompt | llm

    response = rag_chain.invoke({"context": docs, "question": question})
    print(f"this is my response:{response}")
    
    return {"messages": [response]}

from langchain_core.messages import  HumanMessage
def rewrite(state:AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    
    message = [HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent or meaning. 
                    Here is the initial question: {question} 
                    Formulate an improved question: """)
       ]
    response = llm.invoke(message)
    return {"messages": [response]}

workflow=StateGraph(AgentState)
workflow.add_node("My_Ai_Assistant",AI_AGNET)
workflow.add_node("Vector_Retriever", retrieve) 
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite)
workflow.add_edge(START,"My_Ai_Assistant")
workflow.add_conditional_edges("My_Ai_Assistant",
                            tools_condition,
                            {"tools": "Vector_Retriever",
                                END: END,})

workflow.add_conditional_edges("Vector_Retriever",
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)


workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "My_Ai_Assistant")

app=workflow.compile()

from langchain_core.messages import HumanMessage

Problem = app.invoke({"messages": [HumanMessage(content="What is an Autonomous Agent?")]})
print(Problem)

problem1 = app.invoke({"messages": [HumanMessage(content="Who is the president of the USA?")]})
print(problem1)
