
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
#result= llm.invoke(" write a poem about LLM's").content
#print(result)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_vector = embedding_model.embed_query("LangChain RAG example")
#print(embedding_vector[:10])

urls= [ "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
       "https://lilianweng.github.io/posts/2023-06-23-agent/",
       "https://lilianweng.github.io/posts/2025-05-01-thinking/"]

docs_nested = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs_nested for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=20)
doc_chunks = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(documents=doc_chunks, collection_name="rag-chroma", embedding=embedding_model)

retriever = vectorstore.as_retriever()

print("\n✅ RAG Setup Complete. Number of chunks in vector DB:", len(doc_chunks))
from langchain import hub 
from langchain_core.output_parsers import StrOutputParser

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

question = "LLM Powered Autonomous Agents"
docs = retriever.invoke(question)

generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


class Grandscorecheck(BaseModel):
    """ Base model will check the relevent binary score on retival documents """
    binary_score: str= Field( description=" Documents are relevent to the question , 'yes' or 'no'")
    

doc_grad_llm= llm.with_structured_output(Grandscorecheck)

system=""" "You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    

grade_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrived documents : \n\n{documents} \n\n user question : {question}"),
        
    ]
)
grede_retival= grade_prompt| doc_grad_llm

question= "  Extrinsic hallucination: "

docs= retriever.get_relevant_documents(question)
doc_txt=docs[1].page_content

print(grede_retival.invoke({"documents": doc_txt, "question": question}))
"""
question= " tell me about the mahatma Gandhi "

docs= retriever.get_relevant_documents(question)
doc_txt=docs[1].page_content

print(grede_retival.invoke({"documents": doc_txt, "question": question}))
"""

# Prompt template:

from langchain.prompts import ChatPromptTemplate

prompt= """ You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
     

rewrite_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = rewrite_prompt | llm | StrOutputParser()

question_rewriter.invoke({"question": question})

question_rewriter


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
    
    """
    print("---RETRIEVE---")
    
    question = state["question"]
    
    documents = retriever.get_relevant_documents(question)
    
    return {"documents": documents, "question": question}
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    Returns only relevant docs and whether web_search is needed.
    """
    print("---CHECKING DOCUMENT RELEVANT IS TO QUESTION OR NOT---")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []

    for d in documents:
        score = grede_retival.invoke({
            "question": question,
            "documents": d.page_content
        })
        grade = score.binary_score
        if grade.lower().strip() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    # ✅ Only set web_search = Yes if no relevant docs were found
    web_search = "Yes" if len(filtered_docs) == 0 else "No"
    print(f"Total relevant docs: {len(filtered_docs)} | Web search? {web_search}")

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    print("---GENERATE---")
    
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    
    question = state["question"]
    
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    
    return {"documents": documents, "question": better_question}


# web search 
from langchain.schema import Document
from serpapi import GoogleSearch
from langchain.schema import Document
def web_search(state):
    print("---WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]

    search = GoogleSearch({
        "q": question,
        "api_key": SERPAPI_API_KEY
    })

    results = search.get_dict()
    organic_results = results.get("organic_results", [])

    snippets = [r["snippet"] for r in organic_results if "snippet" in r]
    web_text = "\n".join(snippets)

    web_doc = Document(page_content=web_text)
    documents.append(web_doc)

    return {"documents": documents, "question": question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes" :
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
from langgraph.graph import END, StateGraph, START
from typing import List

from typing_extensions import TypedDict

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

workflow= StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)  
workflow.add_node("generate", generate)  
workflow.add_node("transform_query", transform_query) 
workflow.add_node("web_search_node", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
    "transform_query": "transform_query",
    "generate": "generate"
})
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()


inputs = {"question": "tell me about the Extrinsic hallucination:."}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        print(f"Node '{key}':")
        # Optional: print full state at each nodeF
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    print("\n---\n")
# Final generation
print(value["generation"])
"""
output_path = "rag_agent_graph.png"
with open(output_path, "wb") as f:
    f.write(app.get_graph(xray=True).draw_mermaid_png())
print(f"✅ Graph saved as '{output_path}' in current directory.")

"""


inputs1 = {"question": "tell me about the Statue of liberty ."}
for output in app.stream(inputs1):
    for key, value in output.items():
        # Node
        print(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    print("\n---\n")

print(value["generation"])