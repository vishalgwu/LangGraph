
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
#result= llm.invoke(" hello ")
#print(result)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")


url_list =[ "https://lilianweng.github.io/posts/2023-06-23-agent/#scientific-discovery-agent",
       "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
       "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/",
       "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
       ]
web_loader = WebBaseLoader(url_list)
docs = web_loader.load()

#print(f"\n✅ Loaded {len(docs)} document(s).")
#print(f"Preview: {docs[0].page_content[:]}...")

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
#print(doc_splits)


dataset= Chroma.from_documents(documents=doc_splits, collection_name="rag-chrome",embedding=embedding)
retriever = dataset.as_retriever()
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=dataset.as_retriever(),
    name="vector_retriever",
    description="Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]
retrieve = ToolNode(tools)



class score_card(BaseModel):
    """ Binary search on relevant  documents  for relative score """
    binary_score: str= Field( description= " documents are relevant to question , 'yes' or 'no'")
    
structured_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).with_structured_output(score_card, method="function_calling")

system = """You are a grader checking if a document is relevant to a user’s question.The check has to be done very strictly..  
If the document has words or meanings related to the question, mark it as relevant.  
Give a simple 'yes' or 'no' answer to show if the document is relevant or not."""
from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

#print(prompt)
chain= prompt | structured_llm
print("--------- ANSWER----------")
Question = " what is self-Reflection in LLMs "
docs= retriever.get_relevant_documents(Question)
#print(docs)
print("------------------- Binary score for LLM search ---------")
print(chain.invoke({"document":docs, "question":Question}))

text_of_docs=docs[1].page_content
print(text_of_docs)
#print(text_of_docs)
question= " who is the president of USA "
print("-----Binary score-------")
print(chain.invoke({"document":text_of_docs, "question":question}))
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
prompt= hub.pull("rlm/rag-prompt")
print("---- prompt------")
#print(prompt)

print(prompt.pretty_print())

rag_chain= prompt| llm
question ="  what is LLM's ? "
result3= rag_chain.invoke({"context":docs, "question": question})
#print("-- result of 3rd que -----")
#print(result3)
#This grader checks whether the LLM's generation is grounded in the provided source documents.

class check_hallucination(BaseModel):
    """ Binary  score for hallucination -will be present in generated  answer  """
    score: str= Field(
        description =" answer is in the formate of, ' Yes' or 'No' "
    )
    
str_llm= llm.with_structured_output(check_hallucination)

system_prompt= """ You are a grader checking if an LLM generation is grounded in or supported by a set of retrieved facts.  
Give a simple 'yes' or 'no' answer. 'Yes' means the generation is grounded in or supported by a set of retrieved the facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucinations_chain = hallucination_prompt | str_llm
hallucinations_grader = hallucinations_chain

print(" ------------Hallucination chain binary score  result ------")
print(hallucinations_chain.invoke({"documents":docs, "generation": result3}))



#This grader checks whether the LLM's answer actually addresses the user’s question.
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
print(answer_grader.invoke({"question": question, "generation": result3}))

system = """You are a question re-writer that converts an input question into a better optimized version for vector store retrieval document.  
You are given both a question and a document.  
- First, check if the question is relevant to the document by identifying a connection or relevance between them.  
- If there is a little relevancy, rewrite the question based on the semantic intent of the question and the context of the document.  
- If no relevance is found, simply return this single word "question not relevant." dont return the entire phrase 
Your goal is to ensure the rewritten question aligns well with the document for better retrieval."""
     
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human","""Here is the initial question: \n\n {question} \n,
             Here is the document: \n\n {documents} \n ,
             Formulate an improved question. if possible other return 'question not relevant'."""
        ),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()


question="who is a current President of USA?"
question_rewriter.invoke({"question":question,"documents":docs})


from typing import List
from typing_extensions import TypedDict
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    filter_documents: List[str]
    unfilter_documents: List[str]
    retry_count: int

my_retrieval_grader = chain

def retrieve(state: AgentState):
    print("----RETRIEVE----")
    question = state['question']
    documents = retriever.get_relevant_documents(question)
    return {
        "documents": documents,
        "question": question,
        "retry_count": state.get("retry_count", 0)
    }
def grade_documents(state:AgentState):
    print("----CHECK DOCUMENTS RELEVANCE TO THE QUESTION----")
    question = state['question']
    documents = state['documents']
    
    filtered_docs = []
    unfiltered_docs = []
    for doc in documents:
        try:
            score = my_retrieval_grader.invoke({"question": question, "document": doc.page_content})
            grade = score.binary_score
        except Exception as e:
            print(f"[⚠️ Error Grading Doc] Skipping doc due to error: {e}")
            unfiltered_docs.append(doc)
            continue
        
        if grade=='yes':
            print("----GRADE: DOCUMENT RELEVANT----")
            filtered_docs.append(doc)
        else:
            print("----GRADE: DOCUMENT NOT RELEVANT----")
            unfiltered_docs.append(doc)
    if len(unfiltered_docs)>1:
        return {"unfilter_documents": unfiltered_docs,"filter_documents":[], "question": question}
    else:
        return {"filter_documents": filtered_docs,"unfilter_documents":[],"question": question}
    
    
def decide_to_generate(state:AgentState):
    print("----ACCESS GRADED DOCUMENTS----")
    state["question"]
    unfiltered_documents = state["unfilter_documents"]
    filtered_documents = state["filter_documents"]
    
    
    if unfiltered_documents:
        print("----ALL THE DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY----")
        return "transform_query"
    if filtered_documents:
        print("----DECISION: GENERATE----")
        return "generate"
    

import pprint
import pprint

def grade_generation_vs_documents_and_question(state: AgentState):
    print("---CHECK HALLUCINATIONS---")
    question = state['question']
    documents = state['documents']
    generation = state['generation']
    
    score = hallucinations_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    # Check hallucinations
    if grade == 'yes':
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        print("---GRADE GENERATION vs QUESTION ---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        
        if grade == 'yes':
            print("---DECISION: GENERATION ADDRESSES THE QUESTION ---")
            return "useful"
        else:
            print("---DECISION: GENERATION IS NOT RELEVANT, RE-TRY---TRANSFORM QUERY")
            return "not useful"
    else:
        pprint.pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---TRANSFORM QUERY")
        return "not useful"
def generate(state: AgentState):
    print("----GENERATE----")
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
def transform_query(state: AgentState):
    print("----TRANSFORM QUERY----")
    question = state["question"]
    documents = state["documents"]

    print(f"📄 Original Question: {question}")
    print(f"📚 Context Docs: {len(documents)} retrieved")

    # Use the rewriter LLM chain
    response = question_rewriter.invoke({"question": question, "documents": documents}).strip().lower()
    print(f"📝 Rewritten Question: {response}")
    retry_count = state.get("retry_count", 0) + 1

    if response == "question not relevant":
        print("❌ QUERY NOT RELEVANT. Ending path.")
        return {
            "documents": documents,
            "question": "question not relevant",
            "generation": "question was not at all relevant"
        }
    else:
        return {
            "documents": documents,
            "question": response,
             "retry_count": retry_count
            
        }


from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(AgentState)
workflow.add_node("Docs_Vector_Retrieve", retrieve)
workflow.add_node("Grading_Generated_Documents", grade_documents) 
workflow.add_node("Content_Generator", generate)
workflow.add_node("Transform_User_Query", transform_query)

workflow.add_edge(START,"Docs_Vector_Retrieve")
workflow.add_edge("Docs_Vector_Retrieve","Grading_Generated_Documents")
workflow.add_conditional_edges("Grading_Generated_Documents",
                            decide_to_generate,
                            {
                            "generate": "Content_Generator",
                            "transform_query": "Transform_User_Query"
                            }
                            )
workflow.add_conditional_edges("Content_Generator",
                            grade_generation_vs_documents_and_question,
                            {
                            "useful": END,
                            "not useful": "Transform_User_Query",
                            }
                            )
def decide_to_generate_after_transformation(state: AgentState):
    question = state["question"].strip().lower()
    retry_count = state.get("retry_count", 0)

    if question == "question not relevant":
        print(" TRANSFORMED QUERY IS NOT RELEVANT — ENDING FLOW")
        return "query_not_at_all_relevant"
    elif retry_count >= 3:
        print("⚠️ Retry limit exceeded — Ending path")
        return "query_not_at_all_relevant"
    else:
        print(f"🔁 Retry count = {retry_count}. Re-running retriever.")
        return "Retriever"



workflow.add_conditional_edges("Transform_User_Query",
                decide_to_generate_after_transformation,
                {
                "Retriever":"Docs_Vector_Retrieve",
                "query_not_at_all_relevant":END
                }
                )
app=workflow.compile()
inputs = {"question": "Explain how the different types of agent memory work?", "retry_count": 0}
result = app.invoke(inputs)
generation = result.get("generation", "[ No generation produced]")
print("✅ Final Output:")
print(generation)


inputs = {"question": "Explain how the different types of agent memory work?", "retry_count": 0}
result = app.invoke(inputs)
second = result.get("generation", " No generation produced")
print(" ---------- ans for second --------")
print(second)

inputs = {"question": "what is role of data structure while creating ai agentic pattern?", "retry_count": 0}
result = app.invoke(inputs)
third = result.get("generation", " No generation produced")
print(" ---------- ans for third --------")
print(third)

inputs = {"question": "what is role of c language and php while creating ai agentic pattern?", "retry_count": 0}
result = app.invoke(inputs)
fourth = result.get("generation", " No generation produced")
print(" ---------- ans for fourth --------")
print(fourth)
