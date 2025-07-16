"""
from langchain_google_genai  import GoogleGenerativeAIEmbeddings
embedding= GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_google_genai import ChatGoogleGenerativeAI
llm= ChatGoogleGenerativeAI(model="gemini-2.5-pro")

result=llm.invoke(" hey , what up?")
print(result)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_data=DirectoryLoader("../data",glob="./*.txt",loader_cls=TextLoader)
load_docs=load_data.load()

text_load=RecursiveCharacterTextSplitter(chunk_size= 300, chunk_overlap=50)

new_docs=text_load.split_documents(documents=load_docs)
doc_strings = [doc.page_content for doc in new_docs]

data_base= Chroma.from_documents(new_docs,embedding)

data_retriever= data_base.as_retriever(search_kwargs={"k":3})

question = " what is meta llama3 ? "

docs= data_retriever.get_relevant_documents(question)
print(docs[0].metadata)
print(docs[0].page_content)

for doc in docs:
    print(doc)
    
"""

from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph  # ✅ correct import
from langchain.vectorstores.faiss import FAISS  # not used but fine


llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
retriever = RunnablePassthrough().with_config({"run_name": "dummy_retriever"})


class AgentState(TypedDict):
    messages: list
def function_1(state: AgentState) -> AgentState:
    message = state["messages"]
    question = message[-1]
    complete_prompt = (
        "Your task is to provide only the brief answer based on the user query. "
        "Don't include too much reasoning. Following is the user query: " + question
    )
    response = llm.invoke(complete_prompt)
    state['messages'].append(response.content)
    return state


def function_2(state: AgentState) -> AgentState:
    messages = state['messages']
    question = messages[0]

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = retrieval_chain.invoke(question)

    state['messages'].append(result)
    return state

workflow = StateGraph(AgentState)
workflow.add_node("LLM", function_1)
workflow.add_node("RAGtool", function_2)
workflow.add_edge("LLM", "RAGtool")
workflow.set_entry_point("LLM")
workflow.set_finish_point("RAGtool")
app = workflow.compile()

state = {"messages": ["Who was the first president of India?"]}
result = app.invoke(state)

print("\n Final Output from RAGtool:\n", result)
