import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT


llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


load_text = DirectoryLoader(".", glob="*.txt", loader_cls=TextLoader)
docs = load_text.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
new_docs = text_splitter.split_documents(docs)


db = Chroma.from_documents(new_docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

retriever_runnable = RunnableLambda(lambda q: retriever.invoke(q))


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)


retrieval_chain = (
    RunnableParallel({"context": retriever_runnable, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

question = "What is a language model?"
print(retrieval_chain.invoke(question))
