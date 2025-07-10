import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
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



GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["USER_AGENT"] = "my-app/0.1"

from langchain_google_genai import ChatGoogleGenerativeAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# optional user-agent
os.environ["USER_AGENT"] = "my-app/0.1"

from langchain_groq import ChatGroq
"""
GOrk is not working 
# initialize Groq with Gemma2-9b-it
llm = ChatGroq(
    model="llama3-70b-instruct"  # note: use model, not model_name
)
"""
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


openai_api_key = os.getenv("OPENAI_API_KEY")

# put it into environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
LANGCHAIN_PROJECT
# now you can use it with LangChain
from langchain_openai import ChatOpenAI
""""
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2
)
"""
while True:
    massage= input(" i/p $ o/p massage here  , if you want to quit , just say 'quit' ")
    if massage!= "quit":
        print(llm.invoke(massage).content)
    else:
        print("Have a nice one ")
        break

# MEMORY component 

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage

history={}


def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in history:
        history[session_id]= InMemoryChatMessageHistory()
    return history[session_id]


config = {"configurable": {"session_id": "firstchat"}}

memory_model= RunnableWithMessageHistory(llm,get_history)

result =memory_model.invoke((' Hi, i am vishal '),config=config).content 
print(result)

print(history)