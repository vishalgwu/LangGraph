from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
result1=llm.invoke("hi").content
print(result1)


def function_1(input):
    llm= ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    response=llm.invoke(input).content
    return response

def function_2(messages):
    if isinstance(messages, list) and len(messages) > 0:
        content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        return content.upper()
    else:
        return "INVALID INPUT"


from langgraph.graph import MessageGraph
workflow2=MessageGraph()


workflow2.add_node("llm",function_1)

workflow2.add_node("upper_case",function_2)
workflow2.add_edge("llm","upper_case")

workflow2.set_entry_point("llm")
workflow2.set_finish_point("upper_case")

app2=workflow2.compile()

result2=app2.invoke(" who was the first president of india ")
print(result2)

print(app2.invoke(" explain what is the agentic AI : "))


    
    

