from typing_extensions import Literal
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, StateGraph,START,END
from langgraph.types import Command
from dotenv import load_dotenv
#from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from typing import Annotated

load_dotenv()
openai_model=ChatOpenAI(model="gpt-4")
#groq_model=ChatGroq(model="deepseek-r1-distill-llama-70b")
result=openai_model.invoke("hi")
print(result)



def addition_num(state):
    addition= state["num1"]+state["num2"]
    print(f"addition of the numbers is : {addition}")
    return Command(goto="multiply", update ={"sum":addition})

state={'num1':6,"num2":7}

#print(addition_num(state={'num1':5,'num2':6}))


@tool
def transfer_to_multiplication_expert():
    " take help from multiplication agent"
    return 

@tool
def transfer_to_addition_expert():
    "take help from addition expert"
    return 

model_with_tool= openai_model.bind_tools([transfer_to_multiplication_expert])

message=model_with_tool.invoke(" what is the multiiplication of 4 and 12 , and add 20 in into it . ").content
print(message)
    
def additional_expert(state:MessagesState)-> Command[Literal["multiplication_expert", "__end__"]]:
    
    system_prompt = (
        "You are an addition expert, you can ask the multiplication expert for help with multiplication."
        "Always do your portion of calculation before the handoff."
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    
    ai_msg = openai_model.bind_tools([transfer_to_multiplication_expert]).invoke(messages)
    
    
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        
        return Command(
            goto="multiplication_expert", update={"messages": [ai_msg, tool_msg]}
        )
    return {"messages": [ai_msg]}
def multiplication_expert(state:MessagesState)-> Command[Literal["additional_expert", "__end__"]]:
    
    system_prompt = (
        "You are a multiplication expert, you can ask an addition expert for help with addition. "
        "Always do your portion of calculation before the handoff."
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    ai_msg = openai_model.bind_tools([transfer_to_addition_expert]).invoke(messages)
    
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="additional_expert", update={"messages": [ai_msg, tool_msg]})
    return {"messages": [ai_msg]}
graph=StateGraph(MessagesState)


graph.add_node("additional_expert", additional_expert)
graph.add_node("multiplication_expert", multiplication_expert)

graph.add_edge(START,"additional_expert")




app=graph.compile()

result=app.invoke({"messages":[("user","what's (3 + 5) * 12. Provide me the output")]})
final_message = result["messages"][-1]  
print(final_message.content)