from typing_extensions import Literal
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


openai_model = ChatOpenAI(model="gpt-4")
print(openai_model.invoke("Hello from multi-agent system!").content)

@tool
def transfer_to_multiplication_expert():
    "Take help from multiplication expert"
    return

@tool
def transfer_to_addition_expert():
    "Take help from addition expert"
    return

@tool
def transfer_to_subtraction_expert():
    "Take help from subtraction expert"
    return

@tool
def transfer_to_division_expert():
    "Take help from division expert"
    return

@tool
def transfer_to_modulus_expert():
    "Take help from modulus expert"
    return

@tool
def transfer_to_exponent_expert():
    "Take help from exponent expert"
    return

@tool
def transfer_to_sqrt_expert():
    "Take help from square root expert"
    return

@tool
def transfer_to_trig_expert():
    "Take help from trigonometric expert (sin, cos, tan)"
    return


def additional_expert(state: MessagesState) -> Command[Literal["multiplication_expert", "__end__"]]:
    system_prompt = (
        "You are an addition expert. Solve expressions like 3 + 5 + 10. "
        "If the user also needs multiplication, ask the multiplication expert."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = openai_model.bind_tools([transfer_to_multiplication_expert]).invoke(messages)
    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[0]["id"]
        tool_msg = {"role": "tool", "content": "Transferred to multiplication expert", "tool_call_id": tool_call_id}
        return Command(goto="multiplication_expert", update={"messages": [ai_msg, tool_msg]})
    return {"messages": [ai_msg]}

def multiplication_expert(state: MessagesState) -> Command[Literal["additional_expert", "__end__"]]:
    system_prompt = (
        "You are a multiplication expert. Solve expressions like 3 * 5 * 2. "
        "If addition is involved, ask the addition expert."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = openai_model.bind_tools([transfer_to_addition_expert]).invoke(messages)
    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[0]["id"]
        tool_msg = {"role": "tool", "content": "Transferred to addition expert", "tool_call_id": tool_call_id}
        return Command(goto="additional_expert", update={"messages": [ai_msg, tool_msg]})
    return {"messages": [ai_msg]}

def subtraction_expert(state: MessagesState) -> Command[Literal["division_expert", "__end__"]]:
    system_prompt = (
        "You are a subtraction expert. Solve expressions like 10 - 3 - 1. "
        "If division is involved, ask the division expert."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = openai_model.bind_tools([transfer_to_division_expert]).invoke(messages)
    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[0]["id"]
        tool_msg = {"role": "tool", "content": "Transferred to division expert", "tool_call_id": tool_call_id}
        return Command(goto="division_expert", update={"messages": [ai_msg, tool_msg]})
    return {"messages": [ai_msg]}

def division_expert(state: MessagesState) -> Command[Literal["modulus_expert", "__end__"]]:
    system_prompt = (
        "You are a division expert. Solve expressions like 10 / 2. "
        "If modulus is involved, ask the modulus expert."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = openai_model.bind_tools([transfer_to_modulus_expert]).invoke(messages)
    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[0]["id"]
        tool_msg = {"role": "tool", "content": "Transferred to modulus expert", "tool_call_id": tool_call_id}
        return Command(goto="modulus_expert", update={"messages": [ai_msg, tool_msg]})
    return {"messages": [ai_msg]}

def modulus_expert(state: MessagesState) -> Command[Literal["__end__"]]:
    system_prompt = "You are a modulus expert. Handle expressions like 10 % 3."
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = openai_model.invoke(messages)
    return {"messages": [ai_msg]}

def exponent_expert(state: MessagesState) -> Command[Literal["sqrt_expert", "__end__"]]:
    system_prompt = (
        "You are an exponent expert. Solve expressions like 2^5 or 3 ** 3. "
        "If square root is involved, delegate to the sqrt expert."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = openai_model.bind_tools([transfer_to_sqrt_expert]).invoke(messages)
    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[0]["id"]
        tool_msg = {"role": "tool", "content": "Transferred to square root expert", "tool_call_id": tool_call_id}
        return Command(goto="sqrt_expert", update={"messages": [ai_msg, tool_msg]})
    return {"messages": [ai_msg]}

def sqrt_expert(state: MessagesState) -> Command[Literal["__end__"]]:
    system_prompt = "You are a square root expert. Handle expressions like sqrt(49), √16."
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = openai_model.invoke(messages)
    return {"messages": [ai_msg]}

def trig_expert(state: MessagesState) -> Command[Literal["__end__"]]:
    system_prompt = (
        "You are a trigonometry expert. Handle sin(x), cos(x), tan(x) in degrees or radians."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = openai_model.invoke(messages)
    return {"messages": [ai_msg]}


graph = StateGraph(MessagesState)

graph.add_node("additional_expert", additional_expert)
graph.add_node("multiplication_expert", multiplication_expert)
graph.add_node("subtraction_expert", subtraction_expert)
graph.add_node("division_expert", division_expert)
graph.add_node("modulus_expert", modulus_expert)
graph.add_node("exponent_expert", exponent_expert)
graph.add_node("sqrt_expert", sqrt_expert)
graph.add_node("trig_expert", trig_expert)

graph.add_edge(START, "exponent_expert")  
for node in [
    "additional_expert", "multiplication_expert", "subtraction_expert",
    "division_expert", "modulus_expert", "exponent_expert",
    "sqrt_expert", "trig_expert"
]:
    graph.set_finish_point(node)

app = graph.compile()


query = "What is 2^3 + sqrt(49)?"
result = app.invoke({"messages": [HumanMessage(content=query)]})

print("\n✅ Final Result:")
print(result["messages"][-1].content)
