from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from typing import TypedDict

class WorkflowState(TypedDict):
    text: str

def function_1(state: WorkflowState):
    state["text"] += " This is input from 1st function "
    return state

def function_2(state: WorkflowState):
    def function_3(inner):
        return inner + " (output from function 3)"
    output = function_3("this is function 3 in between function one and function 2")
    state["text"] += " " + output + " and this is input from 2nd function"
    return state


function1_runnable = RunnableLambda(function_1)
function2_runnable = RunnableLambda(function_2)

workflow1 = StateGraph(WorkflowState)
workflow1.add_node("function1", function1_runnable)
workflow1.add_node("function2", function2_runnable)
workflow1.add_edge("function1", "function2")

workflow1.set_entry_point("function1")
workflow1.set_finish_point("function2")

app1 = workflow1.compile()


result = app1.invoke({"text": "hello from workflow"})
print(result)

result1= app1.invoke({"text":"hi this is sunny"})
print(result1)