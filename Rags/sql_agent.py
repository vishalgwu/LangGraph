import sqlite3
import os
if os.path.exists("employee_data.db"):
    os.remove("employee_data.db")

connection = sqlite3.connect("employee_data.db")

print(connection)

create_table1=""" CREATE TABLE IF NOT EXISTS EMPLOYEES(
    EMP_ID INTEGER PRIMARY KEY ,
    FIRST_NAME TEXT NOT NULL,
    LAST_NAME TEXT NOT NULL,
    EMAIL_ID TEXT UNIQUE  NOT NULL,
    HIRE_DATE TEXT NOT NULL,
    SALARY REAL NOT NULL);"""
    
create_table2= """ CREATE TABLE IF NOT EXISTS CUSTOMER (
    CUS_ID  INTEGER PRIMARY KEY AUTOINCREMENT, 
    FIRST_NAME TEXT NOT NULL , 
    LAST_NAME TEXT NOT NULL,
    EMAIL_ID TEXT UNIQUE NOT NULL,
    PHONE TEXT UNIQUE);"""
    
create_table3=""" CREATE TABLE IF NOT EXISTS ORDERS(
    ORDER_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    CUS_ID INTEGER  NOT NULL ,
    ORDER_DATE TEXT NOT NULL ,
    AMOUNT REAL NOT NULL,
    FOREIGN KEY (CUS_ID) REFERENCES  CUSTOMER(CUS_ID)); """
    

cursor = connection.cursor()
cursor.execute(create_table1)
cursor.execute(create_table2)
cursor.execute(create_table3)

insert_into_query_emp = """
INSERT INTO EMPLOYEES (EMP_ID, FIRST_NAME, LAST_NAME, EMAIL_ID, HIRE_DATE, SALARY)
VALUES (?, ?, ?, ?, ?, ?);
"""

insert_into_query_cus = """
INSERT INTO CUSTOMER (FIRST_NAME, LAST_NAME, EMAIL_ID, PHONE)
VALUES (?, ?, ?, ?);
"""

insert_into_query_ord = """
INSERT INTO ORDERS (CUS_ID, ORDER_DATE, AMOUNT)
VALUES (?, ?, ?);
"""
employees_data= [
    (1, "Alice", "Johnson", "alice.johnson@example.com", "2022-01-10", 75000),
    (2, "Bob", "Smith", "bob.smith@example.com", "2021-11-23", 68000),
    (3, "Charlie", "Lee", "charlie.lee@example.com", "2023-02-05", 82000),
    (4, "Diana", "Evans", "diana.evans@example.com", "2020-06-15", 90000),
    (5, "Edward", "Turner", "edward.turner@example.com", "2019-03-12", 56000),
    (6, "Fiona", "Martinez", "fiona.martinez@example.com", "2023-04-18", 72000),
    (7, "George", "Taylor", "george.taylor@example.com", "2022-08-30", 61000),
    (8, "Hannah", "Moore", "hannah.moore@example.com", "2021-09-20", 78000),
    (9, "Ian", "Walker", "ian.walker@example.com", "2020-10-01", 83000),
    (10, "Julia", "Scott", "julia.scott@example.com", "2019-12-22", 97000)
]

customer_data= [
    ("David", "Clark", "david.clark@example.com", "1234567890"),
    ("Eva", "Brown", "eva.brown@example.com", "2345678901"),
    ("Frank", "Wilson", "frank.wilson@example.com", "3456789012"),
    ("Grace", "Hall", "grace.hall@example.com", "4567890123"),
    ("Henry", "Allen", "henry.allen@example.com", "5678901234"),
    ("Isla", "Young", "isla.young@example.com", "6789012345"),
    ("Jack", "King", "jack.king@example.com", "7890123456"),
    ("Karen", "Wright", "karen.wright@example.com", "8901234567"),
    ("Leo", "Hill", "leo.hill@example.com", "9012345678"),
    ("Mia", "Green", "mia.green@example.com", "1123456789")
]

order_data= [
    (1, "2024-07-01", 150.75),
    (2, "2024-07-02", 320.50),
    (3, "2024-07-03", 89.99),
    (4, "2024-07-04", 245.00),
    (5, "2024-07-05", 530.30),
    (6, "2024-07-06", 175.25),
    (7, "2024-07-07", 400.10),
    (8, "2024-07-08", 625.80),
    (9, "2024-07-09", 300.00),
    (10, "2024-07-10", 155.45)
]



cursor.executemany(insert_into_query_emp, employees_data)
cursor.executemany(insert_into_query_cus, customer_data)
cursor.executemany(insert_into_query_ord, order_data)


connection.commit()
print(cursor.execute(" select * from employees"))
print("\nEMPLOYEES:")
for row in cursor.execute("SELECT * FROM EMPLOYEES"):
    print(row)

print("\nCUSTOMERS:")
for row in cursor.execute("SELECT * FROM CUSTOMER"):
    print(row)

print("\nORDERS:")
for row in cursor.execute("SELECT * FROM ORDERS"):
    print(row)

print("--------- first name from emp table--------")

cursor.execute("select first_name from employees where salary > 50000.0;")
result= cursor.fetchall()
print(result)

#Query is used to list all the table names in your SQLite database.
cursor.execute(" select name from sqlite_master where type= 'table';")

result1=cursor.fetchall()
print(" -----query is used to list all the table names in your SQLite database.----")
print(result1)

#   below - sql agent 
import os 
import os
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal, Sequence, TypedDict
from langchain import hub
from langchain_community import embeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field  
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from serpapi import GoogleSearch
from langchain_core.tools import tool
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
#result= llm.invoke(" hello").content
#print(result)


from langchain_community.utilities import SQLDatabase
db= SQLDatabase.from_uri("sqlite:///employee_data.db")

#print("Dialect: ",db.dialect)
#print("Usable tables", db.get_usable_table_names())

query_result= db.run("SELECT * FROM EMPLOYEES; ")
#print("Query result from employees table: ", query_result)

toolkit=SQLDatabaseToolkit(db=db, llm=llm)

tools=toolkit.get_tools()

tools
#print(tools)


for tool in tools:
    print(tool.name)

find_list_table_tool= next( (tool for tool in tools if tool.name == "sql_db_list_tables"), None)

find_get_schema_tool= next( (tool for tool in tools if tool.name =="sql_db_schema"), None)

print("\n Tables in DB:")
print(find_list_table_tool.invoke(""))

print("\n EMPLOYEES Table Schema:")
print(find_get_schema_tool.invoke("EMPLOYEES"))

def db_query_logic(query: str) -> str:
    """Execute a SQL query and return the result or error message."""
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please double-check and try again."
    return result

from langchain_core.tools import Tool

db_query_tool = Tool.from_function(
    name="db_query_tool",
    func=db_query_logic,
    description="Run a SQL query on the employee database. Input should be a valid SQL query string.",
)

print(db_query_tool.invoke({"query": "SELECT * FROM EMPLOYEES LIMIT 5;"}))

class Database:
    def run_no_throw(self, query):
        try:
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            return cursor.fetchall()  
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
from langchain_core.tools import StructuredTool

class SubmitFinalAnswer(BaseModel):
    final_answer: str = Field(..., description="The final answer to the user")


submit_final_answer_tool = StructuredTool.from_function(
    name="SubmitFinalAnswer",
    description="Submit the final answer to the user",
    func=lambda final_answer: final_answer,
    args_schema=SubmitFinalAnswer,
)
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
from langchain_core.prompts import ChatPromptTemplate
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages([("system", query_check_system), ("placeholder", "{messages}")])

query_check = query_check_prompt | llm.bind_tools([db_query_tool])

query_check.invoke({"messages": [("user", "SELECT * FROM Employees LIMIT 5;")]})

query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

Output the SQL query that answers the input question without a tool call.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

If you get an error while executing a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. Do not return any sql query except answer."""


query_gen_prompt = ChatPromptTemplate.from_messages([("system", query_gen_system), ("placeholder", "{messages}")])

query_gen = query_gen_prompt | llm.bind_tools([submit_final_answer_tool])


from typing import Annotated, Literal
from langchain_core.messages import AIMessage
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel



def handle_tool_error(state:State) -> dict:
    error = state.get("error") 
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
        ToolMessage(content=f"Error: {repr(error)}\n please fix your mistakes.",tool_call_id=tc["id"],)
        for tc in tool_calls
        ]
    }
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")



def query_gen_node(state: State):
    # Clean message history — only keep HumanMessages and AIMessages that are not tool responses
    cleaned_messages = [
        m for m in state["messages"]
        if not isinstance(m, ToolMessage)
    ]

    message = query_gen.invoke({"messages": cleaned_messages})

    if hasattr(message, "tool_calls") and message.tool_calls:
        return {"messages": [message]}
    else:
        return {
            "messages": [
                AIMessage(
                    content="Error: No SubmitFinalAnswer tool call found. Please revise your output.",
                    tool_calls=[],
                )
            ]
        }



def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"
def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}
workflow = StateGraph(State)

workflow.set_entry_point("query_gen")
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("correct_query", model_check_query)
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

workflow.add_conditional_edges(
    "query_gen",
    should_continue,
    {
        "correct_query": "correct_query",
        "query_gen": "query_gen",
        END: END
    }
)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", END)


app = workflow.compile()

from langchain_core.messages import HumanMessage

# Run the full LangGraph pipeline
response = app.invoke({"messages": [HumanMessage(content="Which employees earn more than 60000?")]})

# Print the result
print(response)


from langchain_core.messages import HumanMessage



last_msg = response["messages"][-1]

if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    final_tool_call = last_msg.tool_calls[0]
    if "args" in final_tool_call and "final_answer" in final_tool_call["args"]:
        print(final_tool_call["args"]["final_answer"])
    else:
        print("No 'final_answer' found in tool call args.")
else:
    print("No tool_calls in the final message.")
    print("Message content:", last_msg.content)


from pprint import pprint

print("\n--- DEBUG: Full Final Message ---")
pprint(vars(last_msg))


def print_final_answer(response):
    last_msg = response["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        final_tool_call = last_msg.tool_calls[0]
        if "args" in final_tool_call and "final_answer" in final_tool_call["args"]:
            print(final_tool_call["args"]["final_answer"])
        else:
            print("No 'final_answer' found in tool call args.")
    else:
        print("No tool_calls in the final message.")
        print("Message content:", last_msg.content)

print("--- DEBUG: Checking tools bound in query_gen ---")


def extract_final_answer(response):
    for msg in reversed(response["messages"]):
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc.get("name") == "SubmitFinalAnswer" and "final_answer" in tc.get("args", {}):
                    return tc["args"]["final_answer"]
    return "No final answer found."
print(extract_final_answer(response))



