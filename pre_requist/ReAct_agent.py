from langchain.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

@tool
def get_cricket_runs(name: str):
    """
    Takes a player name and returns the runs scored.
    """
    scores = {
        "Virat Kohli": 87,
        "Rohit Sharma": 102,
        "Shubman Gill": 54,
        "MS Dhoni": 45,
        "Rishabh Pant": 63
    }
    return f"{name} scored {scores.get(name, 0)} runs."

@tool
def get_strike_rate(name: str):
    """
    Takes a player name and returns strike rate.
    """
    rates = {
        "Virat Kohli": 125.4,
        "Rohit Sharma": 134.2,
        "Shubman Gill": 112.3,
        "MS Dhoni": 98.5,
        "Rishabh Pant": 144.0
    }
    return f"{name} had a strike rate of {rates.get(name, 0)}."

prompt = hub.pull("hwchase17/react")
tools = [get_cricket_runs, get_strike_rate]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({
    "input": "How many runs did Rohit Sharma score and what was his strike rate?"
})
print(response)
