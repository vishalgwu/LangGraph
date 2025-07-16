#  LangGraph - Multi-Agent AI Systems with Structured Output

![LangGraph Banner](https://github.com/vishalgwu/LangGraph/blob/main/LangGraph-image.jpg)

LangGraph is an experimental playground where I've brought to life a wide variety of multi-agent AI workflows, powered by LangChain + LangGraph. This project dives into:
- Chains & Conditional Nodes
- Memory-Powered Agents
- Multiple RAG Pipelines (Agentic, Self-RAG, Corrective)
- ReAct Agent Integration
- Human-in-the-Loop Design

 From reasoning to retrieval to response — this repo explores how agents collaborate intelligently using tools, memory, and human feedback loops.

--------------------------------------------------------------------------------

Folder Structure

LangGraph/
│
├── chat_bot/                 # Gemini and tool-calling chat agents
│   ├── google_langgraph.py
│   ├── intro_langgraph1.py
│   ├── rag_langgraph.py
│   └── rag_llm_langgraph.py
│
├── multi-agent/              # Tool-using, collaborative agents
│   ├── calculator.py
│   ├── multi_agent.py
│   ├── research_agent.py
│   ├── test1.py
│   └── test2.py
│
├── pre_requist/              # Core logic and pre-requisites
│   ├── agent.py
│   ├── ReAct_agent.py
│   ├── text_loader.py
│   ├── tools.py
│   └── web_search.py
│
├── Rags/                     # All RAG variants
│   ├── agentic_rag.py
│   ├── corrective_rag.py
│   ├── self_rag.py
│   ├── sql_agent.py
│   ├── employee_data.db
│   └── rag_agent_graph.png
│
├── Structured_output/        # Output formatting & approvals
│   ├── human_approval_agent.py
│   ├── human_in_loop_agent.py
│   ├── structured_output_agent.py
│   └── test_human.py
│
├── data/                     # Sample text files
│   ├── data.txt
│   └── data2.txt
│
├── .env                      # 🔐 API Keys (not uploaded)
├── req.txt                   # 🔧 Python requirements
└── langgraph_presentation.pptx  # Project overview presentation

--------------------------------------------------------------------------------

 Getting Started

1.  Clone the repository

git clone https://github.com/vishalgwu/LangGraph.git
cd LangGraph

2.  Install dependencies

Make sure Python 3.9+ is installed, then run:

pip install -r req.txt

3.  Add your API keys to `.env`

Create a `.env` file in the root directory and add the following:

OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_PROJECT=your_project_name

--------------------------------------------------------------------------------

 What You’ll Find

Module               | Description
--------------------|---------------------------------------------------------------
Multi-Agent System  | SQL Agent, Web Search, YouTube Search, and Calculator combined into one decision-driven system
Agentic RAG         | A LangGraph agent that plans, retrieves, and reacts
Corrective RAG      | Recovery from irrelevant answers
Self-RAG            | RAG pipeline that rewrites or rethinks its query
Human-in-the-Loop   | Agents requiring approval before final answers
Structured Outputs  | Consistent output formatting across agent runs
