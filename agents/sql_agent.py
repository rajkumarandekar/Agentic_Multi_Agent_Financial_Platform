"""
SQL agent: answers natural-language questions by generating and running SQL
against the local shipments SQLite database.

Uses LangChain's create_sql_agent, which internally runs a ReAct loop:
  question → generate SQL → execute → inspect result → answer.
"""

import os

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
DB_PATH    = os.getenv("DB_PATH", "data/shipments.db")


def run(question: str) -> str:
    """
    Translate a natural-language question into SQL, run it, and return the answer.

    Args:
        question: A question about the shipments data, e.g.
                  "How many shipments are in transit?"

    Returns:
        A natural-language answer derived from the SQL result.
    """
    db  = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    # create_sql_agent wraps the DB as a toolkit of tools (list_tables,
    # schema, query_checker, query_sql) and builds an AgentExecutor.
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",  # tool-calling style — cleaner output than ReAct
        verbose=False,
    )
    result = agent.invoke({"input": question})
    return result["output"]
