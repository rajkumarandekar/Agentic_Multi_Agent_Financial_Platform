"""
Builds and compiles the LangGraph supervisor graph.

Topology:
    START → supervisor_node
                ↓ (conditional edge on state["route"])
         ┌──────┼──────┐
        rag    sql    tool
         └──────┼──────┘
               END

The compiled graph is a module-level singleton — it is built once at import
time and reused for every request, avoiding redundant compilation overhead.
"""

from langchain_core.messages import AIMessage

from langgraph.graph import StateGraph, END

from orchestration.state import AgentState
from orchestration.supervisor import supervisor_node
from agents.rag_agent import ask as rag_ask
from agents.sql_agent import run as sql_run
from agents.tool_agent import run as tool_run


# ---------------------------------------------------------------------------
# Agent nodes — each reads state, calls the agent, writes back a partial dict
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    """Invoke the RAG agent and populate answer + sources in state."""
    result = rag_ask(state["question"])
    return {
        "answer":     result["answer"],
        "sources":    result["sources"],
        "agent_used": "rag",
        "messages":   state["messages"] + [AIMessage(content=result["answer"])],
    }


def sql_node(state: AgentState) -> dict:
    """Invoke the SQL agent and populate answer in state."""
    answer = sql_run(state["question"])
    return {
        "answer":     answer,
        "sources":    [],
        "agent_used": "sql",
        "messages":   state["messages"] + [AIMessage(content=answer)],
    }


async def tool_node(state: AgentState) -> dict:
    """Invoke the native tool agent (async) and populate answer in state."""
    answer = await tool_run(state["question"])
    return {
        "answer":     answer,
        "sources":    [],
        "agent_used": "tool",
        "messages":   state["messages"] + [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Conditional edge function — tells LangGraph which node to jump to next
# ---------------------------------------------------------------------------

def _pick_route(state: AgentState) -> str:
    """Return the route string set by the supervisor node."""
    return state["route"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build() -> StateGraph:
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag",        rag_node)
    builder.add_node("sql",        sql_node)
    builder.add_node("tool",       tool_node)

    # Entry point
    builder.set_entry_point("supervisor")

    # Conditional edges: supervisor decides, LangGraph dispatches
    builder.add_conditional_edges(
        "supervisor",
        _pick_route,
        {"rag": "rag", "sql": "sql", "tool": "tool"},
    )

    # Each agent node goes straight to END
    builder.add_edge("rag",  END)
    builder.add_edge("sql",  END)
    builder.add_edge("tool", END)

    return builder.compile()


# Compiled at import time — reused across all requests
graph = _build()
