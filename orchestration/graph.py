"""
Builds and compiles the LangGraph supervisor graph.

Phase 3 topology (guardrails added):

    START → input_guard
                ↓
        "blocked" ──────────────────────────→ END
        "pass"  → supervisor
                      ↓ (conditional on state["route"])
               ┌──────┼──────┐
              rag    sql    tool
               └──────┼──────┘
                      ↓
                output_guard → END

The compiled graph is a module-level singleton — built once at import time
and reused for every request.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from orchestration.state import AgentState
from orchestration.supervisor import supervisor_node
from guardrails.input_guard import check_input
from guardrails.output_guard import check_output
from agents.rag_agent import ask as rag_ask
from agents.sql_agent import run as sql_run
from agents.tool_agent import run as tool_run


# ---------------------------------------------------------------------------
# Guardrail nodes
# ---------------------------------------------------------------------------

def input_guard_node(state: AgentState) -> dict:
    """
    Run input checks. If blocked, short-circuit with a rejection answer.
    Sets route="blocked" so the conditional edge skips the supervisor.
    """
    result = check_input(state["question"])
    if not result["passed"]:
        reason = result["checks"][-1]["detail"]
        return {
            "route":             "blocked",
            "answer":            f"Request blocked by input guardrail: {reason}",
            "agent_used":        "blocked",
            "guardrail_results": {"input": result, "output": {}},
        }
    # Pass: store input result, leave route empty for supervisor to fill.
    return {
        "guardrail_results": {"input": result, "output": {}},
    }


def output_guard_node(state: AgentState) -> dict:
    """
    Run output checks on the agent's answer.
    Replaces answer with PII-masked version; flags toxicity in results.
    """
    result = check_output(state["answer"])
    existing_input = state.get("guardrail_results", {}).get("input", {})
    return {
        "answer": result["answer"],   # PII-masked text
        "guardrail_results": {
            "input":  existing_input,
            "output": {
                "passed": result["passed"],
                "checks": result["checks"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Agent nodes
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
# Edge routing functions
# ---------------------------------------------------------------------------

def _after_input_guard(state: AgentState) -> str:
    """Route to END if blocked, otherwise continue to supervisor."""
    return "blocked" if state["route"] == "blocked" else "pass"


def _pick_agent(state: AgentState) -> str:
    """Return the route set by the supervisor node."""
    return state["route"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build() -> StateGraph:
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("input_guard",  input_guard_node)
    builder.add_node("supervisor",   supervisor_node)
    builder.add_node("rag",          rag_node)
    builder.add_node("sql",          sql_node)
    builder.add_node("tool",         tool_node)
    builder.add_node("output_guard", output_guard_node)

    # Entry point is now input_guard (was supervisor in Phase 2)
    builder.set_entry_point("input_guard")

    # After input_guard: blocked → END, pass → supervisor
    builder.add_conditional_edges(
        "input_guard",
        _after_input_guard,
        {"blocked": END, "pass": "supervisor"},
    )

    # After supervisor: branch to the right agent
    builder.add_conditional_edges(
        "supervisor",
        _pick_agent,
        {"rag": "rag", "sql": "sql", "tool": "tool"},
    )

    # All agent nodes feed into output_guard (not directly to END)
    builder.add_edge("rag",  "output_guard")
    builder.add_edge("sql",  "output_guard")
    builder.add_edge("tool", "output_guard")

    # output_guard is the single exit point
    builder.add_edge("output_guard", END)

    return builder.compile()


# Compiled at import time — reused across all requests
graph = _build()
