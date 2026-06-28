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

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from orchestration.state import AgentState
from orchestration.supervisor import supervisor_node
from guardrails.input_guard import check_input
from guardrails.output_guard import check_output
from agents.rag_agent import ask as rag_ask
from agents.sql_agent import run as sql_run
from agents.tool_agent import run as tool_run


# ---------------------------------------------------------------------------
# Conversation-context helper
# ---------------------------------------------------------------------------

def _contextual_question(state: AgentState) -> str:
    """
    Return the current question enriched with recent conversation history.

    state["messages"] ends with the current HumanMessage; everything before it
    is prior turns. We prepend at most 6 prior messages (3 exchanges) as a
    short context block so the agent LLM can resolve follow-up references like
    "what about the pending ones?" without seeing the bare question alone.

    If there is no history the bare question is returned unchanged.
    """
    history = state.get("messages", [])
    prior = history[:-1][-6:] if len(history) > 1 else []
    if not prior:
        return state["question"]
    lines = []
    for m in prior:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content[:300]}")
    context = "\n".join(lines)
    return f"[Conversation history]\n{context}\n\n[Current question]\n{state['question']}"


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
    doc_filter = state.get("source_document") or None
    # When a specific document is selected the answer must come from its chunks,
    # not from conversation memory that may contain facts about a different person/doc.
    # Skip the history prefix so retrieved context cannot be overridden by prior turns.
    question = state["question"] if doc_filter else _contextual_question(state)
    result = rag_ask(question, source_document=doc_filter)
    # Return only the new AIMessage — the add_messages reducer appends it.
    return {
        "answer":     result["answer"],
        "sources":    result["sources"],
        "agent_used": "rag",
        "messages":   [AIMessage(content=result["answer"])],
    }


def sql_node(state: AgentState) -> dict:
    """Invoke the SQL agent and populate answer in state."""
    answer = sql_run(_contextual_question(state))
    return {
        "answer":     answer,
        "sources":    [],
        "agent_used": "sql",
        "messages":   [AIMessage(content=answer)],
    }


async def tool_node(state: AgentState) -> dict:
    """Invoke the native tool agent (async) and populate answer in state."""
    answer = await tool_run(_contextual_question(state))
    return {
        "answer":     answer,
        "sources":    [],
        "agent_used": "tool",
        "messages":   [AIMessage(content=answer)],
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

    # MemorySaver persists AgentState (including the full messages list) keyed
    # by thread_id. Each /agent request passes its session_id as thread_id so
    # conversation history accumulates across turns within a session.
    return builder.compile(checkpointer=MemorySaver())


# Compiled at import time — reused across all requests
graph = _build()
