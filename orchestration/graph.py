"""
Looping supervisor graph.

Architecture (NOT a linear pipeline):

                                ┌──────────────────────────────────┐
  START                         │  SUPERVISOR  (LLM, temp=0)       │
    │                           │  reads full scratchpad each turn  │
    v                           │  picks exactly one next action    │
  input_guard                   └──────────────────────────────────┘
    │  (blocked → skip)                  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
    v                                    │  │  │  │  │  │  │  │  │
  supervisor ──sql──────────────────────┘  │  │  │  │  │  │  │  │
             ──rag─────────────────────────┘  │  │  │  │  │  │  │
             ──finance─────────────────────────┘  │  │  │  │  │
             ──research────────────────────────────┘  │  │  │  │
             ──credit───────────────────────────────────┘  │  │
             ──risk──────────────────────────────────────────┘  │
             ──forecast────────────────────────────────────────────┘
             ──clarify──────────────────────────────────────────────┤
             ──confirm_purchase────────────────────────────────────────┤
             ──confirm_loan──────────────────────────────────────────────┘
             ──done──> response ──> output_guard ──> END

The nine loop-back edges (sql/rag/finance/research/credit/risk/forecast/
clarify/confirm_purchase/confirm_loan → supervisor) are what make this
multi-agentic: the supervisor sees accumulated scratchpad results and
decides what to do next on every iteration.

credit/risk/forecast (agents/credit_agent.py, risk_agent.py,
forecast_agent.py) are Phase 2 additions splitting what used to be finance-
only concerns (churn prediction, revenue forecasting) into dedicated agents,
plus genuinely new capability (credit eligibility, EMI, loan proposals,
fraud/anomaly checks). request_loan (credit_agent) itself calls into
risk_agent.predict_customer_risk to factor churn signal into a loan
recommendation — a real cross-agent dependency, not just parallel tools.

confirm_purchase and confirm_loan are SECOND/THIRD interrupt-based
Human-in-the-Loop nodes (see orchestration/confirm.py,
orchestration/confirm_loan.py) — distinct from clarify's "which agent
should answer this" ambiguity, each gates a CONSEQUENTIAL action (an
invoice / a loan proposal) behind an explicit approve/reject. The
supervisor routes to whichever one applies (not "done") whenever the
relevant agent's last result looks like an invoice/proposal and hasn't been
confirmed yet this turn.

The response node is SEPARATE from the supervisor — it only runs when the
supervisor emits "done", turning raw scratchpad into a final prose answer.

The checkpointer (AsyncSqliteSaver, backed by data/checkpoints.db) provides BOTH:
  - multi-turn conversation memory (keyed by thread_id = session_id), surviving
    server restarts
  - interrupt/resume support for the clarify node
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from observability.trace import trace_agent_start
from orchestration.state import AgentState
from orchestration.supervisor import _route, supervisor_node
from orchestration.clarify import clarify_node
from orchestration.confirm import confirm_purchase_node
from orchestration.confirm_loan import confirm_loan_node

load_dotenv()

# SQLite-backed checkpoint DB — same data/ directory as chats.db and company.db.
# Survives server restarts, unlike the in-memory MemorySaver this replaces.
CHECKPOINT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "checkpoints.db"
)

# Holds the open AsyncSqliteSaver context manager so it can be closed cleanly
# on shutdown. None until init_sqlite_checkpointer() runs.
_checkpointer_cm = None


# ── Guard nodes ───────────────────────────────────────────────────────────────

def input_guard_node(state: AgentState) -> dict:
    """Block prompt injections and toxicity at the graph entry point."""
    from guardrails.input_guard import check_input
    result = check_input(state["question"])
    if not result.get("passed", True):
        reason = (result.get("checks") or [{}])[-1].get("detail", "Blocked.")
        return {
            "answer":            f"Request blocked: {reason}",
            "agent_used":        "blocked",
            "guardrail_results": {"input": result, "output": {}},
        }
    return {"guardrail_results": {"input": result, "output": {}}}


def output_guard_node(state: AgentState) -> dict:
    """Mask PII and check toxicity on the final answer before returning to user."""
    from guardrails.output_guard import check_output
    answer = state.get("answer", "")
    result = check_output(answer, agents_used=state.get("agents_used", []))
    return {
        "answer": result.get("answer", answer),
        "guardrail_results": {
            **state.get("guardrail_results", {}),
            "output": {"passed": result["passed"], "checks": result["checks"]},
        },
    }


# ── Context helper ────────────────────────────────────────────────────────────

def _contextual_question(state: AgentState, n: int = 6) -> str:
    """
    Prepend recent conversation history to the question.
    SQL and Finance agents need this to resolve pronouns like "their balance"
    across multi-turn conversations.  RAG uses the raw question so that
    pronoun-laden queries don't degrade embedding similarity.
    """
    history = state.get("messages", [])
    prior   = history[:-1][-n:]
    if not prior:
        return state["question"]
    lines = []
    for m in prior:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content[:300]}")
    history_text = "\n".join(lines)
    return (
        f"[Conversation history]\n{history_text}\n\n"
        f"[Current question]\n{state['question']}"
    )


# ── Agent nodes ───────────────────────────────────────────────────────────────
# Each node runs ONE agent, appends its result to the scratchpad via the `add`
# reducer, then the loop-back edge returns control to the supervisor.
# Imports are inside functions so the graph file is importable even before
# every agent file is fully implemented.

def sql_node(state: AgentState) -> dict:
    """Query the transactions SQLite database."""
    from agents.sql_agent import run
    trace_agent_start("SQL", state["question"])
    result = run(
        # Full context (with history) goes to the LLM for SQL generation —
        # helps resolve pronouns like "their balance" across turns.
        question=_contextual_question(state),
        # Raw current question goes to entity extraction ONLY — must not
        # include history because prior answers may contain CUST IDs that
        # would falsely scope an all-customers query to a single customer.
        entity_question=state["question"],
    )
    return {
        "scratchpad":  state.get("scratchpad", []) + [{"agent": "sql", "result": result}],
        "agents_used": state.get("agents_used", []) + ["sql"],
    }


async def rag_node(state: AgentState) -> dict:
    """Retrieve from ChromaDB using the selected PDF document filter."""
    from agents.rag_agent import ask
    trace_agent_start("RAG", state["question"])
    doc     = state.get("clarified_source") or state.get("source_document")
    result  = ask(question=state["question"], k=4, source_document=doc)
    sources = result.get("sources", [])
    return {
        "scratchpad":  state.get("scratchpad", []) + [{"agent": "rag", "result": result.get("answer", ""), "sources": sources}],
        "agents_used": state.get("agents_used", []) + ["rag"],
        "sources":     sources,
    }


async def finance_node(state: AgentState) -> dict:
    """
    Financial calculations.  Injects prior SQL and RAG results so the Finance
    agent can compute from numbers already gathered without fetching them again.
    """
    from agents.finance_agent import run
    trace_agent_start("FINANCE", state["question"])
    q = _contextual_question(state)

    # Map scratchpad into the format finance_agent.run() expects
    knowledge_result: dict = {}
    for entry in state.get("scratchpad", []):
        if entry.get("agent") == "sql":
            knowledge_result["transactions"] = entry.get("result", "")
        elif entry.get("agent") == "rag":
            knowledge_result["documents"] = entry.get("result", "")

    result = await run(
        question=q,
        knowledge_result=knowledge_result if knowledge_result else None,
        messages=state.get("messages", []),
    )
    return {
        "scratchpad":  state.get("scratchpad", []) + [{"agent": "finance", "result": result}],
        "agents_used": state.get("agents_used", []) + ["finance"],
    }


async def research_node(state: AgentState) -> dict:
    """Web search for external benchmarks / industry data."""
    from agents.research_agent import run
    trace_agent_start("RESEARCH", state["question"])
    result = await run(state["question"])
    return {
        "scratchpad":  state.get("scratchpad", []) + [{"agent": "research", "result": result}],
        "agents_used": state.get("agents_used", []) + ["research"],
    }


async def credit_node(state: AgentState) -> dict:
    """Credit eligibility, EMI, and loan proposals."""
    from agents.credit_agent import run
    trace_agent_start("CREDIT", state["question"])
    result = await run(question=_contextual_question(state), messages=state.get("messages", []))
    return {
        "scratchpad":  state.get("scratchpad", []) + [{"agent": "credit", "result": result}],
        "agents_used": state.get("agents_used", []) + ["credit"],
    }


async def risk_node(state: AgentState) -> dict:
    """Churn/retention risk and transaction fraud/anomaly checks."""
    from agents.risk_agent import run
    trace_agent_start("RISK", state["question"])
    result = await run(question=_contextual_question(state), messages=state.get("messages", []))
    return {
        "scratchpad":  state.get("scratchpad", []) + [{"agent": "risk", "result": result}],
        "agents_used": state.get("agents_used", []) + ["risk"],
    }


async def forecast_node(state: AgentState) -> dict:
    """Flexible day/week/month/year revenue forecasting."""
    from agents.forecast_agent import run
    trace_agent_start("FORECAST", state["question"])
    result = await run(question=_contextual_question(state), messages=state.get("messages", []))
    return {
        "scratchpad":  state.get("scratchpad", []) + [{"agent": "forecast", "result": result}],
        "agents_used": state.get("agents_used", []) + ["forecast"],
    }


async def response_node(state: AgentState) -> dict:
    """
    Fuse the accumulated scratchpad into one clean prose answer.
    Runs ONLY after the supervisor emits "done" — it is entirely separate
    from the routing logic.
    """
    from agents.response_agent import run as response_run
    from orchestration.followup import pending_followup_for
    trace_agent_start("RESPONSE", state["question"])
    scratchpad = state.get("scratchpad", [])
    answer     = await response_run(state["question"], scratchpad, state.get("messages", []))

    # pending_followup_for() tries the deterministic suggest_followup() table
    # FIRST (precise, keyed to the exact finance tool that ran), and falls
    # back to whatever question the answer itself ends on if that doesn't
    # match -- covers the chat LLM asking its OWN free-form clarifying
    # question (e.g. "which product and quantity...?"), which used to leave
    # no trace at all for the next turn's router to resolve a bare "yes"/
    # "ok" reply against.
    #
    # Computed here (not in api/main.py) so it can also be written into
    # pending_followup -- the single source of truth both the API response's
    # `followup` field AND next turn's supervisor routing read from.
    followup = pending_followup_for(answer)

    # Derive a human-readable agent label for the API response
    agents = state.get("agents_used", [])
    _primary = {"sql", "rag", "finance", "credit", "risk", "forecast", "research"}
    if len([a for a in agents if a in _primary]) > 1:
        agent_label = "synthesis"
    elif "finance" in agents:
        agent_label = "finance"
    elif "credit" in agents:
        agent_label = "credit"
    elif "risk" in agents:
        agent_label = "risk"
    elif "forecast" in agents:
        agent_label = "forecast"
    elif "rag" in agents:
        agent_label = "rag"
    elif "sql" in agents:
        agent_label = "sql"
    elif "research" in agents:
        agent_label = "research"
    else:
        agent_label = "chat"

    return {
        "answer":           answer,
        "agent_used":       agent_label,
        "agents_used":      agents + ["response"],
        "messages":         [AIMessage(content=answer)],
        "pending_followup": followup,
    }


# ── Edge routing ──────────────────────────────────────────────────────────────

def _after_input_guard(state: AgentState) -> str:
    """
    If input guard blocked the request it already wrote state["answer"].
    Skip the entire supervisor loop and go straight to output_guard.
    """
    return "output_guard" if state.get("answer") else "supervisor"


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_wiring() -> StateGraph:
    """Register nodes and edges. Returns the uncompiled builder."""
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("input_guard",      input_guard_node)
    builder.add_node("supervisor",       supervisor_node)
    builder.add_node("clarify",          clarify_node)
    builder.add_node("confirm_purchase", confirm_purchase_node)
    builder.add_node("confirm_loan",     confirm_loan_node)
    builder.add_node("sql",          sql_node)
    builder.add_node("rag",          rag_node)
    builder.add_node("finance",      finance_node)
    builder.add_node("research",     research_node)
    builder.add_node("credit",       credit_node)
    builder.add_node("risk",         risk_node)
    builder.add_node("forecast",     forecast_node)
    builder.add_node("response",     response_node)
    builder.add_node("output_guard", output_guard_node)

    # ── Entry ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "input_guard")

    # input_guard → supervisor (normal) or output_guard (blocked)
    builder.add_conditional_edges("input_guard", _after_input_guard, {
        "supervisor":   "supervisor",
        "output_guard": "output_guard",
    })

    # ── Supervisor hub — dynamic routing decided each iteration ───────────────
    builder.add_conditional_edges("supervisor", _route, {
        "sql":      "sql",
        "rag":      "rag",
        "finance":  "finance",
        "research": "research",
        "credit":   "credit",
        "risk":     "risk",
        "forecast": "forecast",
        "clarify":  "clarify",
        "confirm_purchase": "confirm_purchase",
        "confirm_loan":     "confirm_loan",
        "done":     "response",       # only exit from the supervisor loop
        # "unclear" — a genuinely ambiguous question with a preset canned
        # answer already in state["answer"] (set by supervisor_node). Skips
        # response_node entirely so its LLM never gets a chance to invent a
        # follow-up question about a product/customer that was never
        # mentioned — same bypass pattern as the blocked-input path below.
        "unclear":  "output_guard",
    })

    # ── LOOP-BACK EDGES ───────────────────────────────────────────────────────
    # Every worker returns to the supervisor so it can re-read the scratchpad
    # and decide the next step.  This is the defining property of the looping
    # multi-agent architecture — without these edges it would be a pipeline.
    builder.add_edge("sql",      "supervisor")   # ← loop back
    builder.add_edge("rag",      "supervisor")   # ← loop back
    builder.add_edge("finance",  "supervisor")   # ← loop back
    builder.add_edge("research", "supervisor")   # ← loop back
    builder.add_edge("credit",   "supervisor")   # ← loop back
    builder.add_edge("risk",     "supervisor")   # ← loop back
    builder.add_edge("forecast", "supervisor")   # ← loop back
    builder.add_edge("clarify",  "supervisor")   # ← loop back (after user answers)
    builder.add_edge("confirm_purchase", "supervisor")  # ← loop back (after approve/reject)
    builder.add_edge("confirm_loan",     "supervisor")  # ← loop back (after approve/reject)

    # ── Terminal path — runs only once, after supervisor emits "done" ─────────
    builder.add_edge("response",     "output_guard")
    builder.add_edge("output_guard", END)

    return builder


# ── Checkpointer lifecycle ────────────────────────────────────────────────────
#
# Compiled once at import time with an in-memory MemorySaver as a safe default —
# this keeps `graph` importable and usable synchronously (scripts, tests) even
# before init_sqlite_checkpointer() has run.
#
# init_sqlite_checkpointer() (called from FastAPI's startup hook) swaps in a
# database-backed AsyncSqliteSaver by mutating `graph.checkpointer` in place —
# the compiled graph object's identity never changes, so the `graph` reference
# already imported elsewhere (api/main.py) picks up the swap automatically.
graph = _build_wiring().compile(checkpointer=MemorySaver())


async def init_sqlite_checkpointer() -> None:
    """
    Open the SQLite checkpoint DB and swap it in as the graph's checkpointer.

    Uses AsyncSqliteSaver (not the sync SqliteSaver) because api/main.py calls
    graph.ainvoke() — the sync SqliteSaver doesn't implement aput_writes, which
    LangGraph's async run loop calls on every node write and would raise
    NotImplementedError on the very first request.
    """
    global _checkpointer_cm
    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
    _checkpointer_cm = AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
    saver = await _checkpointer_cm.__aenter__()
    await saver.setup()
    graph.checkpointer = saver
    print(f"[graph] Checkpointer: AsyncSqliteSaver -> {CHECKPOINT_DB_PATH}")


async def close_sqlite_checkpointer() -> None:
    """Close the aiosqlite connection cleanly on app shutdown."""
    global _checkpointer_cm
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
        _checkpointer_cm = None


async def has_checkpoint(thread_id: str) -> bool:
    """
    True if the graph already has persisted state for this thread_id.

    Used by api/main.py to decide whether to seed initial_state["messages"]
    from the SQLite chat history — only needed for a thread the checkpointer
    has never seen (e.g. an old chat opened after the checkpoint DB was wiped
    or a message pruned); a thread with an existing checkpoint already has its
    full message history via the `add_messages` reducer, and re-seeding it
    from chat_store on every turn would duplicate messages indefinitely.
    """
    config = {"configurable": {"thread_id": thread_id}}
    tup = await graph.checkpointer.aget_tuple(config)
    return tup is not None


async def _pending_interrupt_task(thread_id: str):
    """
    The PregelTask currently paused on an interrupt for this thread, or None.

    None of clarify_node's, confirm_purchase_node's, or confirm_loan_node's
    interrupt() calls surface via graph.ainvoke()'s return value in this
    LangGraph version — invoke just returns the state as of the last
    completed node, unchanged. The only way to detect a pause is
    graph.aget_state(): a non-empty `.next` means the graph stopped mid-turn
    with unresolved tasks, and the paused task's `.interrupts` holds the
    Interrupt object whose `.value` is whatever text that node passed to
    interrupt(); `.name` is the node name ("clarify", "confirm_purchase", or
    "confirm_loan"), used by api/main.py to label the paused response with
    the right agent_used.

    Safe to call for a thread_id the checkpointer has never seen — returns
    an empty snapshot (next=(), tasks=()), not an error.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        if task.interrupts:
            return task
    return None


async def get_pending_interrupt(thread_id: str) -> str | None:
    """The question/prompt text this thread is currently paused on (from
    whichever HITL node — clarify or confirm_purchase — triggered it), or
    None if it isn't paused."""
    task = await _pending_interrupt_task(thread_id)
    return str(task.interrupts[0].value) if task else None


async def get_pending_interrupt_node(thread_id: str) -> str | None:
    """Name of the node currently paused on an interrupt for this thread
    (e.g. "clarify", "confirm_purchase"), or None if not paused."""
    task = await _pending_interrupt_task(thread_id)
    return task.name if task else None
