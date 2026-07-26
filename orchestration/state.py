"""
AgentState for the looping supervisor architecture.

Key design — scratchpad uses the `add` reducer:
  Every worker node appends its raw result dict to scratchpad rather than
  overwriting anything.  The supervisor reads the full accumulated list
  on each loop iteration to decide whether to call another agent or emit
  "done".  This is what makes the system genuinely multi-agentic: state
  accumulates rather than being replaced.

agents_used also accumulates (add reducer) so the metrics layer sees the
full ordered list of every agent that ran during a request.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Core ─────────────────────────────────────────────────────────────────
    question: str                                        # user question, set at entry each turn
    messages: Annotated[list[BaseMessage], add_messages] # full conversation history (appends)

    # ── Scratchpad — accumulates raw agent outputs across supervisor iterations ─
    # Plain list, NO reducer.  Each node does:
    #   return {"scratchpad": state["scratchpad"] + [new_entry]}
    # so values build up within a single request.  Setting scratchpad=[] in
    # initial_state overwrites the checkpoint and resets it for each new request.
    scratchpad: list[dict]

    # ── Routing ───────────────────────────────────────────────────────────────
    route: str          # supervisor's latest decision (sql|rag|finance|research|clarify|done)
    iteration_count: int  # circuit-breaker counter; hard stop at MAX_SUPERVISOR_ITERATIONS

    # ── Output ────────────────────────────────────────────────────────────────
    answer: str            # final prose answer written by response node
    sources: list[dict]    # RAG chunk metadata for the API response

    # ── Document context (persisted across turns by MemorySaver) ─────────────
    source_document: str | None    # active PDF filter filename
    clarified_source: str | None   # set by clarify node after user answers

    # ── Follow-up tracking (persisted across turns) ───────────────────────────
    # response_node sets this to whatever question is now "outstanding" after
    # its answer (or None) -- via orchestration.followup.pending_followup_for(),
    # which tries the deterministic suggest_followup() table first, then falls
    # back to any free-form question the chat LLM's own answer ends on (e.g.
    # "which product and quantity would you like to generate an invoice
    # for?"). The NEXT turn's api/main.py graph_input deliberately OMITS this
    # key so the checkpointed value survives into supervisor_node -- that's
    # what lets a bare "ok"/"yes"/"show me"/a direct answer be resolved
    # against the specific thing that was just asked, instead of the router
    # guessing blind. Without this, that text only ever existed in the API
    # response's separate `followup` field for the UI chip -- never in
    # `messages`, so the router had no record of it at all.
    pending_followup: str | None

    # ── Purchase confirmation (Human-in-the-Loop) ─────────────────────────────
    # Set True by orchestration/confirm.py's confirm_purchase_node once the
    # user explicitly approves a pending invoice this turn. Reset to False
    # every fresh turn (api/main.py's graph_input) -- this only needs to
    # survive WITHIN a single turn's supervisor loop iterations, not across
    # turns (unlike pending_followup/source_document).
    order_confirmed: bool

    # ── Guardrails ────────────────────────────────────────────────────────────
    guardrail_results: dict        # {"input": {...}, "output": {...}}

    # ── Metrics / observability ───────────────────────────────────────────────
    agent_used: str        # primary agent label for API response
    agents_used: list[str] # plain list, reset to [] in initial_state each request
