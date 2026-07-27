"""
Confirm-purchase node: Human-in-the-Loop via LangGraph interrupt(), applied
to a DIFFERENT kind of decision than orchestration/clarify.py's.

clarify.py resolves "which agent should answer this" (an ambiguity about
routing). This node instead gates a CONSEQUENTIAL action -- an invoice,
i.e. an actual order -- behind an explicit human approve/reject before it's
considered final. Same underlying mechanism (interrupt()/Command(resume=...)
via orchestration/graph.py's get_pending_interrupt() and api/main.py's
pause/resume wiring), different reason to use it.

Flow:
  1. finance_node runs as normal and computes the full invoice (cheap, no
     external side effects -- just DB reads + math, nothing is actually
     charged or shelved in this demo). The computed invoice is appended to
     the scratchpad exactly like any other finance result.
  2. supervisor_node's "finance ran" hard stop (see supervisor.py) checks
     whether that result looks like an invoice and hasn't been confirmed
     yet this turn -- if so, routes to "confirm_purchase" instead of "done".
  3. confirm_purchase_node pauses here, showing the already-computed invoice
     as the confirmation prompt.
  4. On resume: approve -> leaves the scratchpad as-is, sets
     order_confirmed=True, loops back to the supervisor (which now routes
     to "done" since the result still looks like an invoice, but
     order_confirmed is True). Reject -> REPLACES the finance scratchpad
     entry with a cancellation note, so response_node's finance-always-wins
     pass-through reflects the cancellation, not an invoice that was never
     actually approved.
"""
import logging
import re

from langgraph.types import interrupt

from orchestration.state import AgentState

logger = logging.getLogger(__name__)

# Deliberately narrow, whole-word matches -- same rationale as every other
# regex classifier in this codebase: a real reply that happens to CONTAIN
# "no" as part of a longer word ("know", "now") must never misfire.
_APPROVE_RE = re.compile(
    r'\b(approve|yes|confirm|confirmed|ok(ay)?|go\s*ahead|place\s+(the\s+)?order|proceed)\b',
    re.IGNORECASE,
)
_REJECT_RE = re.compile(
    r'\b(reject|no|cancel|don\'?t|do\s+not|stop|abort|never\s*mind)\b',
    re.IGNORECASE,
)


def _looks_like_invoice(result: str) -> bool:
    """Cheap, deterministic check -- generate_invoice's own output always
    contains this exact heading (see agents/finance_agent.py)."""
    return "Invoice #" in result


def confirm_purchase_node(state: AgentState) -> dict:
    """
    Pause the graph and require an explicit approve/reject before an
    already-computed invoice is considered final.

    The return value of interrupt() is the user's answer supplied by the
    FastAPI endpoint via Command(resume=<user_answer>). Execution resumes
    from the line immediately after interrupt().
    """
    scratchpad      = state.get("scratchpad", [])
    finance_entries = [e for e in scratchpad if e.get("agent") == "finance"]
    invoice_text    = str(finance_entries[-1].get("result", "")) if finance_entries else ""

    print(f"[confirm_purchase] pausing for approval — invoice: {invoice_text[:80]!r}")

    user_answer: str = interrupt(
        "Please confirm this order:\n\n" + invoice_text +
        "\n\nReply **approve** to place the order, or **reject** to cancel it."
    )
    answer = str(user_answer).strip()

    # Reject checked first: "no, don't approve" must not match _APPROVE_RE's
    # bare "approve"/"ok" alternatives via a coincidental substring, and an
    # explicit "no"/"cancel" should always win over an ambiguous mixed reply.
    if _REJECT_RE.search(answer) and not _APPROVE_RE.search(answer):
        print("[confirm_purchase] rejected")
        new_scratchpad = [
            e if e.get("agent") != "finance" else {
                "agent": "finance",
                "result": "Order cancelled by user — no invoice was generated.",
            }
            for e in scratchpad
        ]
        return {"scratchpad": new_scratchpad, "order_confirmed": False}

    if _APPROVE_RE.search(answer):
        print("[confirm_purchase] approved")
        new_scratchpad = [
            e if e.get("agent") != "finance" else {
                "agent": "finance",
                "result": "✅ **Order confirmed and placed.**\n\n" + str(e.get("result", "")),
            }
            for e in scratchpad
        ]
        return {"scratchpad": new_scratchpad, "order_confirmed": True}

    # Neither pattern matched (e.g. "maybe", "what's the return policy?") --
    # treat as not-yet-decided rather than guessing either way. Since
    # order_confirmed stays False and the scratchpad's invoice is untouched,
    # the supervisor's "looks like invoice + not confirmed" check will route
    # back here again next iteration... but there is no next interrupt
    # without a new user message, so this degrades to "done" via the
    # iteration-count circuit breaker -- treated the same as a rejection,
    # since an unconfirmed order must never be reported as placed.
    print(f"[confirm_purchase] ambiguous reply {answer!r} -- treating as not approved")
    new_scratchpad = [
        e if e.get("agent") != "finance" else {
            "agent": "finance",
            "result": (
                "I didn't catch a clear approve/reject, so this order was "
                "NOT placed. Please ask again and reply 'approve' or 'reject'."
            ),
        }
        for e in scratchpad
    ]
    return {"scratchpad": new_scratchpad, "order_confirmed": False}
