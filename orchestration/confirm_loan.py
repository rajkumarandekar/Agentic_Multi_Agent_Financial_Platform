"""
Confirm-loan node: Human-in-the-Loop via LangGraph interrupt(), reusing the
EXACT same pattern as orchestration/confirm.py's confirm_purchase_node --
same interrupt()/Command(resume=...) mechanism, same approve/reject regex
contract -- applied to a different consequential action: a loan proposal
instead of a purchase invoice.

Flow:
  1. credit_node runs request_loan, which computes the full proposal (EMI,
     credit-limit check, Risk agent's churn signal) -- appended to the
     scratchpad exactly like any other credit result.
  2. supervisor_node's "credit ran" hard stop (see supervisor.py) checks
     whether that result looks like a loan proposal and hasn't been
     confirmed yet this turn -- if so, routes to "confirm_loan" instead of
     "done".
  3. confirm_loan_node pauses here, showing the already-computed proposal
     as the confirmation prompt.
  4. On resume: approve -> PREPENDS a "Loan Approved" banner onto the same
     credit scratchpad entry (the proposal's own numbers don't change, they
     were already correct) and sets loan_confirmed=True, loops back to the
     supervisor (which now routes to "done"). Reject -> REPLACES the credit
     scratchpad entry with a cancellation note, so response_node's
     credit-pass-through reflects the cancellation, not a proposal that was
     never actually approved.

     The approve banner matters: the proposal card already contains a
     "Recommendation: APPROVE" field (the SYSTEM's pre-approval business
     recommendation, computed before any human ever saw it) -- without a
     distinct banner, the post-approval answer is byte-for-byte identical to
     the pending prompt minus its "please confirm" wrapper, so a real user
     genuinely cannot tell whether their "approve" reply did anything at all.
"""
import logging
import re

from langgraph.types import interrupt

from orchestration.confirm import _APPROVE_RE, _REJECT_RE
from orchestration.state import AgentState

logger = logging.getLogger(__name__)

# request_loan's own text always ends with this disclaimer (see
# agents/credit_agent.py) -- correct while the proposal is still pending,
# but flatly contradicts the "✅ Loan approved and finalized" banner once
# actually approved. Stripped out below rather than just prepending the
# banner on top of it, which is what an earlier version of this file did.
_PENDING_DISCLAIMER_RE = re.compile(
    r'\n*_This is a proposal only — requires explicit approval before the loan is final\._\n*'
)


def _looks_like_loan_proposal(result: str) -> bool:
    """Cheap, deterministic check -- request_loan's own output always
    contains this exact heading (see agents/credit_agent.py)."""
    return "Loan Proposal #" in result


def confirm_loan_node(state: AgentState) -> dict:
    """
    Pause the graph and require an explicit approve/reject before an
    already-computed loan proposal is considered final. Same mechanism as
    confirm_purchase_node -- interrupt()'s return value is the user's
    answer supplied via Command(resume=<user_answer>).
    """
    scratchpad     = state.get("scratchpad", [])
    credit_entries = [e for e in scratchpad if e.get("agent") == "credit"]
    proposal_text  = str(credit_entries[-1].get("result", "")) if credit_entries else ""

    print(f"[confirm_loan] pausing for approval — proposal: {proposal_text[:80]!r}")

    user_answer: str = interrupt(
        "Please confirm this loan proposal:\n\n" + proposal_text +
        "\n\nReply **approve** to finalize the loan, or **reject** to cancel it."
    )
    answer = str(user_answer).strip()

    if _REJECT_RE.search(answer) and not _APPROVE_RE.search(answer):
        print("[confirm_loan] rejected")
        new_scratchpad = [
            e if e.get("agent") != "credit" else {
                "agent": "credit",
                "result": "Loan proposal cancelled by user — no loan was created.",
            }
            for e in scratchpad
        ]
        return {"scratchpad": new_scratchpad, "loan_confirmed": False}

    if _APPROVE_RE.search(answer):
        print("[confirm_loan] approved")
        new_scratchpad = [
            e if e.get("agent") != "credit" else {
                "agent": "credit",
                "result": "✅ **Loan approved and finalized.**\n\n" + _PENDING_DISCLAIMER_RE.sub(
                    "\n", str(e.get("result", "")),
                ),
            }
            for e in scratchpad
        ]
        return {"scratchpad": new_scratchpad, "loan_confirmed": True}

    # Neither pattern matched -- treat as not-yet-decided, same rationale as
    # confirm_purchase_node: an unconfirmed loan must never be reported as
    # finalized.
    print(f"[confirm_loan] ambiguous reply {answer!r} -- treating as not approved")
    new_scratchpad = [
        e if e.get("agent") != "credit" else {
            "agent": "credit",
            "result": (
                "I didn't catch a clear approve/reject, so this loan was "
                "NOT finalized. Please ask again and reply 'approve' or 'reject'."
            ),
        }
        for e in scratchpad
    ]
    return {"scratchpad": new_scratchpad, "loan_confirmed": False}
