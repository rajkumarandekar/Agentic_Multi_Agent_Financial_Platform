"""
Tests for the confirm_loan Human-in-the-Loop node (orchestration/confirm_loan.py).

Mirrors tests/test_confirm_purchase.py exactly -- same interrupt()/
Command(resume=...) mechanism, same approve/reject regex contract (reused
directly from orchestration/confirm.py, not duplicated), applied to a loan
proposal instead of a purchase invoice.

The loan-proposing question is handled entirely by credit_agent's
deterministic _fast_dispatch -- no Groq call for the credit step itself.
Only the SUPERVISOR's router is mocked (straight to "credit").
"""
import uuid

import pytest
from langgraph.types import Command

import orchestration.confirm_loan as confirm_loan_mod
import orchestration.supervisor as sup
from orchestration.graph import graph, get_pending_interrupt, get_pending_interrupt_node

_LOAN_QUESTION = "apply for a loan of rs.50000 for CUST001 for 12 months"


def _initial_state(question: str) -> dict:
    return {
        "question": question,
        "messages": [],
        "scratchpad": [],
        "route": "",
        "answer": "",
        "sources": [],
        "agent_used": "",
        "agents_used": [],
        "iteration_count": 0,
        "source_document": None,
        "clarified_source": None,
        "pending_followup": None,
        "order_confirmed": False,
        "loan_confirmed": False,
        "guardrail_results": {},
    }


class TestLooksLikeLoanProposal:
    def test_detects_loan_proposal_heading(self):
        assert confirm_loan_mod._looks_like_loan_proposal("**Loan Proposal #LNP-TM-...**")

    def test_rejects_unrelated_text(self):
        assert not confirm_loan_mod._looks_like_loan_proposal("Invoice #INV-TM-...")


@pytest.mark.asyncio
class TestConfirmLoanEndToEnd:
    async def test_loan_pauses_then_approve_finalizes_it(self, monkeypatch):
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "credit")

        result = await graph.ainvoke(_initial_state(_LOAN_QUESTION), config=config)
        assert "answer" not in result or not result.get("answer")

        assert await get_pending_interrupt_node(thread_id) == "confirm_loan"
        prompt = await get_pending_interrupt(thread_id)
        assert "Loan Proposal #" in prompt
        assert "approve" in prompt.lower() and "reject" in prompt.lower()

        result2 = await graph.ainvoke(Command(resume="approve"), config=config)
        assert await get_pending_interrupt(thread_id) is None
        assert result2["loan_confirmed"] is True
        assert "Loan Proposal #" in result2["answer"]
        # Real bug this guards against: the proposal card already contains
        # "Recommendation: APPROVE" (the system's PRE-approval business
        # recommendation) -- without a distinct banner, a user can't tell
        # whether their own "approve" reply did anything at all.
        assert "Loan approved and finalized" in result2["answer"]
        # Second real bug: request_loan's own text ends with "This is a
        # proposal only — requires explicit approval..." -- correct before
        # approval, flatly contradictory once actually approved. Must be
        # stripped, not just have the banner prepended on top of it.
        assert "proposal only" not in result2["answer"].lower()

    async def test_reject_cancels_without_ever_finalizing(self, monkeypatch):
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "credit")

        await graph.ainvoke(_initial_state(_LOAN_QUESTION), config=config)
        assert await get_pending_interrupt_node(thread_id) == "confirm_loan"

        result2 = await graph.ainvoke(Command(resume="reject"), config=config)
        assert await get_pending_interrupt(thread_id) is None
        assert result2["loan_confirmed"] is False
        assert "cancelled" in result2["answer"].lower()
        assert "Loan Proposal #" not in result2["answer"]

    async def test_ambiguous_reply_does_not_finalize_the_loan(self, monkeypatch):
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "credit")

        await graph.ainvoke(_initial_state(_LOAN_QUESTION), config=config)

        result2 = await graph.ainvoke(Command(resume="maybe later"), config=config)
        assert result2["loan_confirmed"] is False
        assert "Loan Proposal #" not in result2["answer"]
