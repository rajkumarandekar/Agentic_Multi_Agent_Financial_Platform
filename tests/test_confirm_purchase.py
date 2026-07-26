"""
Tests for the confirm_purchase Human-in-the-Loop node (orchestration/confirm.py).

Two layers:
  1. Approve/reject regex classification -- pure, network-free, fast.
  2. Full graph-level integration test using the REAL interrupt()/
     Command(resume=...) mechanism (same pattern as tests/test_clarify_resume.py),
     proving the whole pause -> approve/reject -> resume loop actually works,
     not just that the pieces exist in isolation.

The invoice-generating question used throughout ("generate an invoice for
CUST001 buying 2 units of PRD001") is handled entirely by finance_agent's
deterministic _fast_dispatch — no Groq call, no rate-limit risk, no mocking
needed for the finance step itself. Only the SUPERVISOR's router is mocked
(straight to "finance"), since that's the one real LLM call in this path.
"""
import uuid

import pytest
from langgraph.types import Command

import orchestration.confirm as confirm_mod
import orchestration.supervisor as sup
from orchestration.graph import graph, get_pending_interrupt, get_pending_interrupt_node

_INVOICE_QUESTION = "generate an invoice for CUST001 buying 2 units of PRD001"


class TestApproveRejectClassification:
    @pytest.mark.parametrize("text", [
        "approve", "Yes", "confirm", "confirmed", "ok", "okay",
        "go ahead", "proceed", "please place the order",
    ])
    def test_approve_words(self, text):
        assert confirm_mod._APPROVE_RE.search(text)

    @pytest.mark.parametrize("text", [
        "reject", "no", "cancel", "don't", "do not", "stop", "abort", "never mind",
    ])
    def test_reject_words(self, text):
        assert confirm_mod._REJECT_RE.search(text)

    def test_reject_wins_over_incidental_approve_substring(self):
        """'no, don't approve it' contains the bare word 'approve' but the
        user is clearly rejecting -- reject must win."""
        text = "no, don't approve it"
        assert confirm_mod._REJECT_RE.search(text)


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
        "guardrail_results": {},
    }


@pytest.mark.asyncio
class TestConfirmPurchaseEndToEnd:
    async def test_invoice_pauses_then_approve_finalizes_it(self, monkeypatch):
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "finance")

        result = await graph.ainvoke(_initial_state(_INVOICE_QUESTION), config=config)
        assert "answer" not in result or not result.get("answer")

        assert await get_pending_interrupt_node(thread_id) == "confirm_purchase"
        prompt = await get_pending_interrupt(thread_id)
        assert "Invoice #" in prompt
        assert "approve" in prompt.lower() and "reject" in prompt.lower()

        result2 = await graph.ainvoke(Command(resume="approve"), config=config)
        assert await get_pending_interrupt(thread_id) is None
        assert result2["order_confirmed"] is True
        assert "Invoice #" in result2["answer"]

    async def test_reject_cancels_without_ever_finalizing(self, monkeypatch):
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "finance")

        await graph.ainvoke(_initial_state(_INVOICE_QUESTION), config=config)
        assert await get_pending_interrupt_node(thread_id) == "confirm_purchase"

        result2 = await graph.ainvoke(Command(resume="reject"), config=config)
        assert await get_pending_interrupt(thread_id) is None
        assert result2["order_confirmed"] is False
        assert "cancelled" in result2["answer"].lower()
        assert "Invoice #" not in result2["answer"]

    async def test_ambiguous_reply_does_not_finalize_the_order(self, monkeypatch):
        """Neither a clear approve nor reject -- must never be treated as
        placed. An unconfirmed order reported as successful would be worse
        than asking again."""
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "finance")

        await graph.ainvoke(_initial_state(_INVOICE_QUESTION), config=config)

        result2 = await graph.ainvoke(Command(resume="maybe later"), config=config)
        assert result2["order_confirmed"] is False
        assert "Invoice #" not in result2["answer"]
