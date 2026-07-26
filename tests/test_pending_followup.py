"""
Tests for pending_followup tracking (orchestration/state.py,
orchestration/graph.py's response_node, orchestration/supervisor.py's
_build_history).

Real bug this closes: a suggested follow-up question (e.g. "Want to see its
profit margin too?") only ever existed in the API response's separate
`followup` field for the UI chip -- it was never written into state["messages"],
so when the user replied with a bare "ok"/"yes"/"show me", the router's
conversation history had literally no record of what was being confirmed.
pending_followup carries that text forward as its own persisted state field
(NOT folded into the message content, which would risk truncation for long
answers) so _build_history can surface it explicitly every turn until the
next answer replaces it.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import orchestration.supervisor as sup
from orchestration.graph import response_node


def _state(**overrides):
    base = {
        "question": "some question",
        "messages": [],
        "scratchpad": [],
        "agents_used": [],
    }
    base.update(overrides)
    return base


class TestBuildHistorySurfacesPendingFollowup:
    def test_no_pending_followup_omits_note(self):
        history = sup._build_history(_state(pending_followup=None))
        assert "Note:" not in history

    def test_pending_followup_appears_as_explicit_note(self):
        state = _state(pending_followup="Want to see its profit margin too?")
        history = sup._build_history(state)
        assert "Want to see its profit margin too?" in history
        assert "RULE 12" in history

    def test_pending_followup_survives_even_with_no_prior_messages(self):
        """The note must appear even on what would otherwise be a
        'No prior conversation.' turn -- otherwise a same-turn resume-style
        case loses it."""
        state = _state(messages=[], pending_followup="Want an invoice for this order?")
        history = sup._build_history(state)
        assert "Want an invoice for this order?" in history


class TestResponseNodeSetsPendingFollowup:
    def test_response_node_writes_suggested_followup_to_state(self):
        # A finance-shaped answer whose title matches a real followup.py rule.
        finance_answer = (
            '<CHART_DATA>{"type": "calculation", "title": "Selling Price — Laptop"}</CHART_DATA>\n\n'
            "**Selling Price — PRD001**\n- Price: ₹47,146.48"
        )
        state = _state(scratchpad=[{"agent": "finance", "result": finance_answer}],
                        agents_used=["finance"])

        with patch("agents.response_agent.run", new=AsyncMock(return_value=finance_answer)):
            result = asyncio.run(response_node(state))

        assert result["pending_followup"] == "Want to see its profit margin too?"

    def test_response_node_sets_none_when_no_followup_applies(self):
        with patch("agents.response_agent.run", new=AsyncMock(return_value="just a plain chat reply")):
            result = asyncio.run(response_node(_state()))

        assert result["pending_followup"] is None
