"""
End-to-end test for the clarify() human-in-the-loop pause/resume wiring.

Covers the gap that used to exist between orchestration/clarify.py (a real
LangGraph interrupt()) and api/main.py (which never checked for a pause or
resumed one). orchestration/graph.py's get_pending_interrupt() is the piece
that closes it: aget_state() is the only way this LangGraph version exposes
a pending interrupt (ainvoke's return value doesn't surface it -- see that
function's docstring), so both the "did we just pause" and "are we resuming"
checks in api/main.py go through it.

Network calls (Groq) are stubbed at the same boundaries the rest of this
test suite already uses: _route_question is forced to pick "clarify" then
"done", clarify's own LLM call is forced to fail over to its deterministic
fallback question, and the final response is stubbed outright (SQL/RAG/
finance formatting is covered elsewhere -- this test is only about the
pause/resume mechanics).
"""
import uuid

import pytest
from langgraph.types import Command

import orchestration.clarify as clarify_mod
import orchestration.supervisor as sup
from orchestration.graph import graph, get_pending_interrupt


class _ExplodingLLM:
    """Stands in for ChatGroq(...) -- any .invoke() raises, forcing
    clarify_node's except-branch fallback question, so the test never
    depends on a real Groq call to produce a deterministic question."""
    def invoke(self, *a, **kw):
        raise RuntimeError("no network in tests")


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
        "guardrail_results": {},
    }


@pytest.mark.asyncio
class TestClarifyPauseAndResume:
    async def test_pause_then_resume_end_to_end(self, monkeypatch):
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # First call to the router: force "clarify". Second call (after
        # resume): force "done" so the loop terminates without needing a
        # real routing decision on the resumed question.
        route_calls = {"n": 0}
        def _fake_route_question(**kw):
            route_calls["n"] += 1
            return "clarify" if route_calls["n"] == 1 else "done"
        monkeypatch.setattr(sup, "_route_question", _fake_route_question)

        # Force clarify_node's LLM call to fail -> deterministic fallback question.
        monkeypatch.setattr(clarify_mod, "ChatGroq", lambda *a, **kw: _ExplodingLLM())

        # Response node's own formatting is out of scope for this test --
        # stub it so "done" doesn't need a real Groq call either.
        async def _fake_response_run(question, scratchpad, messages):
            return "stub final answer"
        monkeypatch.setattr("agents.response_agent.run", _fake_response_run)

        # ── Turn 1: an ambiguous question, short enough to skip guardrail LLM calls ─
        assert await get_pending_interrupt(thread_id) is None

        result = await graph.ainvoke(_initial_state("what does it say"), config=config)

        # graph.ainvoke() does NOT surface the interrupt in its return value
        # in this LangGraph version -- that's exactly the gap being tested.
        assert "answer" not in result or not result.get("answer")

        question = await get_pending_interrupt(thread_id)
        assert question is not None
        assert "uploaded document" in question or "source" in question.lower() or "?" in question

        # ── Turn 2: resume with the user's answer ────────────────────────────
        result2 = await graph.ainvoke(Command(resume="the PDF"), config=config)

        assert await get_pending_interrupt(thread_id) is None
        assert result2["clarified_source"] == "the PDF"
        assert result2["answer"] == "stub final answer"

    async def test_get_pending_interrupt_is_none_for_unknown_thread(self):
        assert await get_pending_interrupt(str(uuid.uuid4())) is None
