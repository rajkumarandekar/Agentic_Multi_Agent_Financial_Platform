"""
Tests for agents/research_agent.py — Phase 3's CrewAI rewrite.

Avoids real network/Groq calls (same convention as the rest of this test
suite, e.g. test_confirm_purchase.py mocking the one real LLM call in its
path): web_search's own graceful-degradation behavior is tested directly
(pure, no network since TAVILY_API_KEY is unset in the test environment),
and run()'s CrewAI synthesis step is exercised with the Crew mocked out so
the test verifies wiring/error-handling, not live model output.
"""
import pytest

import agents.research_agent as ra


class TestWebSearchFallback:
    def test_unavailable_without_tavily_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        out = ra.web_search.invoke({"query": "average retail conversion rate in India"})
        assert "TAVILY_API_KEY is not set" in out
        assert "average retail conversion rate in India" in out


class TestGroqLLMMessageSanitization:
    def test_strips_extra_fields_before_calling_groq(self, monkeypatch):
        """CrewAI attaches extra per-message fields (e.g. cache_breakpoint)
        that Groq's endpoint rejects with a 400 -- _GroqLLM must strip down
        to plain role/content before calling out."""
        captured = {}

        class _FakeCompletions:
            def create(self, model, messages, temperature):
                captured["messages"] = messages
                class _Choice:
                    class message:
                        content = "ok"
                return type("R", (), {"choices": [_Choice()]})()

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        monkeypatch.setattr(ra, "OpenAI", lambda **kw: _FakeClient())

        llm = ra._GroqLLM()
        result = llm.call(messages=[
            {"role": "system", "content": "sys", "cache_breakpoint": True},
            {"role": "user", "content": "hi", "extra_field": 123},
        ])
        assert result == "ok"
        assert captured["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    def test_accepts_bare_string_messages(self, monkeypatch):
        captured = {}

        class _FakeCompletions:
            def create(self, model, messages, temperature):
                captured["messages"] = messages
                class _Choice:
                    class message:
                        content = "ok"
                return type("R", (), {"choices": [_Choice()]})()

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        monkeypatch.setattr(ra, "OpenAI", lambda **kw: _FakeClient())

        llm = ra._GroqLLM()
        result = llm.call(messages="just a plain string")
        assert result == "ok"
        assert captured["messages"] == [{"role": "user", "content": "just a plain string"}]


@pytest.mark.asyncio
class TestRun:
    async def test_falls_back_gracefully_when_crew_fails(self, monkeypatch):
        """If the CrewAI synthesis step itself blows up (bad key, network
        error, whatever), run() must degrade to the friendly fallback
        message, not propagate an exception up through the supervisor."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        def _boom(*a, **kw):
            raise RuntimeError("crew exploded")
        monkeypatch.setattr(ra, "_GroqLLM", _boom)

        out = await ra.run("what's the average online retail margin in India")
        assert out == "External research unavailable. Answer based on internal data only."
