"""
Tests for agents/response_agent.py's greeting/meta-chat handling.

"thanks" and "bye" stay zero-LLM-call canned replies -- added under Groq
TPD-quota pressure so an obvious reply doesn't cost an LLM call when it's
never in doubt. "hi" and "who are you" are answered dynamically instead (a
short Groq call, varied phrasing each time) -- those tests mock ChatGroq so
no real network call happens, and separately verify the fallback path when
that call fails.
"""
import pytest

import agents.response_agent as ra


class TestClassifyGreeting:
    """Pure regex classification -- no LLM call, no side effects."""

    def test_hi_matches(self):
        assert ra._classify_greeting("hi") == "hi"
        assert ra._classify_greeting("Hello!") == "hi"
        assert ra._classify_greeting("heyy") == "hi"

    def test_thanks_matches(self):
        assert ra._classify_greeting("thanks") == "thanks"
        assert ra._classify_greeting("thank you!") == "thanks"

    def test_bye_matches(self):
        assert ra._classify_greeting("bye") == "bye"
        assert ra._classify_greeting("goodbye") == "bye"

    def test_whoami_matches(self):
        assert ra._classify_greeting("who are you") == "whoami"
        assert ra._classify_greeting("what can you do") == "whoami"
        assert ra._classify_greeting("help") == "whoami"

    def test_real_question_does_not_match(self):
        assert ra._classify_greeting("what is the price of PRD001") is None
        assert ra._classify_greeting("hi, price of PRD001 please") is None


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return type("R", (), {"content": self._content})()


class _ExplodingLLM:
    def invoke(self, messages):
        raise RuntimeError("no network in tests")


class TestDynamicReply:
    """'hi' and 'who are you' now call Groq for a fresh reply each time --
    mock ChatGroq so these tests never touch the network."""

    def test_hi_uses_llm_reply_when_call_succeeds(self, monkeypatch):
        monkeypatch.setattr(ra, "ChatGroq", lambda *a, **kw: _FakeLLM("Hey! Ask me about pricing anytime."))
        reply = ra._greeting_reply("hi")
        assert reply == "Hey! Ask me about pricing anytime."

    def test_whoami_uses_llm_reply_when_call_succeeds(self, monkeypatch):
        monkeypatch.setattr(ra, "ChatGroq", lambda *a, **kw: _FakeLLM("I'm backed by SQL, Finance, RAG, and Research agents."))
        reply = ra._greeting_reply("who are you")
        assert "SQL" in reply

    def test_hi_falls_back_when_llm_call_fails(self, monkeypatch):
        monkeypatch.setattr(ra, "ChatGroq", lambda *a, **kw: _ExplodingLLM())
        reply = ra._greeting_reply("hi")
        assert reply in ra._HI_FALLBACK_REPLIES

    def test_whoami_falls_back_when_llm_call_fails(self, monkeypatch):
        monkeypatch.setattr(ra, "ChatGroq", lambda *a, **kw: _ExplodingLLM())
        reply = ra._greeting_reply("who are you")
        assert reply == ra._WHOAMI_FALLBACK_REPLY


class TestThanksAndByeStayCanned:
    """No LLM call for these -- a fixed reply is never wrong."""

    def test_thanks_returns_canned_reply_no_llm_call(self, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("thanks must not call the LLM")
        monkeypatch.setattr(ra, "ChatGroq", _boom)
        reply = ra._greeting_reply("thanks!")
        assert reply in ra._CANNED_THANKS_REPLIES

    def test_bye_returns_canned_reply_no_llm_call(self, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("bye must not call the LLM")
        monkeypatch.setattr(ra, "ChatGroq", _boom)
        reply = ra._greeting_reply("goodbye")
        assert reply in ra._CANNED_BYE_REPLIES


class TestRunUsesGreetingReply:
    @pytest.mark.asyncio
    async def test_hi_returns_dynamic_reply_via_run(self, monkeypatch):
        monkeypatch.setattr(ra, "ChatGroq", lambda *a, **kw: _FakeLLM("Hiya!"))
        answer = await ra.run("hi", scratchpad=[], messages=[])
        assert answer == "Hiya!"

    @pytest.mark.asyncio
    async def test_thanks_returns_canned_reply_via_run_no_llm_call(self, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("thanks must not call the LLM")
        monkeypatch.setattr(ra, "ChatGroq", _boom)
        answer = await ra.run("thanks!", scratchpad=[], messages=[])
        assert answer in ra._CANNED_THANKS_REPLIES
