"""
Regression tests for guardrails/output_guard.py -- specifically the
toxicity-check skip logic added to cut Groq calls under TPD-quota pressure.
Finance/SQL answers are deterministic, DB-derived text (not LLM freeform
generation), so a toxicity check on them has ~zero expected value; short
answers (canned greeting replies) aren't worth a Groq call either.
"""
from unittest.mock import MagicMock, patch

import guardrails.output_guard as og


class TestDeterministicSourceSkip:
    def test_finance_chart_data_skips_toxicity_llm_call(self):
        answer = "<CHART_DATA>\n{}\n</CHART_DATA>\n\n**Selling Price**\n\n- Base Cost: 100"
        with patch("guardrails.output_guard.ChatGroq", side_effect=AssertionError("should not be called")):
            result = og.check_output(answer, agents_used=["finance"])
        assert result["passed"] is True
        assert result["checks"][1]["detail"].startswith("skipped")

    def test_finance_and_sql_only_skips_toxicity_llm_call(self):
        answer = "x" * 500  # long enough to bypass the short-answer skip
        with patch("guardrails.output_guard.ChatGroq", side_effect=AssertionError("should not be called")):
            result = og.check_output(answer, agents_used=["sql", "finance"])
        assert result["checks"][1]["detail"].startswith("skipped")

    def test_rag_answer_still_gets_real_check(self):
        """RAG pulls from a user-uploaded PDF -- real freeform content --
        must NOT be skipped just because it's paired with a deterministic
        agent in the same turn."""
        mock_response = MagicMock()
        mock_response.content = "NO: clean"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        answer = "x" * 500
        with patch("guardrails.output_guard.ChatGroq", return_value=mock_llm):
            result = og.check_output(answer, agents_used=["rag"])
        assert mock_llm.invoke.called


class TestShortAnswerSkip:
    def test_short_answer_skips_toxicity_llm_call(self):
        with patch("guardrails.output_guard.ChatGroq", side_effect=AssertionError("should not be called")):
            result = og.check_output("Hi! Happy to help.", agents_used=[])
        assert result["passed"] is True
        assert result["checks"][1]["detail"].startswith("skipped")

    def test_long_chat_answer_still_gets_real_check(self):
        mock_response = MagicMock()
        mock_response.content = "NO: clean"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        long_answer = "This is a long freeform chat answer. " * 15  # > 400 chars
        with patch("guardrails.output_guard.ChatGroq", return_value=mock_llm):
            result = og.check_output(long_answer, agents_used=["chat"])
        assert mock_llm.invoke.called
