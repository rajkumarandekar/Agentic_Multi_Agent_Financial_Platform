"""
Tests for agents/response_agent.py::_format_single_column_sql_result and its
wiring into run()'s SQL branch. Real bug this guards against: asking "what
data do you have" produced a correct, real 5-row single-column SQL result
(table names from sqlite_master), but the LLM formatting step invented an
entirely fictional multi-column table (fake product/customer names and
amounts) instead of listing the real table names. Single-column results are
now formatted deterministically -- no LLM call, nothing to hallucinate.
"""
from unittest.mock import MagicMock, patch

import pytest

import agents.response_agent as ra


class TestFormatSingleColumnSqlResult:
    def test_formats_table_name_list(self):
        sql_result = "name\nproducts\ncustomers\ntransactions\nmonthly_sales\ncompany_rates"
        out = ra._format_single_column_sql_result(sql_result)
        assert out is not None
        assert "products" in out
        assert "customers" in out
        assert "(5)" in out
        # None of the real bug's fabricated content should ever appear --
        # this function only ever echoes the actual input values.
        assert "Product A" not in out
        assert "John Smith" not in out

    def test_multi_column_result_returns_none(self):
        sql_result = "product_id, product_name\nPRD001, Laptop\nPRD002, Tablet"
        assert ra._format_single_column_sql_result(sql_result) is None

    def test_header_only_no_rows_returns_none(self):
        assert ra._format_single_column_sql_result("name") is None

    def test_empty_string_returns_none(self):
        assert ra._format_single_column_sql_result("") is None


class TestRunUsesDeterministicFormatterForSingleColumn:
    @pytest.mark.asyncio
    async def test_single_column_sql_skips_llm_call(self):
        scratchpad = [{
            "agent": "sql",
            "result": "name\nproducts\ncustomers\ntransactions\nmonthly_sales\ncompany_rates",
        }]
        # No ChatGroq mock provided -- if run() fell through to the LLM
        # formatter, this would attempt a real network call.
        answer = await ra.run("what data do you have", scratchpad, messages=[])
        assert "products" in answer
        assert "customers" in answer

    @pytest.mark.asyncio
    async def test_multi_column_sql_still_uses_llm_formatter(self):
        scratchpad = [{
            "agent": "sql",
            "result": "product_id, product_name\nPRD001, Laptop\nPRD002, Tablet",
        }]
        mock_response = MagicMock()
        mock_response.content = "**2 products found.**\n\n| product_id | product_name |\n|---|---|\n| PRD001 | Laptop |\n| PRD002 | Tablet |"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        with patch("agents.response_agent.ChatGroq", return_value=mock_llm):
            answer = await ra.run("list products", scratchpad, messages=[])
        assert mock_llm.invoke.called
        assert "Laptop" in answer
