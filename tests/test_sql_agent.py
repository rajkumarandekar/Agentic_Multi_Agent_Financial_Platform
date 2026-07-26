"""
Regression tests for agents/sql_agent.py's entity extraction — pure regex
logic, no LLM/DB calls except test_check_entity_exists which hits the real
company.db (fast, local).
"""
from unittest.mock import MagicMock, patch

from langgraph.errors import GraphRecursionError

import agents.sql_agent as sa


class TestNormalizeId:
    def test_pads_shorthand_customer(self):
        assert sa._normalize_id("cust9", "CUST") == "CUST009"

    def test_pads_shorthand_product(self):
        assert sa._normalize_id("prd1", "PRD") == "PRD001"

    def test_already_padded_unchanged(self):
        assert sa._normalize_id("CUST003", "CUST") == "CUST003"


class TestExtractEntity:
    def test_shorthand_customer_id_normalized(self):
        """Regression: 'cust9' used to fail the pre-flight DB existence
        check as 'not found' — purely because the extracted id string wasn't
        zero-padded to match the DB's CUST009 format — before the question
        ever reached the LLM."""
        entity = sa._extract_entity("is cust9 gona leave us")
        assert entity["customer_id"] == "CUST009"

    def test_shorthand_product_id_normalized(self):
        entity = sa._extract_entity("show me prd1 details")
        assert entity["product_id"] == "PRD001"

    def test_no_entity_returns_none(self):
        entity = sa._extract_entity("show me all customers")
        assert entity["customer_id"] is None
        assert entity["product_id"] is None


class TestCheckEntityExists:
    def test_real_customer_exists(self):
        exists, msg = sa._check_entity_exists({"customer_id": "CUST001", "product_id": None})
        assert exists is True

    def test_fake_customer_reported_not_found(self):
        exists, msg = sa._check_entity_exists({"customer_id": "CUST999", "product_id": None})
        assert exists is False
        assert "CUST999" in msg

    def test_shorthand_customer_resolves_to_real_row(self):
        """End-to-end of the normalize fix: 'cust9' -> 'CUST009' must pass
        the existence check, not be reported as not found."""
        entity = sa._extract_entity("cust9 transactions")
        exists, msg = sa._check_entity_exists(entity)
        assert exists is True


class TestRunExceptionHandling:
    """Regression: a broad question ('what data do you have') made the
    ReAct agent query every table in turn until it hit the recursion-limit
    budget -- the old handler caught that under a bare `except Exception`
    and claimed 'No matching records found in the database', which is
    FALSE (real rows almost certainly came back from the first query; they
    just never got returned). GraphRecursionError must now be handled
    separately with an honest message, not folded into the generic
    API-failure fallback."""

    def test_recursion_error_gives_honest_message_not_false_no_results_claim(self):
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = GraphRecursionError("recursion limit reached")
        with patch("agents.sql_agent.create_react_agent", return_value=mock_agent):
            result = sa.run("what data do you have", entity_question="what data do you have")
        assert "No matching records found" not in result
        assert "too broad" in result.lower()

    def test_generic_exception_still_gives_friendly_fallback(self):
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("connection error")
        with patch("agents.sql_agent.create_react_agent", return_value=mock_agent):
            result = sa.run("show me all customers", entity_question="show me all customers")
        assert result  # some friendly message, not a raised exception

    def test_generic_exception_with_known_customer_still_names_them(self):
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("connection error")
        with patch("agents.sql_agent.create_react_agent", return_value=mock_agent):
            result = sa.run("CUST001 transactions", entity_question="CUST001 transactions")
        assert "CUST001" in result
