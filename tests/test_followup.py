"""
Tests for orchestration/followup.py -- deterministic, zero-LLM-call
follow-up suggestion. Exercises real finance tool output (not synthetic
JSON) so a card-shape change in finance_agent.py that breaks this matching
shows up here.
"""
import agents.finance_agent as fa
from orchestration.followup import suggest_followup, pending_followup_for


class TestRealToolOutputMatching:
    def test_selling_price_suggests_margin(self):
        out = fa.calculate_selling_price.invoke({"product_id": "PRD001"})
        assert suggest_followup(out) == "Want to see its profit margin too?"

    def test_bulk_quote_suggests_loyalty(self):
        out = fa.calculate_bulk_quote.invoke({"product_id": "PRD001", "quantity": 5})
        assert "loyalty" in suggest_followup(out).lower()

    def test_multi_product_bulk_quote_not_confused_with_bulk_quote(self):
        """Regression: 'Bulk Quote' is a substring of 'Multi-Product Bulk
        Quote' -- the multi-product title must match its OWN rule (None),
        not fall through to the single-product bulk-quote rule."""
        out = fa.calculate_multi_product_bulk_quote.invoke({
            "product_ids": ["PRD001", "PRD004"], "quantities": [2, 3],
        })
        assert suggest_followup(out) is None

    def test_invoice_has_no_followup(self):
        out = fa.generate_invoice.invoke({
            "customer_id": "CUST001", "product_id": "PRD001", "quantity": 2,
        })
        assert suggest_followup(out) is None

    def test_profit_margin_suggests_price_breakdown(self):
        out = fa.calculate_profit_margin.invoke({"product_id": "PRD001"})
        assert suggest_followup(out) is not None

    def test_churn_risk_suggests_comparison(self):
        out = fa.predict_customer_risk.invoke({"customer_id": "CUST001"})
        assert "compare" in suggest_followup(out).lower()

    def test_compare_customer_risk_suggests_lifetime_value(self):
        out = fa.compare_customer_risk.invoke({"customer_ids": ["CUST001", "CUST003"]})
        assert "lifetime value" in suggest_followup(out).lower()

    def test_forecast_revenue_no_title_field_still_matches_via_heading(self):
        """forecast_revenue's card has no 'title' key -- must fall back to
        matching the plain-text markdown heading instead of silently
        returning None for every forecast."""
        out = fa.forecast_revenue.invoke({"months_ahead": 3})
        assert suggest_followup(out) is not None

    def test_category_performance_no_title_field_still_matches_via_heading(self):
        out = fa.category_performance.invoke({"category": "Electronics"})
        assert "sales trend" in suggest_followup(out).lower()

    def test_monthly_trend_no_title_field_still_matches_via_heading(self):
        out = fa.monthly_trend_analysis.invoke({})
        assert suggest_followup(out) is not None

    def test_gst_impact_has_no_followup(self):
        out = fa.explain_gst_impact.invoke({"product_id": "PRD001"})
        assert suggest_followup(out) is None


class TestNonFinanceAnswers:
    def test_plain_text_answer_returns_none(self):
        assert suggest_followup("Hi! Happy to help.") is None

    def test_empty_answer_returns_none(self):
        assert suggest_followup("") is None
        assert suggest_followup(None) is None

    def test_malformed_chart_data_does_not_crash(self):
        assert suggest_followup("<CHART_DATA>\nnot valid json\n</CHART_DATA>") is None


class TestPendingFollowupFor:
    """
    pending_followup_for() is what response_node actually calls -- it tries
    the deterministic suggest_followup() table first, then falls back to a
    free-form trailing question. Real bug this closes: the chat LLM asking
    its own clarifying question ("which product and quantity would you like
    to generate an invoice for?") used to vanish completely after that turn
    -- it wasn't one of the fixed templates, so pending_followup got reset
    to None, and a later "yes resume it" reply had nothing to resolve against.
    """

    def test_deterministic_table_takes_priority(self):
        out = fa.calculate_selling_price.invoke({"product_id": "PRD001"})
        assert pending_followup_for(out) == "Want to see its profit margin too?"

    def test_falls_back_to_free_form_trailing_question(self):
        answer = (
            "You're looking to create an invoice. Before I assist you with "
            "that, just to confirm - which product and quantity would you "
            "like to generate an invoice for?"
        )
        result = pending_followup_for(answer)
        # Captures everything after the last sentence terminator (".") --
        # the "-" before "which product" isn't a terminator, so the whole
        # confirm-clause is included. What matters is the actual question
        # survives intact, not the exact split point.
        assert result.endswith("which product and quantity would you like to generate an invoice for?")
        assert "You're looking to create an invoice" not in result

    def test_plain_statement_with_no_question_returns_none(self):
        assert pending_followup_for("Here are the 20 products available.") is None

    def test_multi_sentence_answer_only_captures_last_sentence(self):
        answer = "Here's the price. It's a good deal. Want to see the full breakdown too?"
        assert pending_followup_for(answer) == "Want to see the full breakdown too?"

    def test_overlong_final_sentence_is_not_captured(self):
        """A guard against grabbing an entire multi-paragraph answer that
        happens to have no sentence terminators before its final '?'."""
        long_tail = "x" * 250 + "?"
        assert pending_followup_for(long_tail) is None

    def test_empty_and_none_return_none(self):
        assert pending_followup_for("") is None
        assert pending_followup_for(None) is None
