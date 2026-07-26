"""
Correctness tests for finance_agent.py's deterministic calculation tools —
these are pure Python arithmetic over real company.db rows, no LLM involved.
Verifies the actual numbers, not just routing shape (that's covered in
test_finance_fast_dispatch.py).
"""
import agents.finance_agent as fa


class TestSellingPrice:
    def test_gst_included_in_selling_price(self):
        r = fa.calculate_selling_price.invoke({"product_id": "PRD001"})
        assert "PRD001" in r
        assert "GST" in r


class TestMultiProductPrice:
    def test_grand_total_is_sum_of_items(self):
        r = fa.calculate_multi_product_price.invoke({"product_ids": ["PRD001", "PRD005"]})
        assert "Multi-Product Order Summary" in r
        assert "Grand Total" in r
        assert "Average Price" in r

    def test_missing_product_reported_not_crashed(self):
        r = fa.calculate_multi_product_price.invoke({"product_ids": ["PRD001", "PRD999"]})
        assert "PRD999" in r
        assert "Not found" in r or "not found" in r


class TestMultiProductBulkQuote:
    def test_math_is_correct(self):
        r = fa.calculate_multi_product_bulk_quote.invoke({
            "product_ids": ["PRD003", "PRD004"],
            "quantities": [5, 3],
        })
        assert "5" in r
        assert "3" in r
        assert "Grand Total" in r
        assert "Bulk" in r

    def test_quantities_align_with_products_not_swapped(self):
        """Regression: this tool exists specifically because quantities used
        to be silently discarded and every product defaulted to qty=1."""
        r = fa.calculate_multi_product_bulk_quote.invoke({
            "product_ids": ["PRD005"],
            "quantities": [500],
        })
        assert "500" in r
        assert "Cotton Shirt" in r


class TestMultiProductLoyaltyPrice:
    def test_applies_tier_discount_per_product(self):
        r = fa.calculate_multi_product_loyalty_price.invoke({
            "customer_id": "CUST003",  # Platinum, 12%
            "product_ids": ["PRD001", "PRD003"],
        })
        assert "Platinum" in r
        assert "12%" in r
        assert "Grand Total" in r


class TestCompareProducts:
    def test_price_comparison_sorts_ascending(self):
        r = fa.compare_products.invoke({
            "product_ids": ["PRD001", "PRD005"],
            "comparison_type": "price",
        })
        assert "Cheapest" in r
        assert "Most Expensive" in r

    def test_margin_comparison_uses_margin_labels(self):
        r = fa.compare_products.invoke({
            "product_ids": ["PRD001", "PRD003"],
            "comparison_type": "margin",
        })
        assert "Margin" in r
        assert "Cheapest" not in r  # regression: used to always say "Cheapest"


class TestCompareCustomerRisk:
    def test_ranks_by_days_inactive(self):
        r = fa.compare_customer_risk.invoke({
            "customer_ids": ["CUST001", "CUST009"],
        })
        assert "Most At Risk" in r
        assert "LEAST AT RISK" in r


class TestExplainGstImpact:
    def test_single_product_explanation(self):
        r = fa.explain_gst_impact.invoke({"product_id": "PRD001"})
        assert "GST" in r
        assert "does NOT reduce" in r or "does not reduce" in r.lower()

    def test_multi_product_grouped_by_rate(self):
        """Regression: this tool didn't exist — a multi-product GST question
        used to be answered with a flat price table instead of an
        explanation grouped by rate."""
        r = fa.explain_multi_product_gst_impact.invoke({
            "product_ids": ["PRD001", "PRD005", "PRD009"],
        })
        assert "GST Impact Across" in r
        assert "%" in r


class TestNormalizeIdIntegration:
    def test_get_product_resolves_shorthand(self):
        prod = fa._get_product("prd1")
        assert prod is not None
        assert prod["product_id"] == "PRD001"

    def test_get_customer_resolves_shorthand(self):
        cust = fa._get_customer("cust9")
        assert cust is not None
        assert cust["customer_id"] == "CUST009"


class TestEveryToolHasCardAndBulletSummary:
    """Production requirement: every tool's answer must include BOTH a
    <CHART_DATA> card (for the UI's visual widget) AND a plain-text
    '**Summary:**' bullet recap (for users reading raw text) -- not just one
    or the other. Exercises all 19 tools with a representative call each."""

    def _assert_card_and_summary(self, r: str):
        assert "<CHART_DATA>" in r, "missing card"
        assert "</CHART_DATA>" in r, "unterminated card"
        assert "**Summary:**" in r, "missing bullet summary"

    def test_calculate_selling_price(self):
        self._assert_card_and_summary(fa.calculate_selling_price.invoke({"product_id": "PRD001"}))

    def test_calculate_bulk_quote(self):
        self._assert_card_and_summary(
            fa.calculate_bulk_quote.invoke({"product_id": "PRD001", "quantity": 20})
        )

    def test_calculate_loyalty_price_no_product(self):
        self._assert_card_and_summary(fa.calculate_loyalty_price.invoke({"customer_id": "CUST001"}))

    def test_calculate_loyalty_price_with_product(self):
        self._assert_card_and_summary(
            fa.calculate_loyalty_price.invoke({"customer_id": "CUST001", "product_id": "PRD001"})
        )

    def test_calculate_profit_margin(self):
        self._assert_card_and_summary(fa.calculate_profit_margin.invoke({"product_id": "PRD001"}))

    def test_generate_invoice(self):
        self._assert_card_and_summary(fa.generate_invoice.invoke({
            "customer_id": "CUST001", "product_id": "PRD001", "quantity": 2,
        }))

    def test_forecast_revenue(self):
        self._assert_card_and_summary(fa.forecast_revenue.invoke({"months_ahead": 3}))

    def test_predict_customer_risk(self):
        self._assert_card_and_summary(fa.predict_customer_risk.invoke({"customer_id": "CUST001"}))

    def test_predict_demand(self):
        self._assert_card_and_summary(fa.predict_demand.invoke({"product_id": "PRD001"}))

    def test_customer_lifetime_value(self):
        self._assert_card_and_summary(fa.customer_lifetime_value.invoke({"customer_id": "CUST001"}))

    def test_category_performance(self):
        self._assert_card_and_summary(fa.category_performance.invoke({"category": "Electronics"}))

    def test_monthly_trend_analysis(self):
        self._assert_card_and_summary(fa.monthly_trend_analysis.invoke({}))

    def test_compare_products(self):
        self._assert_card_and_summary(
            fa.compare_products.invoke({"product_ids": ["PRD001", "PRD004"]})
        )

    def test_compare_customer_risk(self):
        self._assert_card_and_summary(
            fa.compare_customer_risk.invoke({"customer_ids": ["CUST001", "CUST003"]})
        )

    def test_explain_gst_impact(self):
        self._assert_card_and_summary(fa.explain_gst_impact.invoke({"product_id": "PRD001"}))

    def test_explain_multi_product_gst_impact(self):
        self._assert_card_and_summary(
            fa.explain_multi_product_gst_impact.invoke({"product_ids": ["PRD001", "PRD005"]})
        )

    def test_compare_gst_by_category(self):
        self._assert_card_and_summary(
            fa.compare_gst_by_category.invoke({"category1": "Clothing", "category2": "Electronics"})
        )

    def test_calculate_multi_product_price(self):
        self._assert_card_and_summary(
            fa.calculate_multi_product_price.invoke({"product_ids": ["PRD001", "PRD004"]})
        )

    def test_calculate_multi_product_loyalty_price(self):
        self._assert_card_and_summary(fa.calculate_multi_product_loyalty_price.invoke({
            "customer_id": "CUST003", "product_ids": ["PRD001", "PRD004"],
        }))

    def test_calculate_multi_product_bulk_quote(self):
        self._assert_card_and_summary(fa.calculate_multi_product_bulk_quote.invoke({
            "product_ids": ["PRD001", "PRD004"], "quantities": [5, 3],
        }))
