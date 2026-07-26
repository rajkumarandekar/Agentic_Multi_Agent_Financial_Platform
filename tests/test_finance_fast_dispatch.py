"""
Tests for agents/finance_agent.py::_fast_dispatch — the reconstructed
regex/keyword tool dispatcher that replaced the LLM-based dispatch tiers
(_llm_tool_dispatch + the Plan/Validate/Execute routing pipeline) for
production: no LLM call for tool SELECTION at all, so it is free, instant,
and immune to Groq rate limits or malformed tool-call JSON.

These hit the real local company.db (same convention as
test_finance_tools.py / test_finance_resolvers.py) and assert on key
substrings in the real tool output, not exact snapshots.
"""
import agents.finance_agent as fa


class TestSellingPrice:
    def test_single_product(self):
        out = fa._fast_dispatch("price of PRD001")
        assert out is not None
        assert "Laptop" in out
        assert "Selling Price" in out

    def test_multi_product(self):
        out = fa._fast_dispatch("price of PRD001 and PRD004")
        assert "Multi-Product Order Summary" in out
        assert "Laptop" in out and "Tablet" in out

    def test_natural_language_name(self):
        out = fa._fast_dispatch("how much does a tablet cost")
        assert "Tablet" in out


class TestBulkQuote:
    def test_single_product(self):
        out = fa._fast_dispatch("buy 20 units of PRD001")
        assert "Bulk Quote" in out
        assert "20 units" in out

    def test_multi_product_with_own_quantities(self):
        out = fa._fast_dispatch("5 units of Smartphone and 3 units of Tablet")
        assert "Multi-Product Bulk Quote" in out
        assert "Smartphone" in out and "Tablet" in out
        # quantities must align with their own product, not swapped
        assert "5" in out and "3" in out

    def test_not_confused_by_gst_percentage(self):
        """'18%' must not be misread as '18 units' -- this is an analysis
        question, not a bulk order."""
        out = fa._fast_dispatch("what is the impact of GST (18%) on net profit per unit for PRD001")
        assert out is not None
        assert "GST" in out
        assert "Bulk Quote" not in out


class TestLoyaltyPrice:
    def test_discount_only_no_product(self):
        out = fa._fast_dispatch("what discount does CUST001 get")
        assert "Loyalty Discount" in out
        assert "Selling Price" not in out  # no product -> no price breakdown

    def test_with_product(self):
        out = fa._fast_dispatch("loyalty price for CUST001 on PRD001")
        assert "Loyalty Price" in out
        assert "Laptop" in out

    def test_multi_product_loyalty(self):
        out = fa._fast_dispatch("what discount does CUST003 get on PRD001 and PRD004 combined")
        assert "Combined Loyalty Price" in out


class TestInvoice:
    def test_generates_invoice(self):
        out = fa._fast_dispatch("generate an invoice for CUST001 buying 2 units of PRD001")
        assert "Invoice" in out
        assert "Laptop" in out


class TestProfitMargin:
    def test_single_product(self):
        out = fa._fast_dispatch("what is the profit margin on PRD001")
        assert "Profit Margin" in out or "Margin" in out

    def test_multi_product_uses_comparison_tool(self):
        out = fa._fast_dispatch("compare profit margin of PRD001 and PRD004")
        assert "Margin" in out
        assert "Laptop" in out and "Tablet" in out


class TestForecast:
    def test_default_months(self):
        out = fa._fast_dispatch("forecast revenue")
        assert "Revenue Forecast" in out

    def test_explicit_months(self):
        out = fa._fast_dispatch("forecast revenue for the next 5 months")
        assert "Next 5 Month" in out


class TestChurnRisk:
    def test_single_customer(self):
        out = fa._fast_dispatch("is CUST001 at risk of churning")
        assert "Churn Risk" in out

    def test_compare_multiple_customers(self):
        out = fa._fast_dispatch("compare churn risk between CUST001 and CUST003")
        assert "Risk Comparison" in out


class TestDemand:
    def test_predicts_demand(self):
        out = fa._fast_dispatch("predict demand for PRD001")
        assert "Demand Forecast" in out


class TestCustomerLifetimeValue:
    def test_clv(self):
        out = fa._fast_dispatch("what is the lifetime value of CUST001")
        assert "CLV" in out


class TestCategoryAndTrend:
    def test_category_performance(self):
        out = fa._fast_dispatch("how is Electronics doing")
        assert "Category Performance" in out

    def test_monthly_trend(self):
        out = fa._fast_dispatch("show me the overall sales trend")
        assert "Revenue Trend" in out


class TestCompareProducts:
    def test_compares_by_price(self):
        out = fa._fast_dispatch("which is cheaper, PRD001 or PRD004")
        assert "Product Comparison" in out
        assert "CHEAPEST" in out.upper()


class TestGstImpact:
    def test_single_product(self):
        out = fa._fast_dispatch("how does GST affect the profit margin of PRD001")
        assert "GST" in out
        assert "does NOT reduce" in out or "does not reduce" in out.lower()

    def test_all_products(self):
        out = fa._fast_dispatch("how does GST impact the pricing of all products")
        assert "GST Impact Across 20 Product" in out

    def test_category_gst_impact(self):
        out = fa._fast_dispatch("how does GST affect Clothing pricing in general")
        assert "GST Impact Across" in out

    def test_compare_between_categories(self):
        out = fa._fast_dispatch("compare GST between Clothing and Electronics")
        assert "GST Comparison" in out
        assert "Clothing" in out and "Electronics" in out


class TestBulkPolicyRule:
    def test_policy_question_not_a_calculation(self):
        out = fa._fast_dispatch("what is the bulk discount threshold")
        assert "Bulk Discount Policy" in out
        assert "Bulk Quote" not in out


class TestNoMatch:
    def test_casual_greeting_returns_none(self):
        assert fa._fast_dispatch("hello, how are you today") is None

    def test_meta_conversation_question_returns_none(self):
        assert fa._fast_dispatch("summarise everything we discussed") is None


class TestBareNumberListGuard:
    def test_bare_number_list_does_not_leak_history_product(self):
        """'2 and 15 and 19' has no prod/prd prefix -- must NOT fall back to
        scanning conversation history for an unrelated product id. Real bug
        this guards against: this exact phrasing used to silently return a
        product mentioned two turns earlier instead of respecting the
        numbers actually typed."""
        contextual = (
            "[Conversation history]\n"
            "User: price of PRD007\n"
            "Assistant: Kurta Set pricing shown.\n\n"
            "[Current question]\n"
            "cost of 2 and 15 and 19"
        )
        out = fa._fast_dispatch(contextual)
        assert out is None


class TestRecencyAndContextualQuestion:
    def test_uses_current_question_marker_over_history(self):
        """When given the full contextual string _contextual_question builds,
        dispatch must resolve against '[Current question]', not accidentally
        match a product only mentioned in history."""
        contextual = (
            "[Conversation history]\n"
            "User: price of PRD001\n"
            "Assistant: Laptop pricing shown.\n\n"
            "[Current question]\n"
            "what about the profit margin on PRD004"
        )
        out = fa._fast_dispatch(contextual)
        assert "Tablet" in out
        assert "Margin" in out

    def test_follow_up_resolves_from_history_when_current_has_no_entity(self):
        """'what about its margin?' has no product of its own -- must
        resolve the LAST product discussed in history (PRD004, mentioned
        after PRD001), not the first."""
        contextual = (
            "[Conversation history]\n"
            "User: price of PRD001\n"
            "Assistant: Laptop pricing shown.\n"
            "User: now compare that with PRD004\n"
            "Assistant: Tablet pricing shown.\n\n"
            "[Current question]\n"
            "what about its margin?"
        )
        out = fa._fast_dispatch(contextual)
        assert out is not None
        assert "Tablet" in out

    def test_compare_that_with_merges_current_and_history_product(self):
        """'now compare that with PRD004' names only PRD004 explicitly in
        the current message, plus a referential 'that' for the earlier
        product (PRD001). Real bug this guards against, reproduced live:
        the old resolver returned early with just the 1 current-message
        match, the comparison branch saw too few products and declined,
        and the question fell all the way through to the 60s ReAct/Groq
        fallback -- which timed out at 71s for what should be an instant
        regex answer."""
        contextual = (
            "[Conversation history]\n"
            "User: price of PRD001\n"
            "Assistant: Laptop pricing shown.\n\n"
            "[Current question]\n"
            "now compare that with PRD004"
        )
        out = fa._fast_dispatch(contextual)
        assert out is not None
        assert "Laptop" in out and "Tablet" in out
        assert "Product Comparison" in out

    def test_single_customer_churn_question_does_not_merge_with_history(self):
        """Guards the flip side of the merge fix: a plain single-customer
        churn question must NOT pull in an unrelated customer from history
        just because min_count logic exists -- only an explicit comparison
        cue ('compare', 'between', 'vs', 'both') should trigger merging."""
        contextual = (
            "[Conversation history]\n"
            "User: what discount does CUST001 get\n"
            "Assistant: Arjun Mehta is Gold tier.\n\n"
            "[Current question]\n"
            "is CUST003 at risk of churning"
        )
        out = fa._fast_dispatch(contextual)
        assert out is not None
        assert "Churn Risk" in out
        assert "Vikram Singh" in out  # CUST003
        assert "Risk Comparison" not in out
        assert "Arjun Mehta" not in out  # CUST001 must not leak in

    def test_bare_margin_followup_does_not_merge_stale_products_into_a_comparison(self):
        """Real bug, reproduced live: 'Want to see its profit margin too?'
        names no product at all. The old code merged in EVERY product
        mentioned anywhere earlier in the conversation (Laptop, Wireless
        Earbuds) and ran a product COMPARISON nobody asked for, instead of
        the profit margin for the ONE product actually just under
        discussion (PRD001, from the immediately preceding turn)."""
        contextual = (
            "[Conversation history]\n"
            "User: give me a bulk quote for laptop and wireless earbuds, each 20 items\n"
            "Assistant: Multi-Product Bulk Quote shown.\n"
            "User: what is the selling price of PRD001\n"
            "Assistant: Selling Price — Laptop shown.\n\n"
            "[Current question]\n"
            "Want to see its profit margin too?"
        )
        out = fa._fast_dispatch(contextual)
        assert out is not None
        assert "Profit Margin" in out
        assert "Laptop" in out
        assert "Product Comparison" not in out
        assert "Wireless Earbuds" not in out  # the stale product must not leak in

    def test_bare_loyalty_followup_does_not_merge_stale_products_into_a_combined_quote(self):
        """Same bug class as the margin case above, in the loyalty branch:
        'Want a loyalty discount quote for a specific customer instead?'
        names no product -- must resolve to the single most recently
        discussed product (Smartphone), not a combined quote across every
        product mentioned anywhere earlier in the conversation."""
        contextual = (
            "[Conversation history]\n"
            "User: give me a bulk quote for laptop and wireless earbuds, each 20 items\n"
            "Assistant: Multi-Product Bulk Quote shown.\n"
            "User: PRD003, 15 units\n"
            "Assistant: Bulk Quote — Smartphone shown.\n\n"
            "[Current question]\n"
            "Want a loyalty discount quote for CUST001 instead?"
        )
        out = fa._fast_dispatch(contextual)
        assert out is not None
        assert "Smartphone" in out
        assert "Combined Loyalty Price" not in out
        assert "Laptop" not in out  # the stale products must not leak in
        assert "Wireless Earbuds" not in out

    def test_explicit_comparison_language_still_allows_history_merge(self):
        """The gate must not break the legitimate case: explicit comparison
        wording DOES allow merging products from history."""
        contextual = (
            "[Conversation history]\n"
            "User: price of PRD001\n"
            "Assistant: Laptop pricing shown.\n"
            "User: price of PRD004\n"
            "Assistant: Tablet pricing shown.\n\n"
            "[Current question]\n"
            "compare their margins"
        )
        out = fa._fast_dispatch(contextual)
        assert out is not None
        assert "Product Comparison" in out
        assert "Laptop" in out and "Tablet" in out


class TestUncertainEntityAsksInsteadOfGuessing:
    """Real bug, reproduced live: 'i want to buy customer CUST001 a bulk
    order but not sure which product' silently defaulted to two unrelated
    products left over from several turns earlier instead of asking what
    the user just said they didn't know."""

    def test_not_sure_which_product_asks_instead_of_guessing(self):
        out = fa._fast_dispatch(
            "i want to buy customer CUST001 a bulk order but not sure which product"
        )
        assert out is not None
        assert "which product" in out.lower()
        assert "₹" not in out  # must not be a computed quote

    def test_dont_know_which_customer_asks_instead_of_guessing(self):
        out = fa._fast_dispatch("give me a loyalty quote but don't know which customer")
        assert out is not None
        assert "₹" not in out

    def test_normal_question_is_unaffected(self):
        out = fa._fast_dispatch("price of PRD001")
        assert "Laptop" in out
