"""
Regression tests for the product/customer id resolvers in agents/finance_agent.py.

Every case here traces back to a REAL bug found and fixed during manual testing:
each test's docstring names the bug it guards against. These are pure functions
(regex + string logic, no LLM/DB calls), so the whole file runs in well under a
second — this is the cheapest, highest-value layer to keep covered.
"""
import agents.finance_agent as fa


# ── _resolve_product_id ────────────────────────────────────────────────────────

class TestResolveProductId:
    def test_strict_id(self):
        assert fa._resolve_product_id("price of PRD001") == "PRD001"

    def test_shorthand_digit(self):
        """Bug: 'prod1' resolved to nothing, silently fell back to scanning
        the entire conversation history and grabbing an unrelated product."""
        assert fa._resolve_product_id("product cost of prod1 which i buy 10 units") == "PRD001"

    def test_shorthand_digit_prd_form(self):
        assert fa._resolve_product_id("prd 12 price") == "PRD012"

    def test_product_name(self):
        assert fa._resolve_product_id("price of Water Purifier") == "PRD012"

    def test_recency_last_match_wins(self):
        """Bug: a short follow-up used to resolve against the FIRST product
        ever mentioned in a session instead of the most recently discussed one."""
        text = "we discussed PRD005 earlier, then moved on to PRD003"
        assert fa._resolve_product_id(text) == "PRD003"

    def test_word_boundary_information_not_yoga_mat(self):
        """Bug: 'Customer Information for CUST003' silently matched the bare
        product-name key 'mat' (Yoga Mat) via substring search inside
        'inforMATion', pulling an unrelated product into the answer."""
        text = "Customer Information for CUST003, discount on Smartphone"
        assert fa._resolve_product_id(text) == "PRD003"  # Smartphone, not Yoga Mat

    def test_word_boundary_combat_not_cricket_bat(self):
        text = "this is a combat scenario, price of Smartphone"
        assert fa._resolve_product_id(text) == "PRD003"  # not Cricket Bat (PRD019)

    def test_no_match_returns_none(self):
        assert fa._resolve_product_id("hello how are you") is None


# ── _resolve_multi_product_ids ─────────────────────────────────────────────────

class TestResolveMultiProductIds:
    def test_shorthand_list_with_prefix(self):
        ids = fa._resolve_multi_product_ids("i want to buy prod1, 4 ,6,8 how much")
        assert ids == ["PRD001", "PRD004", "PRD006", "PRD008"]

    def test_shorthand_list_with_connector_word(self):
        """Bug: 'products OF 12,3,1,4,5,7' didn't match because the shorthand
        regex required the number list immediately after the prod/prd prefix,
        with no tolerance for a connector word like 'of' in between."""
        ids = fa._resolve_multi_product_ids("find the cost of products of 12,3,1,4,5,7")
        assert ids == ["PRD012", "PRD003", "PRD001", "PRD004", "PRD005", "PRD007"]

    def test_no_digit_splitting_on_multi_digit_tokens(self):
        """Bug: '12,3' used to get misread as separate digits 1,2,3."""
        ids = fa._resolve_multi_product_ids("prod 12,3")
        assert "PRD012" in ids
        assert "PRD002" not in ids  # would appear if "12" were split into "1","2"

    def test_separate_prefixed_mentions_not_a_list(self):
        """Bug: 'prd1 or prd5' is two SEPARATE prefixed mentions, not one
        comma/and-joined list — only scanning the first match used to drop
        every id after the first 'or'-separated one."""
        ids = fa._resolve_multi_product_ids("which 1 cheper prd1 or prd5")
        assert set(ids) == {"PRD001", "PRD005"}

    def test_overlap_safe_water_purifier(self):
        """Bug: 'water purifier' contains 'purifier' as a literal substring,
        and the two map to DIFFERENT products (PRD012 vs PRD009) — a single
        mention used to resolve as TWO products."""
        assert fa._resolve_multi_product_ids("price of Water Purifier") == ["PRD012"]

    def test_overlap_safe_air_purifier_still_works(self):
        assert fa._resolve_multi_product_ids("price of Air Purifier") == ["PRD009"]

    def test_word_boundary_no_false_positive_from_prose(self):
        text = "Customer Information for CUST003"
        assert "PRD017" not in fa._resolve_multi_product_ids(text)  # Yoga Mat

    def test_comparison_typo_and_shorthand(self):
        ids = fa._resolve_multi_product_ids("why is laptop more expensive than cotton shirt")
        assert set(ids) == {"PRD001", "PRD005"}


# ── _normalize_id ───────────────────────────────────────────────────────────────

class TestNormalizeId:
    def test_pads_shorthand_product(self):
        assert fa._normalize_id("prd1", "PRD") == "PRD001"

    def test_pads_shorthand_customer(self):
        assert fa._normalize_id("cust9", "CUST") == "CUST009"

    def test_already_padded_unchanged(self):
        assert fa._normalize_id("PRD012", "PRD") == "PRD012"

    def test_non_numeric_suffix_passthrough(self):
        # No digits to normalize — falls back to uppercased original
        assert fa._normalize_id("abc", "PRD") == "ABC"


# ── _looks_like_bare_number_list / _resolve_single_product ────────────────────

class TestBareNumberListGuard:
    def test_detects_and_joined_numbers(self):
        assert fa._looks_like_bare_number_list("give me margin for 2 and 15 and 19") is True

    def test_single_number_is_not_a_list(self):
        assert fa._looks_like_bare_number_list("PRD001 price") is False

    def test_resolve_single_product_skips_noisy_history_fallback(self):
        """Bug: 'give me margin for 2 and 15 and 19' has no prod/prd prefix,
        so it resolved to nothing in the current message, then silently fell
        back to scanning conversation history and grabbed an unrelated
        product (PRD007) that wasn't even one of the three numbers asked for."""
        q = "give me margin for 2 and 15 and 19"
        history_question = f"[Conversation history]\nUser: price of PRD007\n\n[Current question]\n{q}"
        assert fa._resolve_single_product(q, history_question) is None

    def test_resolve_single_product_uses_current_message_first(self):
        q = "price of prod1"
        assert fa._resolve_single_product(q, q) == "PRD001"

    def test_resolve_single_product_falls_back_to_history_when_safe(self):
        q = "what about its margin?"
        history_question = f"[Conversation history]\nUser: price of PRD003\n\n[Current question]\n{q}"
        assert fa._resolve_single_product(q, history_question) == "PRD003"


# ── Quantity extraction ───────────────────────────────────────────────────────

class TestExtractAllQuantities:
    def test_units_word(self):
        assert fa._extract_all_quantities("bulk order for 50 units") == [50]

    def test_items_word(self):
        """Bug: 'items' was missing from the unit-word regex entirely --
        'bulk order each 20 items' matched ZERO quantities, so both products
        in a multi-product bulk quote silently defaulted to quantity 1
        instead of the 20 actually asked for."""
        assert fa._extract_all_quantities("bulk order each 20 items") == [20]

    def test_pieces_word(self):
        assert fa._extract_all_quantities("30 pieces please") == [30]

    def test_multiple_quantities_in_order(self):
        assert fa._extract_all_quantities("5 units of Smartphone and 3 units of Tablet") == [5, 3]

    def test_no_unit_word_finds_nothing(self):
        assert fa._extract_all_quantities("give me the price of PRD001") == []


class TestExtractQuantitiesForProducts:
    def test_matching_count_pairs_in_order(self):
        assert fa._extract_quantities_for_products(
            "5 units of Smartphone and 3 units of Tablet", ["PRD003", "PRD004"]
        ) == [5, 3]

    def test_single_shared_quantity_applies_to_every_product(self):
        """The 'each 20 items' case: one quantity mentioned, two products --
        must distribute that same quantity to both, not default to 1."""
        assert fa._extract_quantities_for_products(
            "customer1 wants PRD002 & customer2 wants PRD001, bulk order each 20 items",
            ["PRD002", "PRD001"],
        ) == [20, 20]

    def test_no_quantity_mentioned_defaults_to_one_for_all(self):
        assert fa._extract_quantities_for_products(
            "give me a quote for PRD001 and PRD002", ["PRD001", "PRD002"]
        ) == [1, 1]
