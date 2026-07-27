"""
Tests for orchestration/semantic_router.py and its wiring into supervisor.py.

Three layers under test:
  1. regex_route()   — exact keyword-combination matches, no model needed.
  2. semantic_route() — embedding cosine similarity against example
     utterances. Loads the real HuggingFace model (already required by
     ingestion/pdf_ingest.py), so these tests are slower and skipped if the
     model can't be loaded (e.g. no network on first run to download it).
  3. supervisor_node() integration — confirms the fast path only ever fires
     on a genuinely fresh question, and that ambiguous/follow-up questions
     still fall through to the (mocked-out) LLM router untouched.
"""
import pytest

import orchestration.semantic_router as router
import orchestration.supervisor as sup


# ── Layer 1: regex ────────────────────────────────────────────────────────────

class TestRegexRoute:
    @pytest.mark.parametrize("question", [
        "show me all customers",
        "list the products you have",
        "how many transactions happened last month",
        "count the orders for CUST001",
        "give me products u have",
        "gimme the customer list",
        "i need the list of transactions",
        "i want the products list",
    ])
    def test_sql_phrasings(self, question):
        assert router.regex_route(question) == "sql"

    @pytest.mark.parametrize("question", [
        "what is the profit margin on PRD001",
        "calculate GST for this order",
        "give me a bulk quote for 50 units",
        "what discount does a gold tier customer get",
    ])
    def test_finance_phrasings(self, question):
        assert router.regex_route(question) == "finance"

    @pytest.mark.parametrize("question", [
        "is CUST009 at risk of churning",
        "flag any suspicious transactions for CUST001",
    ])
    def test_risk_phrasings(self, question):
        assert router.regex_route(question) == "risk"

    @pytest.mark.parametrize("question", [
        "forecast revenue for next quarter",
        "predict revenue for the next 10 days",
    ])
    def test_forecast_phrasings(self, question):
        assert router.regex_route(question) == "forecast"

    @pytest.mark.parametrize("question", [
        "apply for a loan of 50000 rupees",
        "what's the EMI on this loan",
        "check credit eligibility for CUST001",
    ])
    def test_credit_phrasings(self, question):
        assert router.regex_route(question) == "credit"

    @pytest.mark.parametrize("question", [
        "any industry benchmark data on this",
        "what are the market trends in electronics",
        "how do competitors price similar products",
    ])
    def test_research_phrasings(self, question):
        assert router.regex_route(question) == "research"

    @pytest.mark.parametrize("question", [
        "what about that one",
        "summarise what we discussed",
        "tell me more",
        "PRD005",
    ])
    def test_ambiguous_questions_return_none(self, question):
        assert router.regex_route(question) is None

    def test_finance_takes_priority_over_sql_pattern(self):
        """A question matching both patterns (has 'products' AND 'price')
        must resolve to finance -- pricing is never answered by sql."""
        assert router.regex_route("show me the price of all products") == "finance"


# ── Layer 2: semantic (skipped if the embedding model can't load) ────────────

def _model_available() -> bool:
    try:
        router._get_route_vectors()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="embedding model unavailable")
class TestSemanticRoute:
    @pytest.mark.parametrize("question,expected", [
        ("which customers are based in Chennai", "sql"),
        ("give me the full list of orders", "sql"),
        ("what's the loyalty discount for a platinum customer", "finance"),
        ("could you work out the GST breakdown for this purchase", "finance"),
        ("how does this stack up against industry benchmarks", "research"),
    ])
    def test_paraphrases_match_expected_route(self, question, expected):
        route, score = router.semantic_route(question)
        assert route == expected, f"got {route!r} (score={score:.2f}) for {question!r}"

    def test_low_confidence_returns_none(self):
        route, score = router.semantic_route("xyz qwerty asdf 12345")
        assert route is None or score < 0.9  # never a confident false match


class TestMarginGuard:
    """
    Deterministic tests for the runner-up margin check, independent of
    finding a naturally ambiguous real-world phrase: a high top score alone
    isn't enough to trust the fast path -- it must also clearly beat the
    second-best route, or an ambiguous question (plausible under two routes)
    gets confidently misrouted instead of falling through to the LLM.
    """

    def test_clear_winner_is_accepted(self, monkeypatch):
        # finance clearly ahead of both runners-up -> trusted
        scores = {"finance": 0.90, "sql": 0.30, "research": 0.20}
        monkeypatch.setattr(router, "_get_route_vectors", lambda: scores)
        monkeypatch.setattr(router, "_get_embeddings",
                             lambda: type("E", (), {"embed_query": staticmethod(lambda q: q)})())
        monkeypatch.setattr(router, "_cosine_sim", lambda route_score, qv: [route_score])
        route, score = router.semantic_route("anything")
        assert route == "finance"
        assert score == 0.90

    def test_near_tie_is_rejected_even_with_high_top_score(self, monkeypatch):
        # sql and finance are nearly tied (margin 0.05 < _MARGIN_THRESHOLD=0.10)
        # despite BOTH clearing the absolute similarity threshold -- must not
        # confidently guess between two plausible routes.
        scores = {"sql": 0.70, "finance": 0.65, "research": 0.20}
        monkeypatch.setattr(router, "_get_route_vectors", lambda: scores)
        monkeypatch.setattr(router, "_get_embeddings",
                             lambda: type("E", (), {"embed_query": staticmethod(lambda q: q)})())
        monkeypatch.setattr(router, "_cosine_sim", lambda route_score, qv: [route_score])
        route, score = router.semantic_route("anything")
        assert route is None

    def test_clear_margin_but_below_absolute_threshold_is_rejected(self, monkeypatch):
        # Big margin between routes, but the top score itself is too low to
        # trust as a real match to ANY route.
        scores = {"sql": 0.40, "finance": 0.10, "research": 0.05}
        monkeypatch.setattr(router, "_get_route_vectors", lambda: scores)
        monkeypatch.setattr(router, "_get_embeddings",
                             lambda: type("E", (), {"embed_query": staticmethod(lambda q: q)})())
        monkeypatch.setattr(router, "_cosine_sim", lambda route_score, qv: [route_score])
        route, score = router.semantic_route("anything")
        assert route is None


# ── Layer 3: supervisor integration ───────────────────────────────────────────

def _state(**overrides):
    base = {
        "question": "some question",
        "iteration_count": 0,
        "scratchpad": [],
        "source_document": None,
        "messages": [],
    }
    base.update(overrides)
    return base


class TestSupervisorFastPathWiring:
    def test_fast_path_skips_llm_for_clear_sql_question(self, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("LLM router must not be called when fast path is confident")
        monkeypatch.setattr(sup, "_route_question", _boom)
        monkeypatch.setattr(router, "fast_route", lambda q: "sql")
        monkeypatch.setattr(sup, "fast_route", router.fast_route)

        state  = _state(question="show me all customers in Chennai")
        result = sup.supervisor_node(state)
        assert result["route"] == "sql"

    def test_fast_path_disabled_when_pdf_active(self, monkeypatch):
        """PDF active always forces rag -- the fast path must not even run,
        so a regex/semantic 'sql' guess never gets computed then overridden."""
        calls = []
        def _spy(q):
            calls.append(q)
            return "sql"
        monkeypatch.setattr(sup, "fast_route", _spy)

        def _fake_llm_route(**kw):
            return "rag"
        monkeypatch.setattr(sup, "_route_question", lambda **kw: _fake_llm_route(**kw))

        state  = _state(question="show me all customers", source_document="report.pdf")
        result = sup.supervisor_node(state)
        assert calls == []  # fast_route never invoked
        assert result["route"] == "rag"

    def test_fast_path_disabled_mid_loop(self, monkeypatch):
        """A follow-up turn (agents already ran) must never use the fast
        path -- only the LLM router has the context to resolve it."""
        calls = []
        monkeypatch.setattr(sup, "fast_route", lambda q: calls.append(q) or "sql")
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "finance")

        state = _state(
            question="what about those",
            iteration_count=1,
            # Multi-row/multi-column result, not a placeholder -- a bare
            # "..." would accidentally satisfy supervisor.py's "sql already
            # gave a complete single-value answer -> done" override (see
            # that override's own comment) and mask what this test is
            # actually checking (fast path disabled mid-loop).
            scratchpad=[{"agent": "sql", "result": "product_id,category\nPRD001,Electronics\nPRD002,Clothing"}],
        )
        result = sup.supervisor_node(state)
        assert calls == []
        assert result["route"] == "finance"

    def test_llm_fallback_when_fast_path_has_no_opinion(self, monkeypatch):
        monkeypatch.setattr(sup, "fast_route", lambda q: None)
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "research")

        state  = _state(question="something genuinely ambiguous")
        result = sup.supervisor_node(state)
        assert result["route"] == "research"

    def test_fast_path_disabled_for_own_followup_suggestions(self, monkeypatch):
        """Real bug: clicking the suggested follow-up chip "Want an invoice
        for one of these?" sent that exact string back as the question, and
        the finance regex/semantic layer matched on "invoice" -- running
        finance with no product actually specified (silently defaulting to
        PRD001) instead of the LLM resolving "one of these" from history.
        These strings are a fixed, known set (orchestration/followup.py) and
        are ALWAYS context-dependent by construction -- must never fast-path,
        regardless of how confidently regex/semantic would guess."""
        calls = []
        monkeypatch.setattr(sup, "fast_route", lambda q: calls.append(q) or "finance")
        monkeypatch.setattr(sup, "_route_question", lambda **kw: "finance")

        for q in sup.followup.ALL_FOLLOWUPS:
            state  = _state(question=q)
            result = sup.supervisor_node(state)
            assert result["route"] == "finance"  # still routes correctly...
        assert calls == []  # ...but only via the LLM router, never fast_route
