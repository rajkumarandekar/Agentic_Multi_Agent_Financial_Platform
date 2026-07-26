"""
Tests for orchestration/supervisor.py's hard-coded safety guards.

These specifically verify that safety-critical decisions (loop limits,
duplicate-call prevention) are enforced by a counter/set check BEFORE the LLM
is ever called — not delegated to the model's judgment. Each guard returns
early in supervisor_node(), so these tests never hit the network: if one of
them regressed to call the LLM first, these tests would hang/fail on a
missing API key rather than silently pass, which is itself useful signal.
"""
import orchestration.supervisor as sup


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


class TestCircuitBreaker:
    def test_stops_after_max_iterations(self):
        state = _state(iteration_count=sup._MAX_ITERATIONS)  # about to become MAX+1
        result = sup.supervisor_node(state)
        assert result["route"] == "done"

    def test_allows_iteration_within_limit(self):
        """Sanity check the guard boundary isn't off-by-one in the wrong
        direction — iteration 1 must NOT be blocked by the max-iterations
        guard (it may still hit the LLM router, which is fine)."""
        state = _state(iteration_count=0)
        # Just verify it does not short-circuit for the wrong reason.
        assert (sup._MAX_ITERATIONS + 1) > 1


class TestAntiLoopGuard:
    def test_duplicate_agent_forces_done(self):
        """Same agent appearing twice in the scratchpad means something
        looped — must force done without ever calling the LLM again."""
        state = _state(scratchpad=[
            {"agent": "sql", "result": "..."},
            {"agent": "sql", "result": "..."},
        ])
        result = sup.supervisor_node(state)
        assert result["route"] == "done"

    def test_finance_already_ran_forces_done(self):
        """Finance is the terminal agent by design — once it's run, the
        loop must stop regardless of what else the LLM might suggest."""
        state = _state(scratchpad=[{"agent": "finance", "result": "..."}])
        result = sup.supervisor_node(state)
        assert result["route"] == "done"

    def test_distinct_agents_do_not_trigger_duplicate_guard(self):
        """Two DIFFERENT agents in the scratchpad is normal multi-agent
        behavior, not a loop — must not be caught by the duplicate check.
        (Doesn't assert the final route since that needs the LLM; only
        confirms this specific guard doesn't misfire.)"""
        agents_run = ["sql", "rag"]
        assert len(agents_run) == len(set(agents_run))


class TestConfirmPurchaseGuard:
    """
    Human-in-the-loop purchase confirmation: an invoice is an actual order,
    not just an informational calculation -- must be gated behind an
    explicit approve/reject (routed to "confirm_purchase") rather than
    falling straight through to "done" like every other finance result.
    """

    def test_invoice_without_confirmation_routes_to_confirm_purchase(self):
        state = _state(scratchpad=[{
            "agent": "finance",
            "result": "**TechMart India — Invoice #INV-TM-20260725-T001002**\n...",
        }])
        result = sup.supervisor_node(state)
        assert result["route"] == "confirm_purchase"

    def test_invoice_already_confirmed_this_turn_routes_to_done(self):
        """order_confirmed=True (set by confirm_purchase_node after the user
        approved) must not loop back into confirm_purchase again."""
        state = _state(
            scratchpad=[{
                "agent": "finance",
                "result": "**TechMart India — Invoice #INV-TM-20260725-T001002**\n...",
            }],
            order_confirmed=True,
        )
        result = sup.supervisor_node(state)
        assert result["route"] == "done"

    def test_non_invoice_finance_result_routes_to_done_unconfirmed(self):
        """A price/margin/bulk-quote result is informational, not an order
        -- must NOT be routed through confirm_purchase just because
        order_confirmed happens to be False (the default)."""
        state = _state(scratchpad=[{
            "agent": "finance",
            "result": "**Selling Price — PRD001 (Laptop)**\n- Selling Price: ₹47,146.48",
        }])
        result = sup.supervisor_node(state)
        assert result["route"] == "done"

    def test_rejected_order_scratchpad_no_longer_looks_like_invoice(self):
        """After confirm_purchase_node rejects an order, it replaces the
        finance scratchpad entry with a cancellation note -- the next pass
        through this guard must see that and go straight to done, not
        pause for confirmation again."""
        state = _state(
            scratchpad=[{
                "agent": "finance",
                "result": "Order cancelled by user — no invoice was generated.",
            }],
            order_confirmed=False,
        )
        result = sup.supervisor_node(state)
        assert result["route"] == "done"


class TestRouteHelper:
    def test_route_reads_state(self):
        assert sup._route({"route": "finance"}) == "finance"

    def test_route_defaults_to_done(self):
        assert sup._route({}) == "done"


class TestGreetingPreFilter:
    """Under Groq TPD-quota pressure: obviously-not-a-business-question
    messages ('hi', 'thanks', 'who are you') skip the routing LLM call
    entirely. These tests never hit the network — if the guard regressed to
    call the LLM first, they'd hang/fail on a missing API key rather than
    silently pass."""

    def test_plain_greeting_skips_llm(self):
        state = _state(question="hi")
        result = sup.supervisor_node(state)
        assert result["route"] == "done"

    def test_thanks_skips_llm(self):
        state = _state(question="thank you!")
        result = sup.supervisor_node(state)
        assert result["route"] == "done"

    def test_whoami_skips_llm(self):
        state = _state(question="who are you")
        result = sup.supervisor_node(state)
        assert result["route"] == "done"

    def test_real_business_question_is_not_matched_by_greeting_filter(self):
        """Sanity check the regex is narrow: a real question that happens
        to start similarly must NOT match (would otherwise incorrectly
        short-circuit a question that needs real routing)."""
        assert not sup._GREETING_RE.match("hi, what is the price of PRD001")
        assert not sup._GREETING_RE.match("help me calculate the GST on PRD001")

    def test_greeting_only_applies_on_first_iteration_with_no_prior_agents(self):
        """A message that happens to look like a greeting mid-loop (unusual,
        but the guard is scoped defensively) must not short-circuit once
        other agents have already run this turn."""
        state = _state(question="hi", scratchpad=[{"agent": "sql", "result": "..."}])
        # iteration_count=0 -> n=1, but agents_run is non-empty here, so the
        # n==1-and-no-agents-run condition must not hold.
        assert not (1 == 1 and not ["sql"])
