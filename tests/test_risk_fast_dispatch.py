"""
Tests for agents/risk_agent.py::_fast_dispatch — churn risk moved out of
finance_agent.py (Phase 2 of the multi-agent expansion) into a dedicated
Risk agent, plus its new fraud/anomaly detection tool.

Same conventions as test_finance_fast_dispatch.py: real local company.db,
assertions on key substrings rather than exact snapshots.
"""
import agents.risk_agent as ra


class TestChurnRisk:
    def test_single_customer(self):
        out = ra._fast_dispatch("is CUST001 at risk of churning")
        assert out is not None
        assert "Churn Risk" in out

    def test_compare_multiple_customers(self):
        out = ra._fast_dispatch("compare churn risk between CUST001 and CUST003")
        assert out is not None
        assert "Risk Comparison" in out

    def test_no_customer_id_returns_none(self):
        assert ra._fast_dispatch("is this customer at risk of churning") is None

    def test_single_customer_churn_question_does_not_merge_with_history(self):
        """A plain single-customer churn question must NOT pull in an
        unrelated customer from history just because both ids are present
        in the contextual string -- only an explicit comparison cue
        ('compare', 'between', 'vs', 'both') should trigger merging."""
        contextual = (
            "[Conversation history]\n"
            "User: what discount does CUST001 get\n"
            "Assistant: Arjun Mehta is Gold tier.\n\n"
            "[Current question]\n"
            "is CUST003 at risk of churning"
        )
        out = ra._fast_dispatch(contextual)
        assert out is not None
        assert "Churn Risk" in out
        assert "Vikram Singh" in out  # CUST003
        assert "Risk Comparison" not in out


class TestFraudDetection:
    def test_flags_or_clears_customer(self):
        out = ra._fast_dispatch("any suspicious transactions for CUST001")
        assert out is not None
        assert "Fraud Check" in out

    def test_anomaly_phrasing(self):
        out = ra._fast_dispatch("check for anomaly on CUST003")
        assert out is not None
        assert "Fraud Check" in out

    def test_no_customer_id_returns_none(self):
        assert ra._fast_dispatch("is there any fraud happening") is None


class TestUnrelatedQuestion:
    def test_pricing_question_returns_none(self):
        """Risk agent must not try to answer questions outside its scope --
        pricing belongs to Finance."""
        assert ra._fast_dispatch("what is the price of PRD001") is None
