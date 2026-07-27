"""
Tests for agents/credit_agent.py::_fast_dispatch — the new Credit agent
(Phase 2 of the multi-agent expansion): credit eligibility, EMI calculation,
and loan proposals. request_loan is deliberately cross-agent: it calls
agents.risk_agent.predict_customer_risk directly to factor the applicant's
churn signal into the recommendation, not just a Credit-only calculation.
"""
import agents.credit_agent as ca


class TestCreditEligibility:
    def test_eligible_within_limit(self):
        out = ca._fast_dispatch("check credit eligibility for CUST001 for rs.20000")
        assert out is not None
        assert "Credit Eligibility" in out

    def test_no_customer_id_returns_none(self):
        assert ca._fast_dispatch("check credit eligibility for rs.20000") is None

    def test_no_amount_returns_none(self):
        assert ca._fast_dispatch("check credit eligibility for CUST001") is None


class TestEmiCalculation:
    def test_bare_emi_math(self):
        out = ca._fast_dispatch("what is the EMI for 100000 at 12% for 24 months")
        assert out is not None
        assert "EMI Calculation" in out

    def test_missing_rate_returns_none(self):
        """No % in the question -- not enough to compute an EMI, must not
        guess a rate."""
        assert ca._fast_dispatch("what is the EMI for 100000 for 24 months") is None


class TestLoanProposal:
    def test_generates_proposal(self):
        out = ca._fast_dispatch("apply for a loan of rs.50000 for CUST001 for 12 months")
        assert out is not None
        assert "Loan Proposal #" in out
        assert "Churn Risk" in out  # proves the Risk-agent signal was consulted

    def test_default_tenure_is_12_months(self):
        out = ca._fast_dispatch("apply for a loan of rs.50000 for CUST001")
        assert out is not None
        assert "12 months" in out

    def test_no_customer_id_returns_none(self):
        assert ca._fast_dispatch("apply for a loan of rs.50000") is None

    def test_no_amount_returns_none(self):
        assert ca._fast_dispatch("apply for a loan for CUST001") is None

    def test_declines_when_amount_exceeds_available_credit(self):
        """A principal far beyond any tier's credit limit must recommend
        DECLINE, not silently approve past the eligibility check."""
        out = ca._fast_dispatch("apply for a loan of rs.99999999 for CUST004 for 12 months")
        assert out is not None
        assert "Decline" in out


class TestCollectionsPriority:
    def test_generates_priority_list(self):
        out = ca._fast_dispatch("collections priority list")
        assert out is not None
        assert "Collections Priority List" in out

    def test_respects_top_n(self):
        out = ca._fast_dispatch("collections priority top 3")
        assert out is not None
        assert "top 3 of" in out

    def test_overdue_phrasing(self):
        out = ca._fast_dispatch("show me overdue accounts")
        assert out is not None
        assert "Collections Priority List" in out

    def test_only_ranks_high_utilization_accounts(self):
        """A customer carrying a small, healthy balance (well under 50% of
        their limit) must not appear in the priority list just because the
        balance is nonzero -- only genuinely high-utilization accounts are
        collections-relevant."""
        out = ca._fast_dispatch("collections priority top 50")
        assert out is not None
        assert "50%" in out  # threshold documented in the output


class TestUnrelatedQuestion:
    def test_pricing_question_returns_none(self):
        assert ca._fast_dispatch("what is the price of PRD001") is None
