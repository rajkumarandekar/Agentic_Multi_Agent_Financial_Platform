"""
Tests for agents/forecast_agent.py::_fast_dispatch — revenue forecasting
moved out of finance_agent.py (Phase 2) into a dedicated Forecast agent,
now with flexible day/week/month/year horizons instead of months only.

Month-horizon requests still delegate straight to finance_agent's own
trained ARIMA model output (forecast_revenue) — day/week/year requests are
the new capability, derived from that same model via
forecast_revenue_flexible (see agents/forecast_agent.py's module docstring
for why this is a scaled derivation, not a second model).
"""
import agents.forecast_agent as fca


class TestMonthHorizon:
    def test_default_months(self):
        out = fca._fast_dispatch("forecast revenue")
        assert out is not None
        assert "Revenue Forecast" in out

    def test_explicit_months(self):
        out = fca._fast_dispatch("forecast revenue for the next 5 months")
        assert out is not None
        assert "Next 5 Month" in out


class TestFlexibleHorizon:
    def test_days(self):
        out = fca._fast_dispatch("predict revenue for the next 10 days")
        assert out is not None
        assert "Next 10 Day" in out

    def test_weeks(self):
        out = fca._fast_dispatch("forecast revenue for the next 2 weeks")
        assert out is not None
        assert "Next 2 Week" in out

    def test_years(self):
        out = fca._fast_dispatch("forecast revenue for the next 1 year")
        assert out is not None
        assert "Next 1 Year" in out

    def test_wider_uncertainty_at_finer_granularity(self):
        """Day-level estimates should carry a wider uncertainty band than
        year-level ones -- a genuine forecasting property this agent
        surfaces explicitly rather than hiding."""
        day_out  = fca._fast_dispatch("forecast revenue for the next 1 days")
        year_out = fca._fast_dispatch("forecast revenue for the next 1 years")
        assert "±35%" in day_out
        assert "±10%" in year_out


class TestUnrelatedQuestion:
    def test_pricing_question_returns_none(self):
        assert fca._fast_dispatch("what is the price of PRD001") is None
