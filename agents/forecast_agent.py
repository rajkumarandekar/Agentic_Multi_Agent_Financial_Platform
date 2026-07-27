"""
forecast_agent.py — TechMart India Forecast Agent.

Owns revenue forecasting, split out of finance_agent.py (Phase 2 of the
multi-agent expansion). For a plain month-based horizon it reuses
finance_agent.forecast_revenue directly (the trained monthly ARIMA model's
own per-month predictions — unchanged, still the most precise view).

For day/week/year horizons, forecast_revenue_flexible derives an estimate
from that SAME monthly model by scaling, rather than training a second model
at finer granularity: an earlier attempt at fitting ARIMA directly on
daily/weekly revenue rollups produced a 111-263% test MAPE (the synthetic
transaction data is generated per calendar month, so day/week boundaries
don't carry its real signal) — worse than just guessing the average. One
model that's actually well-fit, with wider uncertainty bands the finer the
requested granularity (a genuine forecasting property, not an arbitrary
knob), is simpler and more honest than shipping a second weak model.
"""
import logging
import re

import numpy as np
from langchain_core.tools import tool

from agents.finance_agent import (
    FORECAST_METRICS,
    FORECAST_MODEL,
    _bullet_summary,
    _chart,
    forecast_revenue,
)

logger = logging.getLogger(__name__)

_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
# Wider uncertainty band the finer the requested granularity -- point
# estimates are inherently less reliable at fine granularity; this is a
# genuine forecasting characteristic, not a tuning knob to hide error.
_UNIT_CI_PCT = {"day": 0.35, "week": 0.25, "month": 0.15, "year": 0.10}


@tool
def forecast_revenue_flexible(horizon: int, unit: str = "month") -> str:
    """Forecast TechMart's total revenue for a flexible horizon -- N days,
    weeks, or years ahead (months use forecast_revenue instead) -- derived
    from the trained monthly ARIMA model by scaling.
    Use when asked: 'predict revenue for the next N days/weeks/years',
    'daily/weekly/yearly forecast'. unit must be one of: day, week, year."""
    if FORECAST_MODEL is None:
        return "Revenue forecast model not available — run models/train_models.py first."

    unit = unit.lower().rstrip("s") or "month"
    if unit not in _UNIT_DAYS:
        unit = "month"

    # Run the underlying monthly model forward far enough to cover the
    # requested horizon, then scale its average month down/up to the unit
    # actually asked for -- one coherent model, several honest views of it.
    months_needed = max(1, round(horizon * _UNIT_DAYS[unit] / 30))
    preds       = list(np.array(FORECAST_MODEL.forecast(months_needed)).flatten())
    monthly_avg = sum(preds) / len(preds)
    per_day     = monthly_avg / 30
    total       = per_day * _UNIT_DAYS[unit] * horizon

    ci_pct        = _UNIT_CI_PCT[unit]
    ci_lo, ci_hi  = total * (1 - ci_pct), total * (1 + ci_pct)
    mape          = FORECAST_METRICS.get("mape", "N/A")
    last          = FORECAST_METRICS.get("last_month", "2026-06")
    unit_label    = f"{unit}{'s' if horizon != 1 else ''}"

    card = {
        "type":         "forecast",
        "title":        f"Revenue Forecast — Next {horizon} {unit_label.title()}",
        "result_value": f"₹{total:,.0f}",
        "result_label": f"Predicted Revenue ({horizon} {unit_label})",
        "metrics": [
            {"label": "Lower Bound", "value": f"₹{ci_lo:,.0f}"},
            {"label": "Upper Bound", "value": f"₹{ci_hi:,.0f}"},
            {"label": "Based On",    "value": f"Monthly ARIMA model (MAPE {mape}%, trained through {last})"},
            {"label": "Uncertainty", "value": f"±{ci_pct*100:.0f}% (wider at finer granularity)"},
        ],
    }
    return (
        _chart(card)
        + f"**Revenue Forecast — Next {horizon} {unit_label}**\n\n"
        f"**Predicted Revenue: ₹{total:,.0f}** (range ₹{ci_lo:,.0f} – ₹{ci_hi:,.0f})\n\n"
        f"Derived from the trained monthly ARIMA model (MAPE {mape}%, trained "
        f"through {last}) by scaling to the requested {unit} horizon — day/week/"
        f"year estimates carry wider uncertainty than the model's native monthly view.\n\n"
        + _bullet_summary(card)
    )


_ALL_TOOLS     = [forecast_revenue, forecast_revenue_flexible]
_TOOLS_BY_NAME = {t.name: t for t in _ALL_TOOLS}

_HORIZON_RE = re.compile(
    r'\bnext\s+(\d{1,3})\s+(day|days|week|weeks|month|months|year|years)\b',
    re.IGNORECASE,
)


def _fast_dispatch(question: str) -> str | None:
    """
    Deterministic regex dispatch, same pattern as finance_agent's. `question`
    may be the full contextual string built by _contextual_question; current-
    message text is extracted first via the "[Current question]" marker.
    """
    marker  = "[Current question]\n"
    current = question.split(marker, 1)[1].strip() if marker in question else question.strip()
    q_lower = current.lower()

    if not re.search(
        r'\b(forecast|predict(ed)?\s+revenue|future\s+revenue|revenue\s+(next|for))\b',
        q_lower,
    ):
        return None

    m = _HORIZON_RE.search(q_lower)
    if m:
        horizon = int(m.group(1))
        unit    = m.group(2).rstrip("s")
    else:
        horizon, unit = 3, "month"

    if unit == "month":
        return forecast_revenue.invoke({"months_ahead": horizon})
    return forecast_revenue_flexible.invoke({"horizon": horizon, "unit": unit})


async def run(question: str, knowledge_result: dict | None = None, messages: list | None = None) -> str:
    """
    Answer a TechMart revenue-forecast question.

    Deterministic fast dispatch only -- no ReAct fallback. A forecast
    question that doesn't match "next N <unit>" phrasing defaults to a
    3-month forecast rather than guessing at a novel parameterization an
    LLM tool-call might hallucinate (e.g. an invalid unit).
    """
    dispatched = _fast_dispatch(question)
    if dispatched is not None:
        return dispatched
    return forecast_revenue.invoke({"months_ahead": 3})
