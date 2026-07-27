"""
Deterministic single follow-up question suggestion -- zero LLM calls.

Earlier this session a chip-based "Continue the conversation" follow-up
feature was removed because the suggestions were often irrelevant to what
was actually just answered. This is a narrower replacement: ONE suggestion,
chosen from a fixed table keyed to which finance tool actually ran -- so
it's always a genuine, on-topic next step for the exact answer just given,
never a generic "want to know more?" prompt. No LLM call at all, which also
matters under Groq TPD-quota pressure.

Most finance tools put a "title" on their <CHART_DATA> card; three
(forecast_revenue, category_performance, monthly_trend_analysis) use the
older "type": "forecast" card shape which has no title field at all, so
those are matched on their markdown heading text instead. Order matters:
entries are checked most-specific-substring-first so e.g. "Multi-Product
Bulk Quote" is matched before the shorter "Bulk Quote" (which is itself a
substring of it).
"""
import json
import re

_CHART_RE = re.compile(r"<CHART_DATA>\s*(\{.*?\})\s*</CHART_DATA>", re.DOTALL)

# (substring to match, suggested follow-up or None for "no natural next
# step"). Checked in order against the card's own "title" field.
_TITLE_RULES: list[tuple[str, str | None]] = [
    ("Multi-Product Order Summary",  "Want a bulk quote for these instead?"),
    ("Combined Loyalty Price",       "Want an invoice for one of these?"),
    ("Multi-Product Bulk Quote",     None),
    ("Customer Risk Comparison",     "Want to check either customer's lifetime value?"),
    ("Product Comparison",           "Want a bulk quote for the cheaper one?"),
    ("Selling Price",                "Want to see its profit margin too?"),
    ("Bulk Quote",                   "Want a loyalty discount quote for a specific customer instead?"),
    ("Loyalty Price",                "Want an invoice for this order?"),
    ("Loyalty Tier",                 "Want a full loyalty price quote for a specific product?"),
    ("Profit Margin",                "Want to see the full price breakdown too?"),
    ("Invoice",                      None),
    ("Churn Risk",                   "Want to compare this with another customer?"),
    ("Demand Forecast",              "Want to check if this product needs a bulk order?"),
    ("Lifetime Value",               "Want to check this customer's churn risk too?"),
    ("GST Impact",                   None),
    ("GST Comparison",               None),
]

# Fallback for the 3 tools whose card has no "title" field -- matched
# against the plain-text markdown heading instead.
_TEXT_HEADING_RULES: list[tuple[str, str | None]] = [
    # "-Month Revenue Trend" (not "18-Month...") -- monthly_trend_analysis's
    # heading now shows the ACTUAL number of months of data (see
    # agents/finance_agent.py), which grows over time as more data is
    # generated, so this must match regardless of the specific number.
    ("-Month Revenue Trend",  "Want a category performance breakdown too?"),
    ("Category Performance",   "Want to see the overall sales trend too?"),
    ("Revenue Forecast",       "Want to see the demand forecast for a specific product?"),
]


# Every follow-up string this module can ever produce. Exposed so
# orchestration/supervisor.py can recognize "this question IS one of our own
# follow-up suggestions" and never fast-path it: these are deliberately
# context-dependent ("...instead?", "...one of these?", "...this one?") --
# they only make sense in light of the answer they were suggested after, so
# they must always go through the LLM router (which sees conversation
# history), never the regex/semantic fast path (which doesn't).
ALL_FOLLOWUPS = frozenset(
    followup
    for _, followup in _TITLE_RULES + _TEXT_HEADING_RULES
    if followup is not None
)


# Matches the last sentence of a piece of text IF that sentence is itself a
# question -- i.e. everything after the last ".", "!", "?" or newline, up to
# a final "?". Used only as a fallback (see pending_followup() below) when
# suggest_followup()'s fixed table doesn't match -- the chat LLM asking its
# own free-form clarifying question ("which product and quantity would you
# like to generate an invoice for?") is NOT one of the deterministic
# templates, but it's exactly as context-dependent, and it used to leave
# NO trace in state for the next turn's router to resolve a bare "yes"/"ok"
# reply against.
_TRAILING_QUESTION_RE = re.compile(r'([^.!?\n]*\?)\s*$')
_MAX_TRAILING_QUESTION_LEN = 200


def _extract_trailing_question(answer: str) -> str | None:
    """Best-effort: if `answer`'s last sentence is a question, return it
    (trimmed), else None."""
    if not answer:
        return None
    m = _TRAILING_QUESTION_RE.search(answer.strip())
    if not m:
        return None
    question = m.group(1).strip()
    # Guard against grabbing a whole multi-paragraph answer that happens to
    # have no sentence terminators before its final "?" -- not the kind of
    # short, quotable question this is meant to capture.
    if not question or len(question) > _MAX_TRAILING_QUESTION_LEN:
        return None
    return question


def pending_followup_for(answer: str) -> str | None:
    """
    The single entry point response_node uses to decide what to persist as
    pending_followup: the deterministic suggest_followup() table takes
    priority (it's precise, keyed to the exact tool that ran); if that
    doesn't match, fall back to whatever question the answer itself ends
    on, however it was phrased. Either way, the caller gets one thing to
    track -- it doesn't need to know which source it came from.
    """
    return suggest_followup(answer) or _extract_trailing_question(answer)


def suggest_followup(answer: str) -> str | None:
    """Return one relevant follow-up question for this answer, or None if
    there genuinely isn't a natural next step (e.g. an invoice is usually
    the end of a flow, not a jumping-off point) or this isn't a finance
    card-carrying answer at all."""
    if not answer:
        return None

    m = _CHART_RE.search(answer)
    if m:
        try:
            card = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            card = {}
        title = card.get("title", "")
        if title:
            for substring, followup in _TITLE_RULES:
                if substring in title:
                    return followup
            return None

    for substring, followup in _TEXT_HEADING_RULES:
        if substring in answer:
            return followup
    return None
