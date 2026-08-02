"""
risk_agent.py — TechMart India Risk Agent.

Owns churn/retention risk and transaction-level fraud/anomaly detection.
Split out of finance_agent.py (Phase 2 of the multi-agent expansion) so
churn prediction is a genuinely separate agent with its own routing and
graph node, not just one finance tool among many — agents/credit_agent.py
calls straight into this module (predict_customer_risk) so a loan decision
can factor in the applicant's churn signal before it's finalized.

2 tools:
  1. predict_customer_risk / compare_customer_risk — reused directly from
     finance_agent.py (RandomForest churn classifier); the functions still
     physically live there and are imported here rather than duplicated.
  2. detect_fraud_alerts — flags a customer's own transactions that are
     anomalously large relative to their own historical average order value
     (a simple z-score anomaly check, not a trained model).
"""
import asyncio
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agents.groq_errors import is_rate_limited, RATE_LIMIT_MESSAGE

from agents.finance_agent import (
    _MODEL,
    _bullet_summary,
    _chart,
    _get_customer,
    _looks_like_leaked_tool_call,
    _normalize_id,
    _query_db,
    compare_customer_risk,
    predict_customer_risk,
)

logger = logging.getLogger(__name__)

# A transaction more than 2 standard deviations above the customer's OWN
# historical average order value is flagged — deliberately relative to each
# customer's own baseline, not a fixed rupee threshold, since a Platinum-tier
# heavy shopper's normal order is a Bronze-tier customer's anomaly.
_FRAUD_Z_THRESHOLD = 2.0


@tool
def detect_fraud_alerts(customer_id: str) -> str:
    """Flag a customer's own transactions that are anomalously large compared
    to their typical order value — a simple fraud/anomaly signal.
    Use when asked: fraud, suspicious activity, anomaly, unusual transaction,
    flag this customer, is this order suspicious."""
    customer_id = _normalize_id(customer_id, "CUST")
    cust = _get_customer(customer_id)
    if not cust:
        return f"Customer {customer_id} was not found."

    rows = _query_db(
        "SELECT transaction_id, product_name, final_amount, date FROM transactions "
        "WHERE customer_id = ? AND status = 'Completed' ORDER BY date",
        (customer_id,),
    )
    if len(rows) < 3:
        return (
            f"**Fraud Check — {cust['customer_name']}** ({customer_id})\n\n"
            f"Not enough transaction history ({len(rows)} completed order(s)) to "
            f"establish a reliable baseline — no anomaly can be flagged yet."
        )

    amounts  = [r["final_amount"] for r in rows]
    mean     = sum(amounts) / len(amounts)
    variance = sum((a - mean) ** 2 for a in amounts) / len(amounts)
    std      = variance ** 0.5

    flagged = []
    for r in rows:
        z = (r["final_amount"] - mean) / std if std > 0 else 0.0
        if z >= _FRAUD_Z_THRESHOLD:
            flagged.append({**r, "z_score": round(z, 2)})

    card = {
        "type":         "calculation",
        "title":        f"Fraud Check — {cust['customer_name']}",
        "result_value": str(len(flagged)),
        "result_label": "Flagged Transactions",
        "metrics": [
            {"label": "Orders Reviewed",      "value": str(len(rows))},
            {"label": "Average Order Value",  "value": f"₹{mean:,.2f}"},
            {"label": "Flag Threshold",        "value": f"{_FRAUD_Z_THRESHOLD}σ above average"},
        ],
    }

    if not flagged:
        return (
            _chart(card)
            + f"**Fraud Check — {cust['customer_name']}** ({customer_id})\n\n"
            f"No anomalous transactions found across {len(rows)} completed orders "
            f"(average ₹{mean:,.2f}/order). Nothing flagged.\n\n"
            + _bullet_summary(card)
        )

    lines = "\n".join(
        f"| {f['transaction_id']} | {f['product_name']} | ₹{f['final_amount']:,.2f} | "
        f"{f['z_score']}σ | {f['date']} |"
        for f in flagged
    )
    return (
        _chart(card)
        + f"**Fraud Check — {cust['customer_name']}** ({customer_id})\n\n"
        f"**{len(flagged)} anomalous transaction(s)** out of {len(rows)} "
        f"(average order ₹{mean:,.2f}):\n\n"
        f"| Transaction | Product | Amount | Deviation | Date |\n|---|---|---|---|---|\n{lines}\n\n"
        + _bullet_summary(card)
    )


_ALL_TOOLS     = [predict_customer_risk, compare_customer_risk, detect_fraud_alerts]
_TOOLS_BY_NAME = {t.name: t for t in _ALL_TOOLS}

_SYSTEM_PROMPT = (
    "You are TechMart India's Risk Agent. You ONLY handle two things: "
    "churn/retention risk prediction and transaction fraud/anomaly checks. "
    "Call exactly one tool per question. If neither genuinely fits, say so "
    "in plain text rather than forcing a mismatched tool call."
)

# Matches CUST002, cust2, "customer 2", "customer no 2", "customer number 2",
# etc. -- see agents/credit_agent.py's identical fix for the real bug this
# guards against (spelled-out "customer N" phrasing with no CUST prefix).
_CID_RE = re.compile(r'\bcust(?:omer)?\.?\s*(?:no\.?|number)?\s*0*(\d+)\b', re.IGNORECASE)


def _fast_dispatch(question: str) -> str | None:
    """
    Deterministic regex dispatch — same pattern as
    finance_agent.py::_fast_dispatch. `question` may be the full contextual
    string built by orchestration/graph.py::_contextual_question; current-
    message text is extracted first via the "[Current question]" marker.
    """
    marker  = "[Current question]\n"
    current = question.split(marker, 1)[1].strip() if marker in question else question.strip()
    q_lower = current.lower()

    if re.search(r'\b(fraud|suspicious|anomaly|anomalous|unusual\s+transaction|flag\s+this\s+customer)\b', q_lower):
        m = _CID_RE.search(current)
        if m:
            return detect_fraud_alerts.invoke({"customer_id": f"CUST{int(m.group(1)):03d}"})
        return None

    if re.search(r'\b(churn|at\s+risk|retention|will\s+.*\bleave\b|inactive\s+customer)\b', q_lower):
        # Only compare when the question itself signals a comparison --
        # otherwise a plain single-customer churn question must not merge
        # in an unrelated customer id that happens to appear in history.
        wants_comparison = bool(re.search(r'\b(compare|between|versus|vs\.?|both)\b', q_lower))
        ids = [f"CUST{int(n):03d}" for n in _CID_RE.findall(current)]
        if wants_comparison and len(ids) >= 2:
            return compare_customer_risk.invoke({"customer_ids": ids[:2]})
        if ids:
            return predict_customer_risk.invoke({"customer_id": ids[0]})
        return None

    return None


async def run(question: str, knowledge_result: dict | None = None, messages: list | None = None) -> str:
    """
    Answer a TechMart risk question (churn or fraud).

    Priority order mirrors finance_agent.run(): deterministic fast dispatch
    first (free, instant), full ReAct agent as a last resort for phrasing
    the regex patterns don't recognize.

    knowledge_result/messages accepted for the same graph-node calling
    convention finance_agent.run() uses, currently unused.
    """
    from langchain_core.messages import ToolMessage  # local import avoids circular

    dispatched = _fast_dispatch(question)
    if dispatched is not None:
        return dispatched

    # max_tokens caps the RESERVED completion budget Groq counts toward its
    # TPM rate limit -- see agents/sql_agent.py's identical comment.
    llm   = ChatGroq(model=_MODEL, temperature=0, max_tokens=1024, max_retries=1)
    agent = create_react_agent(
        llm, _ALL_TOOLS,
        state_modifier=SystemMessage(content=_SYSTEM_PROMPT),
        checkpointer=None,   # REQUIRED -- prevents MultipleSubgraphsError
    )
    try:
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config={"recursion_limit": 10},
            ),
            timeout=60.0,
        )
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        if tool_msgs:
            return tool_msgs[0].content
        final_text = result["messages"][-1].content
        if _looks_like_leaked_tool_call(final_text):
            logger.warning("risk_agent: model hallucinated an unexecuted tool call for: %s", question)
            return (
                "I don't have a specific tool for that request. Could you ask about a "
                "specific customer's churn risk or a suspicious transaction check instead?"
            )
        return final_text
    except asyncio.TimeoutError:
        logger.error("risk_agent timed out for: %s", question)
        return "Request timed out. Please try again."
    except Exception as exc:
        logger.error("risk_agent error: %s", exc, exc_info=True)
        if is_rate_limited(exc):
            return RATE_LIMIT_MESSAGE
        return "Unable to complete the risk analysis. Please rephrase the question."
