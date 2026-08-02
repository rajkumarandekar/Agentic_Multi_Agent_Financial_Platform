"""
credit_agent.py — TechMart India Credit Agent.

New in Phase 2 of the multi-agent expansion (see project chat history) —
pushes the platform from an e-commerce pricing assistant toward genuine
financial-services concepts: credit eligibility, EMI calculation, and a
loan-approval flow that reuses the SAME Human-in-the-Loop
interrupt()/Command(resume=...) pattern confirm_purchase already uses for
orders (see orchestration/confirm_loan.py) — a loan disbursement is exactly
the kind of consequential action that pattern exists for.

4 tools:
  1. check_credit_eligibility — available credit vs a requested amount
  2. calculate_emi            — pure EMI/amortization math, no DB lookup
  3. request_loan             — computes EMI + credit-limit eligibility,
     THEN consults the Risk agent's churn signal for the applicant before
     finalizing a recommendation (a genuinely cross-agent decision, not a
     Credit-only calculation), and produces a "Loan Proposal #..." that
     gates the HITL approve/reject step in orchestration/confirm_loan.py.
  4. prioritize_collections   — Phase 3 addition: ranks customers with an
     outstanding balance by how urgently to chase them, combining amount
     owed, days inactive, and (again) the Risk agent's churn signal --
     genuinely portfolio-level, not a single-customer lookup.
"""
import asyncio
import logging
import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agents.finance_agent import (
    _MODEL,
    COMPANY,
    _bullet_summary,
    _chart,
    _looks_like_leaked_tool_call,
    _normalize_id,
    _query_db,
)
from agents.risk_agent import predict_customer_risk

logger = logging.getLogger(__name__)

_LOAN_RATES = COMPANY.get("loans", {}).get("interest_rate_by_tier", {
    "Bronze": 18.0, "Silver": 15.0, "Gold": 12.0, "Platinum": 10.0,
})


def _emi(principal: float, annual_rate: float, months: int) -> float:
    """Standard reducing-balance EMI formula."""
    r = annual_rate / 12 / 100
    if r == 0:
        return round(principal / months, 2)
    return round(principal * r * (1 + r) ** months / ((1 + r) ** months - 1), 2)


def _get_customer_credit(customer_id: str) -> dict | None:
    rows = _query_db(
        "SELECT customer_id, customer_name, tier, credit_limit, outstanding_balance "
        "FROM customers WHERE customer_id = ?", (customer_id,),
    )
    return rows[0] if rows else None


@tool
def check_credit_eligibility(customer_id: str, requested_amount: float) -> str:
    """Check whether a customer has enough available credit for a requested
    amount. Use when asked: credit eligibility, can this customer afford,
    available credit, credit limit check."""
    customer_id = _normalize_id(customer_id, "CUST")
    c = _get_customer_credit(customer_id)
    if not c:
        return f"Customer {customer_id} was not found."

    available = round(c["credit_limit"] - c["outstanding_balance"], 2)
    eligible  = requested_amount <= available

    card = {
        "type":         "calculation",
        "title":        f"Credit Eligibility — {c['customer_name']}",
        "result_value": "Eligible" if eligible else "Not Eligible",
        "result_label": "Decision",
        "metrics": [
            {"label": "Credit Limit",       "value": f"₹{c['credit_limit']:,.2f}"},
            {"label": "Outstanding Balance", "value": f"₹{c['outstanding_balance']:,.2f}"},
            {"label": "Available Credit",    "value": f"₹{available:,.2f}"},
            {"label": "Requested Amount",    "value": f"₹{requested_amount:,.2f}"},
        ],
    }
    verdict = (
        f"**Eligible** — ₹{requested_amount:,.2f} is within the ₹{available:,.2f} available credit."
        if eligible else
        f"**Not eligible** — ₹{requested_amount:,.2f} exceeds the ₹{available:,.2f} available credit "
        f"(short by ₹{requested_amount - available:,.2f})."
    )
    return (
        _chart(card)
        + f"**Credit Eligibility — {c['customer_name']}** ({customer_id}, {c['tier']} tier)\n\n"
        f"{verdict}\n\n" + _bullet_summary(card)
    )


@tool
def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> str:
    """Calculate the monthly EMI for a loan given principal, annual interest
    rate, and tenure in months. Use when asked: EMI, monthly installment,
    what would I pay per month."""
    emi      = _emi(principal, annual_rate, tenure_months)
    total    = round(emi * tenure_months, 2)
    interest = round(total - principal, 2)

    card = {
        "type":         "calculation",
        "title":        "EMI Calculation",
        "result_value": f"₹{emi:,.2f}",
        "result_label": "Monthly EMI",
        "metrics": [
            {"label": "Principal",      "value": f"₹{principal:,.2f}"},
            {"label": "Annual Rate",    "value": f"{annual_rate}%"},
            {"label": "Tenure",         "value": f"{tenure_months} months"},
            {"label": "Total Repayment", "value": f"₹{total:,.2f}"},
            {"label": "Total Interest",  "value": f"₹{interest:,.2f}"},
        ],
    }
    return (
        _chart(card)
        + f"**EMI Calculation**\n\n"
        f"**Monthly EMI: ₹{emi:,.2f}** over {tenure_months} months at {annual_rate}% p.a.\n\n"
        f"Total repayment: ₹{total:,.2f} (₹{interest:,.2f} interest on ₹{principal:,.2f} principal)\n\n"
        + _bullet_summary(card)
    )


@tool
def request_loan(customer_id: str, principal: float, tenure_months: int = 12) -> str:
    """Generate a loan proposal for a customer -- computes EMI at their
    tier's interest rate, checks credit-limit eligibility, and factors in
    their churn-risk signal before recommending approval. The result is a
    PROPOSAL only; it requires explicit human approval before being final
    (see orchestration/confirm_loan.py).
    Use when asked: loan, apply for a loan, loan proposal, request credit."""
    customer_id = _normalize_id(customer_id, "CUST")
    c = _get_customer_credit(customer_id)
    if not c:
        return f"Customer {customer_id} was not found."

    tier = c["tier"]
    rate = _LOAN_RATES.get(tier, 15.0)
    emi  = _emi(principal, rate, tenure_months)
    total = round(emi * tenure_months, 2)
    available = round(c["credit_limit"] - c["outstanding_balance"], 2)
    credit_ok = principal <= available

    # Consult the Risk agent's churn signal for this applicant BEFORE
    # finalizing a recommendation -- a genuinely cross-agent decision, not
    # just a Credit-only calculation. A customer flagged High churn risk is
    # a materially different loan risk than a stable one, even at identical
    # credit-limit headroom.
    risk_text  = predict_customer_risk.invoke({"customer_id": customer_id})
    risk_label = "Unknown"
    m = re.search(r'\*\*Risk Level:\s*(High Risk|Medium Risk|Low Risk)\*\*', risk_text)
    if m:
        risk_label = m.group(1)

    if not credit_ok:
        recommendation = "DECLINE"
        reason = "requested principal exceeds available credit"
    elif risk_label == "High Risk":
        recommendation = "MANUAL REVIEW"
        reason = "customer flagged High churn risk — needs human underwriting"
    else:
        recommendation = "APPROVE"
        reason = "within available credit, acceptable churn risk"

    loan_num = f"LNP-TM-{date.today().strftime('%Y%m%d')}-{customer_id[-4:]}"
    today    = date.today().isoformat()

    card = {
        "type":         "calculation",
        "title":        f"Loan Proposal {loan_num}",
        "result_value": recommendation.title(),
        "result_label": "Recommendation",
        "metrics": [
            {"label": "Principal",               "value": f"₹{principal:,.2f}"},
            {"label": f"Interest Rate ({tier})", "value": f"{rate}%"},
            {"label": "Tenure",                  "value": f"{tenure_months} months"},
            {"label": "Monthly EMI",              "value": f"₹{emi:,.2f}"},
            {"label": "Total Repayment",          "value": f"₹{total:,.2f}"},
            {"label": "Available Credit",         "value": f"₹{available:,.2f}"},
            {"label": "Churn Risk",               "value": risk_label},
        ],
    }
    return (
        _chart(card)
        + f"**TechMart India — Loan Proposal #{loan_num}**\n"
        f"**Date:** {today}\n\n"
        f"| Applicant | {c['customer_name']} ({customer_id}) — {tier} Tier |\n"
        f"|---|---|\n"
        f"| Principal | ₹{principal:,.2f} |\n"
        f"| Interest Rate | {rate}% p.a. ({tier} tier) |\n"
        f"| Tenure | {tenure_months} months |\n"
        f"| Monthly EMI | ₹{emi:,.2f} |\n"
        f"| Total Repayment | ₹{total:,.2f} |\n"
        f"| Available Credit | ₹{available:,.2f} |\n"
        f"| Churn Risk Signal | {risk_label} |\n"
        f"| **Recommendation** | **{recommendation}** ({reason}) |\n\n"
        f"_This is a proposal only — requires explicit approval before the loan is final._"
        + "\n\n" + _bullet_summary(card)
    )


_RISK_WEIGHT = {"High Risk": 1.5, "Medium Risk": 1.0, "Low Risk": 0.6}


@tool
def prioritize_collections(top_n: int = 10) -> str:
    """Rank customers with an outstanding credit balance by collections
    priority -- combining amount owed, days inactive, and churn risk (a
    customer who owes a lot AND has gone quiet AND is flagged High churn
    risk is much harder to recover than one who owes a lot but is still
    actively buying).
    Use when asked: collections priority, who should we chase for payment,
    overdue accounts, outstanding balance priority list."""
    # High UTILIZATION (balance vs. limit), not merely nonzero -- almost
    # every customer carries some small balance by design (see
    # data/generate_company_data.py), so "outstanding_balance > 0" alone
    # would flag nearly the entire base and dilute the signal. >50% of
    # limit is a standard, defensible "needs attention" threshold.
    rows = _query_db(
        "SELECT customer_id, customer_name, tier, outstanding_balance, credit_limit, days_inactive "
        "FROM customers WHERE credit_limit > 0 AND outstanding_balance / credit_limit > 0.5 "
        "ORDER BY outstanding_balance DESC"
    )
    if not rows:
        return "No customers currently have a high-utilization outstanding balance."

    # Cap how many customers get a live risk-model call -- this is a
    # priority-ranking tool, not a full-portfolio audit; the top 3x pool by
    # raw balance already contains every plausible top-N candidate.
    pool = rows[: max(top_n * 3, top_n)]

    scored = []
    for r in pool:
        risk_text  = predict_customer_risk.invoke({"customer_id": r["customer_id"]})
        risk_label = "Unknown"
        m = re.search(r'\*\*Risk Level:\s*(High Risk|Medium Risk|Low Risk)\*\*', risk_text)
        if m:
            risk_label = m.group(1)
        weight = _RISK_WEIGHT.get(risk_label, 1.0)
        # Inactivity compounds the risk: a balance owed by someone who
        # hasn't bought in a year is harder to collect than the same
        # balance from someone active last week.
        score = r["outstanding_balance"] * weight * (1 + r["days_inactive"] / 365)
        scored.append({**r, "risk_label": risk_label, "priority_score": round(score, 2)})

    scored.sort(key=lambda x: -x["priority_score"])
    top = scored[:top_n]

    card = {
        "type":         "calculation",
        "title":        "Collections Priority List",
        "result_value": str(len(top)),
        "result_label": "Accounts Ranked",
        "metrics": [
            {"label": "Total Outstanding (high-utilization accounts)", "value": f"₹{sum(r['outstanding_balance'] for r in rows):,.2f}"},
            {"label": "High-Utilization Accounts (>50% of limit)",     "value": str(len(rows))},
        ],
    }
    lines = "\n".join(
        f"| {r['customer_id']} | {r['customer_name']} | {r['tier']} | "
        f"₹{r['outstanding_balance']:,.2f} | {r['days_inactive']}d | {r['risk_label']} | {r['priority_score']:,.0f} |"
        for r in top
    )
    return (
        _chart(card)
        + f"**Collections Priority List** (top {len(top)} of {len(rows)} accounts above 50% credit utilization)\n\n"
        f"| Customer | Name | Tier | Outstanding | Inactive | Churn Risk | Priority Score |\n"
        f"|---|---|---|---|---|---|---|\n{lines}\n\n"
        f"_Priority score = outstanding balance × churn-risk weight × (1 + days inactive / 365) "
        f"-- higher score means chase sooner._\n\n"
        + _bullet_summary(card)
    )


_ALL_TOOLS     = [check_credit_eligibility, calculate_emi, request_loan, prioritize_collections]
_TOOLS_BY_NAME = {t.name: t for t in _ALL_TOOLS}

_SYSTEM_PROMPT = (
    "You are TechMart India's Credit Agent. You ONLY handle credit "
    "eligibility checks, EMI calculations, loan proposals, and collections "
    "prioritization -- nothing else. Call exactly one tool per question. "
    "Map customer names to CUST IDs before calling. If product/pricing/"
    "churn questions come up, decline them in plain text — those belong "
    "to Finance/Risk agents."
)

# Matches CUST002, cust2, "customer 2", "customer no 2", "customer number 2",
# etc. -- real bug this fixes: "give me loan for customer 2" (spelled out,
# no CUST prefix) previously matched nothing at all, since the old pattern
# required the literal "CUST" prefix, so the customer id was silently
# dropped and the ReAct fallback got a question with no id it could act on.
_CID_RE    = re.compile(r'\bcust(?:omer)?\.?\s*(?:no\.?|number)?\s*0*(\d+)\b', re.IGNORECASE)
_TENURE_RE = re.compile(r'(\d{1,3})\s*months?\b', re.IGNORECASE)
_RATE_RE   = re.compile(r'(\d{1,2}(?:\.\d+)?)\s*%')
_AMOUNT_RE = re.compile(
    r'₹\s*([\d,]+(?:\.\d+)?)|(?:rs\.?|inr)\s*([\d,]+(?:\.\d+)?)|\b(\d{4,}(?:\.\d+)?)\b',
    re.IGNORECASE,
)


def _extract_amount(text: str) -> float | None:
    m = _AMOUNT_RE.search(text)
    if not m:
        return None
    raw = next(g for g in m.groups() if g)
    return float(raw.replace(",", ""))


def _extract_tenure(text: str, default: int = 12) -> int:
    m = _TENURE_RE.search(text)
    return int(m.group(1)) if m else default


def _fast_dispatch(question: str) -> str | None:
    """
    Deterministic regex dispatch, same pattern as finance_agent's. `question`
    may be the full contextual string built by _contextual_question; current-
    message text is extracted first via the "[Current question]" marker.
    """
    marker  = "[Current question]\n"
    current = question.split(marker, 1)[1].strip() if marker in question else question.strip()
    q_lower = current.lower()

    cid_match = _CID_RE.search(current)
    cid = f"CUST{int(cid_match.group(1)):03d}" if cid_match else None

    # ── Collections priority (portfolio-level, checked first -- no customer
    # id of its own, so it must not fall through to the credit-eligibility
    # branch below just because "credit"/"balance" words might overlap) ───
    if re.search(r'\b(collections?\s+priorit\w*|(who|which\s+customers?)\s+should\s+we\s+chase|'
                 r'overdue\s+account\w*|outstanding\s+balance\s+priorit\w*)\b', q_lower):
        n_m = re.search(r'\btop\s+(\d{1,3})\b', q_lower)
        top_n = int(n_m.group(1)) if n_m else 10
        return prioritize_collections.invoke({"top_n": top_n})

    # ── Loan proposal (checked before bare EMI -- "loan" implies EMI too) ──
    if re.search(r'\b(loan|apply\s+for\s+a\s+loan|request\s+credit)\b', q_lower):
        if not cid:
            return None
        amount = _extract_amount(current)
        if amount is None:
            return None
        tenure = _extract_tenure(current)
        return request_loan.invoke({
            "customer_id": cid, "principal": amount, "tenure_months": tenure,
        })

    # ── Bare EMI calculation (principal + rate% + tenure, no customer) ────
    if re.search(r'\bemi\b|monthly\s+installment', q_lower):
        amount = _extract_amount(current)
        rate_m = _RATE_RE.search(current)
        if amount is not None and rate_m:
            tenure = _extract_tenure(current)
            return calculate_emi.invoke({
                "principal": amount, "annual_rate": float(rate_m.group(1)),
                "tenure_months": tenure,
            })
        return None

    # ── Credit eligibility ─────────────────────────────────────────────
    if re.search(r'\b(credit\s+eligib\w*|available\s+credit|credit\s+limit|can\s+.*afford)\b', q_lower):
        if not cid:
            return None
        amount = _extract_amount(current)
        if amount is None:
            return None
        return check_credit_eligibility.invoke({
            "customer_id": cid, "requested_amount": amount,
        })

    return None


async def run(question: str, knowledge_result: dict | None = None, messages: list | None = None) -> str:
    """
    Answer a TechMart credit question (eligibility, EMI, or loan proposal).

    Priority order mirrors finance_agent.run(): deterministic fast dispatch
    first, full ReAct agent as a last resort for phrasing the regex
    patterns don't recognize.
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
            logger.warning("credit_agent: model hallucinated an unexecuted tool call for: %s", question)
            return (
                "I don't have a specific tool for that request. Could you ask about a "
                "specific customer's credit eligibility, EMI, or a loan application instead?"
            )
        return final_text
    except asyncio.TimeoutError:
        logger.error("credit_agent timed out for: %s", question)
        return "Request timed out. Please try again."
    except Exception as exc:
        logger.error("credit_agent error: %s", exc, exc_info=True)
        return "Unable to complete the credit analysis. Please rephrase the question."
