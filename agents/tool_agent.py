"""
Personal Financial Advisor agent — native tools, no subprocess required.

13 tools covering: transaction analysis, financial health scoring, spend
forecasting, goal planning, investment comparison, anomaly detection, and
financial math (CAGR, ROI, EMI, compound interest, tax, etc.).

Stdlib only: sqlite3, math, statistics, datetime, os.
"""

import json
import math
import os
import sqlite3
import statistics
from datetime import date, datetime, timedelta

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shipments.db")


# ---------------------------------------------------------------------------
# Internal helper — not a tool
# ---------------------------------------------------------------------------

def _get_customer_data(customer_id: str, days: int = 60, category: str = None) -> list[dict]:
    """
    Query transactions table for a given customer over the last N days.
    Excludes salary credits (amount < 0). Returns list of row dicts.
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if category:
            rows = conn.execute(
                "SELECT * FROM transactions "
                "WHERE customer_id=? AND date>=? AND category=? AND amount>0 "
                "ORDER BY date",
                (customer_id, cutoff, category),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM transactions "
                "WHERE customer_id=? AND date>=? AND amount>0 "
                "ORDER BY date",
                (customer_id, cutoff),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _get_latest_balance(customer_id: str) -> float:
    """Return the most recent balance for a customer."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT balance FROM transactions WHERE customer_id=? ORDER BY date DESC, transaction_id DESC LIMIT 1",
            (customer_id,),
        ).fetchone()
        return row[0] if row else 0.0
    finally:
        conn.close()


def _customer_name(customer_id: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT customer_name FROM transactions WHERE customer_id=? LIMIT 1",
            (customer_id,),
        ).fetchone()
        return row[0] if row else customer_id
    finally:
        conn.close()


def _chart_block(data: dict) -> str:
    """Wrap structured data in a sentinel tag for frontend rendering."""
    return f"<CHART_DATA>\n{json.dumps(data, ensure_ascii=False)}\n</CHART_DATA>\n\n"


# ---------------------------------------------------------------------------
# TOOL 1 — query_customer_transactions
# ---------------------------------------------------------------------------

@tool
def query_customer_transactions(customer_id: str, days: int = 60, category: str = None) -> str:
    """Fetch and summarise customer transactions from the database.
    Use this FIRST before any analysis. Returns spend summary by category,
    top merchants, daily averages, highest/lowest spend day, transaction count.
    customer_id must be CUST001 or CUST002. category is optional (e.g. 'Groceries')."""
    rows = _get_customer_data(customer_id, days, category)
    if not rows:
        return f"No transactions found for {customer_id} in the last {days} days."

    name = rows[0]["customer_name"]
    total = sum(r["amount"] for r in rows)
    count = len(rows)

    by_cat: dict[str, float] = {}
    by_merchant: dict[str, float] = {}
    by_day: dict[str, float] = {}

    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["amount"]
        by_merchant[r["merchant"]] = by_merchant.get(r["merchant"], 0) + r["amount"]
        by_day[r["date"]] = by_day.get(r["date"], 0) + r["amount"]

    n_days = max((date.today() - (date.today() - timedelta(days=days))).days, 1)
    avg_daily = total / n_days

    top5 = sorted(by_merchant.items(), key=lambda x: x[1], reverse=True)[:5]

    high_day, high_amt = max(by_day.items(), key=lambda x: x[1])
    low_day, low_amt  = min(by_day.items(), key=lambda x: x[1])

    balance = _get_latest_balance(customer_id)

    lines = [
        f"{'='*48}",
        f"📊 TRANSACTION SUMMARY — {name} ({customer_id})",
        f"Period: last {days} days | {count} transactions",
        f"{'='*48}",
        f"Total Spend:    ₹{total:,.0f}",
        f"Avg Daily:      ₹{avg_daily:,.0f}",
        f"Current Balance:₹{balance:,.0f}",
        f"",
        f"💳 Spend by Category:",
    ]
    for cat, amt in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
        pct = amt / total * 100
        lines.append(f"  {cat:<18} ₹{amt:>8,.0f}  ({pct:.1f}%)")

    lines += [
        f"",
        f"🏪 Top 5 Merchants:",
    ]
    for merchant, amt in top5:
        lines.append(f"  {merchant:<20} ₹{amt:>8,.0f}")

    lines += [
        f"",
        f"📅 Highest Spend Day: {high_day}  ₹{high_amt:,.0f}",
        f"📅 Lowest Spend Day:  {low_day}   ₹{low_amt:,.0f}",
        f"{'='*48}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL 2 — financial_health_score
# ---------------------------------------------------------------------------

@tool
def financial_health_score(customer_id: str, monthly_income: float = 80000) -> str:
    """Comprehensive financial health score 0-100 with grade A/B/C/D.
    Use when asked: analyse finances, health check, am I on track,
    financial report, how am I doing, score my finances."""
    rows = _get_customer_data(customer_id, days=60)
    if not rows:
        return f"No transaction data found for {customer_id}."

    name = rows[0]["customer_name"]
    total_spend = sum(r["amount"] for r in rows)
    days = 60
    income_60 = monthly_income * 2  # 2-month window

    # Category buckets
    essentials_cats = {"Groceries", "Transport", "Utilities"}
    wants_cats      = {"Dining", "Entertainment", "Shopping"}

    by_cat: dict[str, float] = {}
    by_day: dict[str, float] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["amount"]
        by_day[r["date"]] = by_day.get(r["date"], 0) + r["amount"]

    essentials = sum(v for k, v in by_cat.items() if k in essentials_cats)
    wants      = sum(v for k, v in by_cat.items() if k in wants_cats)

    savings_rate      = (income_60 - total_spend) / income_60 * 100
    essentials_ratio  = essentials / income_60 * 100
    wants_ratio       = wants      / income_60 * 100

    daily_totals = list(by_day.values())
    try:
        consistency_stdev = statistics.stdev(daily_totals)
    except statistics.StatisticsError:
        consistency_stdev = 0.0

    balance = _get_latest_balance(customer_id)
    avg_daily_spend = total_spend / days if days else 1
    emergency_buffer = balance / avg_daily_spend if avg_daily_spend else 0

    # Scoring
    s_savings = 30 if savings_rate > 30 else (20 if savings_rate >= 20 else (10 if savings_rate >= 10 else 0))
    s_essent  = 25 if essentials_ratio < 50 else (15 if essentials_ratio <= 60 else 5)
    s_wants   = 20 if wants_ratio < 30 else (10 if wants_ratio <= 40 else 0)
    s_consist = 15 if consistency_stdev < 500 else (8 if consistency_stdev <= 1500 else 3)
    s_buffer  = 10 if emergency_buffer > 30 else (6 if emergency_buffer >= 15 else 2)

    score = s_savings + s_essent + s_wants + s_consist + s_buffer
    grade = "A" if score >= 80 else ("B" if score >= 65 else ("C" if score >= 50 else "D"))
    stars = "⭐⭐⭐⭐⭐" if score >= 90 else ("⭐⭐⭐⭐" if score >= 80 else ("⭐⭐⭐" if score >= 65 else ("⭐⭐" if score >= 50 else "⭐")))

    # Recommendations
    recs = []
    areas = [
        (s_savings, 30, f"Boost savings — currently {savings_rate:.1f}%, target 30%+"),
        (s_wants,   20, f"Reduce discretionary (dining/shopping/entertainment) — {wants_ratio:.1f}% of income"),
        (s_essent,  25, f"Essentials at {essentials_ratio:.1f}% — review transport/utilities for savings"),
        (s_consist, 15, f"Smooth daily spending — high volatility (stdev ₹{consistency_stdev:,.0f})"),
        (s_buffer,  10, f"Build emergency fund — only {emergency_buffer:.0f} days of runway"),
    ]
    for score_val, max_val, msg in sorted(areas, key=lambda x: x[1] - x[0], reverse=True)[:3]:
        recs.append(msg)

    def _bd_status(pts, mx):
        return "good" if pts == mx else ("ok" if pts >= mx * 0.5 else "low")

    chart = {
        "type": "health_score",
        "title": "Financial Health Report",
        "customer": name,
        "score": score,
        "grade": grade,
        "summary": (
            f"Score {score}/100 (Grade {grade}) over last 60 days. "
            f"{recs[0] if recs else 'Keep up current habits.'}"
        ),
        "breakdown": [
            {"label": "Savings Rate",  "value": f"{savings_rate:.0f}%",   "points": s_savings, "max": 30, "status": _bd_status(s_savings, 30)},
            {"label": "Essentials",    "value": f"{essentials_ratio:.0f}%","points": s_essent,  "max": 25, "status": _bd_status(s_essent, 25)},
            {"label": "Wants",         "value": f"{wants_ratio:.0f}%",     "points": s_wants,   "max": 20, "status": "good" if s_wants == 20 else ("ok" if s_wants == 10 else "high")},
            {"label": "Consistency",   "value": "Low" if s_consist == 15 else ("Moderate" if s_consist == 8 else "High"), "points": s_consist, "max": 15, "status": _bd_status(s_consist, 15)},
            {"label": "Buffer",        "value": f"{emergency_buffer:.0f} days", "points": s_buffer, "max": 10, "status": _bd_status(s_buffer, 10)},
        ],
        "metrics": {
            "total_spend":    round(total_spend),
            "savings_rate":   f"{savings_rate:.1f}%",
            "balance":        round(balance),
            "monthly_income": round(monthly_income),
        },
        "chart_data": [
            {"name": "Savings Rate", "score": s_savings, "max": 30},
            {"name": "Essentials",   "score": s_essent,  "max": 25},
            {"name": "Wants",        "score": s_wants,   "max": 20},
            {"name": "Consistency",  "score": s_consist, "max": 15},
            {"name": "Buffer",       "score": s_buffer,  "max": 10},
        ],
        "recommendations": recs[:3],
    }
    plain = (
        f"Financial Health Score: {score}/100 (Grade {grade}) for {name}. "
        f"Savings rate {savings_rate:.1f}%, essentials {essentials_ratio:.1f}%, wants {wants_ratio:.1f}%. "
        f"Top priority: {recs[0] if recs else 'Maintain current habits.'}"
    )
    return _chart_block(chart) + plain


# ---------------------------------------------------------------------------
# TOOL 3 — spend_forecast
# ---------------------------------------------------------------------------

@tool
def spend_forecast(customer_id: str) -> str:
    """Forecast next 30 days spending using weighted moving average.
    Recent weeks weighted higher (0.1, 0.2, 0.3, 0.4).
    Use when asked: predict spend, forecast, next month estimate, spending trend."""
    rows = _get_customer_data(customer_id, days=60)
    if not rows:
        return f"No transaction data found for {customer_id}."

    today = date.today()
    weights = [0.1, 0.2, 0.3, 0.4]  # oldest → newest
    window_days = 14

    # Build 4 windows of 14 days each (oldest first)
    windows: list[dict[str, float]] = [{} for _ in range(4)]
    for r in rows:
        tx_date = date.fromisoformat(r["date"])
        days_ago = (today - tx_date).days
        if days_ago >= 56:
            continue  # beyond 4 windows
        w_idx = 3 - (days_ago // window_days)  # 0=oldest, 3=newest
        if 0 <= w_idx <= 3:
            cat = r["category"]
            windows[w_idx][cat] = windows[w_idx].get(cat, 0) + r["amount"]

    # Per-window daily averages by category
    all_cats = set()
    for w in windows:
        all_cats.update(w.keys())

    # Compare most recent two windows for trend
    prev_total  = sum(windows[2].values())
    recent_total = sum(windows[3].values())

    lines = [
        f"{'='*56}",
        f"📅 SPEND FORECAST — Next 30 Days",
        f"Customer: {_customer_name(customer_id)} ({customer_id})",
        f"{'='*56}",
        f"{'Category':<18} | {'Forecast':>10} | {'Trend':<12} | vs Last 2w",
        f"{'-'*56}",
    ]

    total_forecast = 0.0
    alerts = []
    chart_rows: list[dict] = []
    for cat in sorted(all_cats):
        daily_weighted = 0.0
        for w_idx, w in enumerate(windows):
            daily_amt = w.get(cat, 0) / window_days
            daily_weighted += daily_amt * weights[w_idx]

        forecast_30 = daily_weighted * 30
        total_forecast += forecast_30

        # Trend: compare window[3] vs window[2]
        prev_cat  = windows[2].get(cat, 0)
        rec_cat   = windows[3].get(cat, 0)
        if prev_cat > 0:
            chg = (rec_cat - prev_cat) / prev_cat * 100
        else:
            chg = 0.0

        if chg > 10:
            trend = "↑ Rising"
            trend_key = "rising"
            alerts.append(f"⚠️  {cat} trending +{chg:.0f}% — consider reviewing")
        elif chg < -10:
            trend = "↓ Falling"
            trend_key = "falling"
        else:
            trend = "→ Stable"
            trend_key = "stable"

        chart_rows.append({"name": cat, "value": round(forecast_30), "trend": trend_key, "change_pct": round(chg, 1)})
        chg_str = f"+{chg:.0f}%" if chg >= 0 else f"{chg:.0f}%"
        lines.append(f"  {cat:<16} | ₹{forecast_30:>8,.0f} | {trend:<12} | {chg_str}")

    monthly_income_est = 80000  # default; shown as reference
    proj_save = monthly_income_est - total_forecast
    proj_rate = proj_save / monthly_income_est * 100

    chart = {
        "type": "forecast",
        "title": "Spend Forecast — Next 30 Days",
        "total_forecast": round(total_forecast),
        "income": monthly_income_est,
        "projected_savings": round(proj_save),
        "chart_data": [
            {"name": r["name"], "forecast": r["value"], "trend": r["trend"], "change": r["change_pct"]}
            for r in chart_rows[:10]
        ],
    }
    plain = (
        f"Forecast: ₹{total_forecast:,.0f} spend next 30 days, "
        f"projected savings ₹{proj_save:,.0f} ({proj_rate:.1f}%). "
        f"{alerts[0] if alerts else 'No significant trend changes detected.'}"
    )
    return _chart_block(chart) + plain


# ---------------------------------------------------------------------------
# TOOL 4 — goal_planner
# ---------------------------------------------------------------------------

@tool
def goal_planner(
    target_amount: float,
    years: float,
    monthly_income: float,
    customer_id: str = None,
) -> str:
    """Plan savings goal with 3 options and feasibility check.
    Use when asked: save for X, reach Y amount, how long to save Z, goal planning.
    Provide target_amount (₹), years, monthly_income. customer_id is optional."""
    months = years * 12

    # Option 1: simple monthly savings needed
    simple_monthly = target_amount / months

    # Option 2: savings account at 6% p.a. (monthly compounding)
    r2 = 0.06 / 12
    if r2 > 0:
        fv_factor2 = ((1 + r2) ** months - 1) / r2
        sa_monthly = target_amount / fv_factor2
    else:
        sa_monthly = simple_monthly

    # Option 3: SIP at 12% p.a. (monthly compounding)
    r3 = 0.12 / 12
    fv_factor3 = ((1 + r3) ** months - 1) / r3
    sip_monthly = target_amount / fv_factor3

    # Real surplus from DB if customer_id given
    if customer_id:
        rows = _get_customer_data(customer_id, days=60)
        if rows:
            total_60 = sum(r["amount"] for r in rows)
            avg_monthly_spend = total_60 / 2
        else:
            avg_monthly_spend = monthly_income * 0.7
    else:
        avg_monthly_spend = monthly_income * 0.7

    actual_surplus = monthly_income - avg_monthly_spend
    months_at_current = target_amount / actual_surplus if actual_surplus > 0 else float("inf")

    def feasibility(needed: float) -> str:
        if actual_surplus <= 0:
            return "❌ No surplus"
        if needed <= actual_surplus:
            return f"✅ FEASIBLE (surplus ₹{actual_surplus:,.0f})"
        return f"❌ Needs ₹{needed - actual_surplus:,.0f} more/month"

    saving_vs1 = simple_monthly - sa_monthly
    saving_vs3 = simple_monthly - sip_monthly

    lines = [
        f"{'='*52}",
        f"🎯 GOAL PLANNER: ₹{target_amount:,.0f} in {years:.1f} years",
        f"{'='*52}",
        f"Option 1 — Simple Save:     ₹{simple_monthly:>9,.0f}/month",
        f"  {feasibility(simple_monthly)}",
        f"",
        f"Option 2 — Savings A/c (6%):₹{sa_monthly:>9,.0f}/month  ✅ saves ₹{saving_vs1:,.0f}",
        f"  {feasibility(sa_monthly)}",
        f"",
        f"Option 3 — SIP (12% p.a.):  ₹{sip_monthly:>9,.0f}/month  ✅ saves ₹{saving_vs3:,.0f}",
        f"  {feasibility(sip_monthly)}",
        f"",
        f"Your monthly surplus: ₹{actual_surplus:,.0f}",
    ]

    if months_at_current < months * 12:
        lines.append(f"At current rate: goal in {months_at_current:.1f} months "
                     f"({'ahead of schedule! 🎉' if months_at_current < months else 'on track'})")
    else:
        lines.append(f"At current rate: goal in {months_at_current:.1f} months")

    best = "SIP at 12%" if actual_surplus >= sip_monthly else ("Savings A/c at 6%" if actual_surplus >= sa_monthly else "Simple savings")

    chart = {
        "type": "goal",
        "title": f"Goal: ₹{target_amount:,.0f} in {years:.1f} years",
        "target": target_amount,
        "years": years,
        "surplus": round(actual_surplus),
        "months_to_goal": round(months_at_current, 1) if months_at_current != float("inf") else None,
        "breakdown": [
            {"label": "Simple Savings",   "monthly": round(simple_monthly), "feasible": actual_surplus >= simple_monthly},
            {"label": "Savings A/c (6%)", "monthly": round(sa_monthly),     "feasible": actual_surplus >= sa_monthly},
            {"label": "SIP (12% p.a.)",   "monthly": round(sip_monthly),    "feasible": actual_surplus >= sip_monthly},
        ],
        "recommendation": best,
        "metrics": {
            "Target":       f"₹{target_amount:,.0f}",
            "Your Surplus": f"₹{actual_surplus:,.0f}/mo",
            "Best Option":  best,
        },
    }

    eta = f"{months_at_current:.1f} months" if months_at_current != float("inf") else "N/A"
    plain = (
        f"Goal ₹{target_amount:,.0f} in {years:.1f} years. "
        f"Best option: {best} at ₹{sip_monthly:,.0f}/month. "
        f"Your surplus ₹{actual_surplus:,.0f}/month — at current rate, reach goal in {eta}."
    )
    return _chart_block(chart) + plain


# ---------------------------------------------------------------------------
# TOOL 5 — investment_comparison
# ---------------------------------------------------------------------------

@tool
def investment_comparison(principal: float, years: int, rates_str: str) -> str:
    """Compare investment options. Pass rates as 'FD:7,SIP:12,PPF:7.1'.
    Use when asked: compare investments, FD vs SIP, best option, which is better."""
    INFLATION = 6.0

    options: dict[str, float] = {}
    for part in rates_str.split(","):
        part = part.strip()
        if ":" in part:
            name, rate = part.split(":", 1)
            try:
                options[name.strip()] = float(rate.strip())
            except ValueError:
                pass

    if not options:
        return "Could not parse rates_str. Use format: 'FD:7,SIP:12,PPF:7.1'"

    lines = [
        f"{'='*60}",
        f"📊 INVESTMENT COMPARISON",
        f"Principal: ₹{principal:,.0f} | Period: {years} years",
        f"{'='*60}",
        f"{'Option':<8} | {'Rate':>6} | {'Maturity':>12} | {'Real Return':>12} | {'Ann. Yield':>10}",
        f"{'-'*60}",
    ]

    best_name, best_val = "", 0.0
    results = []
    for name, rate in options.items():
        r = rate / 100
        # Quarterly compounding
        n = 4
        maturity = principal * (1 + r / n) ** (n * years)
        real_return_pct = ((1 + r) / (1 + INFLATION / 100) - 1) * 100
        effective_annual = (1 + r / n) ** n - 1
        monthly_equiv = (maturity - principal) / (years * 12)
        results.append((name, rate, maturity, real_return_pct, effective_annual * 100, monthly_equiv))
        if maturity > best_val:
            best_val, best_name = maturity, name

    chart_rows = [
        {"name": n, "value": round(m), "rate": r, "real_return": round(rr, 1)}
        for n, r, m, rr, _, __ in results
    ]
    chart = {
        "type": "comparison",
        "title": "Investment Comparison",
        "chart_data": chart_rows,
        "winner": best_name,
        "metrics": {
            "Principal":   f"₹{principal:,.0f}",
            "Period":      f"{years} years",
            "Best Return": f"₹{best_val:,.0f}",
        },
    }
    plain = (
        f"Comparing {len(results)} options over {years} years on ₹{principal:,.0f}. "
        f"Winner: {best_name} with maturity ₹{best_val:,.0f} (after {INFLATION}% inflation adjustment)."
    )
    return _chart_block(chart) + plain


# ---------------------------------------------------------------------------
# TOOL 6 — anomaly_detector
# ---------------------------------------------------------------------------

@tool
def anomaly_detector(customer_id: str) -> str:
    """Detect unusual spending patterns and anomalies in transactions.
    Use when asked: unusual spend, suspicious, where did money go,
    spending alerts, anything strange, anomalies."""
    rows = _get_customer_data(customer_id, days=60)
    if not rows:
        return f"No transaction data found for {customer_id}."

    monthly_income = 80000
    total_spend = sum(r["amount"] for r in rows)
    avg_daily = total_spend / 60

    # Per-category monthly averages (based on first 30 days)
    today = date.today()
    cat_month: dict[str, float] = {}
    cat_recent: dict[str, float] = {}
    merchant_first: dict[str, str] = {}
    merchant_total: dict[str, float] = {}

    by_day: dict[str, float] = {}

    for r in rows:
        tx_date = date.fromisoformat(r["date"])
        days_ago = (today - tx_date).days
        cat = r["category"]
        merchant = r["merchant"]
        amt = r["amount"]

        by_day[r["date"]] = by_day.get(r["date"], 0) + amt

        if days_ago > 30:
            cat_month[cat] = cat_month.get(cat, 0) + amt
        else:
            cat_recent[cat] = cat_recent.get(cat, 0) + amt

        if merchant not in merchant_first:
            merchant_first[merchant] = r["date"]
        merchant_total[merchant] = merchant_total.get(merchant, 0) + amt

    anomalies: list[str] = []
    anomaly_objects: list[dict] = []

    # 1. Category spike: recent 30d > 2x prior 30d for same category
    for cat in cat_recent:
        prev = cat_month.get(cat, 0)
        rec  = cat_recent[cat]
        if prev > 0 and rec > 2 * prev:
            anomalies.append(
                f"📈 Category spike: {cat} — ₹{rec:,.0f} (recent 30d) vs ₹{prev:,.0f} (prior 30d) — {rec/prev:.1f}x normal"
            )
            anomaly_objects.append({"type": "Category Spike", "category": cat, "amount": round(rec),
                                    "detail": f"{rec/prev:.1f}x normal vs prior 30d"})

    # 2. Large single transaction: any row > 20% monthly income
    threshold_single = monthly_income * 0.20
    for r in rows:
        if r["amount"] > threshold_single:
            anomalies.append(
                f"💸 Large transaction: ₹{r['amount']:,.0f} at {r['merchant']} on {r['date']} "
                f"({r['amount']/monthly_income*100:.0f}% of monthly income)"
            )
            anomaly_objects.append({"type": "Large Transaction", "category": r["category"],
                                    "amount": round(r["amount"]),
                                    "detail": f"at {r['merchant']} on {r['date']}"})

    # 3. High velocity day: daily total > 3x avg daily
    for day_str, day_total in by_day.items():
        if day_total > 3 * avg_daily:
            anomalies.append(
                f"⚡ High-spend day: {day_str} — ₹{day_total:,.0f} ({day_total/avg_daily:.1f}x daily average)"
            )
            anomaly_objects.append({"type": "High-Spend Day", "category": "All",
                                    "amount": round(day_total),
                                    "detail": f"{day_str} — {day_total/avg_daily:.1f}x daily avg"})

    # 4. New merchant with large amount (only appears in last 30 days, total > ₹1000)
    all_rows_60 = _get_customer_data(customer_id, days=60)
    merchant_early: set[str] = set()
    for r in all_rows_60:
        tx_date = date.fromisoformat(r["date"])
        if (today - tx_date).days > 30:
            merchant_early.add(r["merchant"])

    for merchant, total in merchant_total.items():
        if merchant not in merchant_early and total > 1000:
            anomalies.append(
                f"🆕 New merchant: {merchant} — ₹{total:,.0f} spent, first seen {merchant_first[merchant]}"
            )
            anomaly_objects.append({"type": "New Merchant", "category": "Unknown",
                                    "amount": round(total),
                                    "detail": f"{merchant}, first seen {merchant_first[merchant]}"})

    name = _customer_name(customer_id)
    chart = {
        "type": "anomaly",
        "title": "Spending Anomaly Report",
        "count": len(anomalies),
        "anomalies": anomaly_objects,
        "metrics": {
            "Anomalies Found": str(len(anomalies)) if anomalies else "0",
            "Period":          "Last 60 days",
            "Customer":        name,
        },
    }

    if not anomalies:
        return _chart_block(chart) + f"No unusual spending detected for {name}. All patterns normal."

    plain = (
        f"{len(anomalies)} anomaly/anomalies detected for {name}. "
        f"Types: {', '.join(set(a['type'] for a in anomaly_objects))}. "
        f"Review the chart above for details."
    )
    return _chart_block(chart) + plain


# ---------------------------------------------------------------------------
# TOOLS 7–13 — Financial Math
# ---------------------------------------------------------------------------

@tool
def cagr(start_value: float, end_value: float, years: float) -> str:
    """Calculate Compound Annual Growth Rate (CAGR).
    Use for: CAGR, annual growth rate, portfolio growth rate."""
    if start_value <= 0 or years <= 0:
        return "Error: start_value and years must be positive."
    rate = (end_value / start_value) ** (1 / years) - 1
    gain = end_value - start_value
    chart = {"type": "calculation", "title": "CAGR Calculation",
             "result_label": "CAGR", "result_value": f"{rate*100:.2f}% p.a.",
             "metrics": [
                 {"label": "Start Value",      "value": f"₹{start_value:,.0f}"},
                 {"label": "End Value",         "value": f"₹{end_value:,.0f}"},
                 {"label": "Duration",          "value": f"{years:.1f} years"},
                 {"label": "Total Gain",        "value": f"₹{gain:,.0f}"},
                 {"label": "Absolute Return",   "value": f"{gain/start_value*100:.1f}%"},
             ]}
    return _chart_block(chart) + f"CAGR is {rate*100:.2f}% p.a. — ₹{start_value:,.0f} grew to ₹{end_value:,.0f} over {years:.1f} years (total gain ₹{gain:,.0f})."


@tool
def roi(gain: float, cost: float) -> str:
    """Calculate Return on Investment (ROI).
    Use for: ROI, return on investment, profit percentage."""
    if cost == 0:
        return "Error: cost cannot be zero."
    pct = (gain - cost) / cost * 100
    chart = {"type": "calculation", "title": "ROI Calculation",
             "result_label": "ROI", "result_value": f"{pct:.2f}%",
             "metrics": [
                 {"label": "Invested",   "value": f"₹{cost:,.0f}"},
                 {"label": "Returned",   "value": f"₹{gain:,.0f}"},
                 {"label": "Profit",     "value": f"₹{gain-cost:,.0f}"},
                 {"label": "Per ₹100",   "value": f"₹{pct:.2f} gained"},
             ]}
    return _chart_block(chart) + f"ROI is {pct:.2f}%. Invested ₹{cost:,.0f}, gained ₹{gain:,.0f}, profit ₹{gain-cost:,.0f}."


@tool
def compound_interest(principal: float, rate_percent: float, years: float, n: int = 12) -> str:
    """Calculate compound interest.
    principal=initial amount, rate_percent=annual rate, years=duration, n=compounds per year (default 12=monthly).
    Use for: compound interest, how much will X grow, future value."""
    r = rate_percent / 100
    total = principal * (1 + r / n) ** (n * years)
    interest = total - principal
    chart = {"type": "calculation", "title": "Compound Interest",
             "result_label": "Total Amount", "result_value": f"₹{total:,.0f}",
             "metrics": [
                 {"label": "Principal",        "value": f"₹{principal:,.0f}"},
                 {"label": "Rate",             "value": f"{rate_percent}% p.a. ({n}x/yr)"},
                 {"label": "Duration",         "value": f"{years} years"},
                 {"label": "Interest Earned",  "value": f"₹{interest:,.0f}"},
                 {"label": "Effective Return", "value": f"{interest/principal*100:.1f}%"},
             ]}
    return _chart_block(chart) + f"₹{principal:,.0f} at {rate_percent}% for {years} years grows to ₹{total:,.0f} (interest earned ₹{interest:,.0f}, return {interest/principal*100:.1f}%)."


@tool
def emi_calculator(principal: float, rate_percent: float, years: float) -> str:
    """Calculate EMI (Equated Monthly Instalment) for a loan.
    Use for: EMI, loan payment, home loan, car loan, how much per month."""
    r = rate_percent / 100 / 12
    n = years * 12
    if r == 0:
        emi = principal / n
    else:
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_payment = emi * n
    total_interest = total_payment - principal
    chart = {"type": "calculation", "title": "EMI Calculator",
             "result_label": "Monthly EMI", "result_value": f"₹{emi:,.0f}",
             "metrics": [
                 {"label": "Principal",       "value": f"₹{principal:,.0f}"},
                 {"label": "Rate",            "value": f"{rate_percent}% p.a."},
                 {"label": "Tenure",          "value": f"{years:.0f} years ({n:.0f} months)"},
                 {"label": "Total Payment",   "value": f"₹{total_payment:,.0f}"},
                 {"label": "Total Interest",  "value": f"₹{total_interest:,.0f}"},
             ]}
    return _chart_block(chart) + f"EMI for ₹{principal:,.0f} at {rate_percent}% over {years:.0f} years is ₹{emi:,.0f}/month. Total payment ₹{total_payment:,.0f} (interest ₹{total_interest:,.0f})."


@tool
def savings_rate_calc(income: float, expenses: float) -> str:
    """Calculate savings rate given income and expenses.
    Use for: savings rate, how much am I saving, savings percentage."""
    if income <= 0:
        return "Error: income must be positive."
    savings = income - expenses
    rate = savings / income * 100
    annual_savings = savings * 12
    chart = {"type": "calculation", "title": "Savings Rate",
             "result_label": "Savings Rate", "result_value": f"{rate:.1f}%",
             "metrics": [
                 {"label": "Monthly Income",   "value": f"₹{income:,.0f}"},
                 {"label": "Monthly Expenses", "value": f"₹{expenses:,.0f}"},
                 {"label": "Monthly Savings",  "value": f"₹{savings:,.0f}"},
                 {"label": "Annual Savings",   "value": f"₹{annual_savings:,.0f}"},
             ]}
    return _chart_block(chart) + f"Savings rate is {rate:.1f}% — saving ₹{savings:,.0f}/month (₹{annual_savings:,.0f}/year). {'Excellent!' if rate >= 30 else 'Good.' if rate >= 20 else 'Below recommended 20%.'}"


@tool
def percentage_change(old_value: float, new_value: float) -> str:
    """Calculate percentage change between two values.
    Use for: percentage change, increase, decrease, how much did X change."""
    if old_value == 0:
        return "Error: old_value cannot be zero."
    change = (new_value - old_value) / old_value * 100
    direction = "↑ increased" if change > 0 else ("↓ decreased" if change < 0 else "→ unchanged")
    chart = {"type": "calculation", "title": "Percentage Change",
             "result_label": "Change", "result_value": f"{change:+.2f}%",
             "metrics": [
                 {"label": "Old Value",  "value": f"₹{old_value:,.2f}"},
                 {"label": "New Value",  "value": f"₹{new_value:,.2f}"},
                 {"label": "Difference", "value": f"₹{abs(new_value - old_value):,.2f}"},
             ]}
    return _chart_block(chart) + f"Value {direction} by {change:+.2f}% — from ₹{old_value:,.2f} to ₹{new_value:,.2f} (₹{abs(new_value-old_value):,.2f} difference)."


@tool
def gst_calculator(amount: float, gst_rate: float = 18.0) -> str:
    """Calculate GST (Goods and Services Tax) on a base amount.
    amount = base price before tax. gst_rate = percentage (default 18).
    Use for: GST, goods and services tax, 18% GST on X, tax on amount."""
    gst   = round(amount * gst_rate / 100, 2)
    total = round(amount + gst, 2)
    chart = {
        "type":         "calculation",
        "title":        f"GST Calculator ({gst_rate}%)",
        "result_label": "Total Amount",
        "result_value": f"₹{total:,.2f}",
        "metrics": [
            {"label": "Base Amount",    "value": f"₹{amount:,.2f}"},
            {"label": f"GST ({gst_rate}%)", "value": f"₹{gst:,.2f}"},
            {"label": "CGST",           "value": f"₹{gst/2:,.2f}"},
            {"label": "SGST",           "value": f"₹{gst/2:,.2f}"},
            {"label": "Total",          "value": f"₹{total:,.2f}"},
        ],
    }
    return (
        _chart_block(chart)
        + f"GST @ {gst_rate}% on ₹{amount:,.2f}: "
        f"GST = ₹{gst:,.2f}, Total = ₹{total:,.2f} "
        f"(CGST ₹{gst/2:,.2f} + SGST ₹{gst/2:,.2f})."
    )


@tool
def profit_margin(revenue: float, cost: float) -> str:
    """Calculate gross profit margin given revenue and total cost.
    Use for: profit margin, gross margin, profitability, margin percentage."""
    if revenue <= 0:
        return "Error: revenue must be positive."
    profit = revenue - cost
    margin = profit / revenue * 100
    chart = {
        "type":         "calculation",
        "title":        "Profit Margin",
        "result_label": "Profit Margin",
        "result_value": f"{margin:.2f}%",
        "metrics": [
            {"label": "Revenue",        "value": f"₹{revenue:,.2f}"},
            {"label": "Cost",           "value": f"₹{cost:,.2f}"},
            {"label": "Gross Profit",   "value": f"₹{profit:,.2f}"},
            {"label": "Profit Margin",  "value": f"{margin:.2f}%"},
        ],
    }
    return (
        _chart_block(chart)
        + f"Profit Margin: {margin:.2f}%. "
        f"Revenue ₹{revenue:,.2f}, Cost ₹{cost:,.2f}, Profit ₹{profit:,.2f}."
    )


@tool
def tax_estimator(annual_income: float) -> str:
    """Estimate Indian income tax under the new regime (FY 2024-25).
    Use for: tax, how much tax, income tax, take-home salary, tax calculation."""
    # New regime slabs (FY 2024-25)
    new_slabs = [
        (300000,  0.00),
        (400000,  0.05),  # 3L–7L at 5%
        (300000,  0.10),  # 7L–10L at 10%
        (200000,  0.15),  # 10L–12L at 15%
        (300000,  0.20),  # 12L–15L at 20%
        (float("inf"), 0.30),  # >15L at 30%
    ]
    # Old regime slabs
    old_slabs = [
        (250000,  0.00),
        (250000,  0.05),  # 2.5L–5L
        (500000,  0.20),  # 5L–10L
        (float("inf"), 0.30),  # >10L
    ]

    def calc_tax(income: float, slabs: list) -> float:
        tax = 0.0
        remaining = income
        for limit, rate in slabs:
            taxable = min(remaining, limit)
            tax += taxable * rate
            remaining -= taxable
            if remaining <= 0:
                break
        return tax

    new_tax = calc_tax(annual_income, new_slabs)
    # Standard deduction ₹75k in new regime (2024-25)
    new_tax_after_sd = calc_tax(max(0, annual_income - 75000), new_slabs)

    # Old regime: 80C deduction ₹1.5L assumed
    old_tax = calc_tax(max(0, annual_income - 150000 - 50000), old_slabs)  # 80C + std deduction

    eff_new = new_tax_after_sd / annual_income * 100
    eff_old = old_tax / annual_income * 100
    monthly_takehome = (annual_income - new_tax_after_sd) / 12
    better = "New Regime" if new_tax_after_sd <= old_tax else "Old Regime"

    chart = {"type": "calculation", "title": "Tax Estimator FY 2024-25",
             "result_label": "Monthly Take-Home", "result_value": f"₹{monthly_takehome:,.0f}",
             "metrics": [
                 {"label": "Annual Income",    "value": f"₹{annual_income:,.0f}"},
                 {"label": "Tax (New Regime)", "value": f"₹{new_tax_after_sd:,.0f} ({eff_new:.1f}%)"},
                 {"label": "Tax (Old Regime)", "value": f"₹{old_tax:,.0f} ({eff_old:.1f}%)"},
                 {"label": "Tax Savings",      "value": f"₹{abs(new_tax_after_sd-old_tax):,.0f} with {better}"},
             ]}
    return (
        _chart_block(chart) +
        f"On ₹{annual_income:,.0f} annual income, tax is ₹{new_tax_after_sd:,.0f} ({eff_new:.1f}%) under new regime. "
        f"Monthly take-home ₹{monthly_takehome:,.0f}. Recommended: {better} (saves ₹{abs(new_tax_after_sd-old_tax):,.0f})."
    )


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Personal Financial Advisor with access to real customer transaction \
data and advanced financial tools.

For complex analysis: ALWAYS call query_customer_transactions first to get data, \
then chain analysis tools for deeper insights.

Be specific with numbers. Use ₹ for Indian Rupees. Format answers with clear sections.
When a customer ID (CUST001, CUST002) is mentioned, use it in tool calls.
You are the most sophisticated agent in this system — provide expert, data-driven advice.
"""

_ALL_TOOLS = [
    query_customer_transactions,
    financial_health_score,
    spend_forecast,
    goal_planner,
    investment_comparison,
    anomaly_detector,
    cagr,
    roi,
    compound_interest,
    emi_calculator,
    savings_rate_calc,
    percentage_change,
    tax_estimator,
    gst_calculator,
    profit_margin,
]


# ---------------------------------------------------------------------------
# Public interface — same signature as original so graph.py is unchanged
# ---------------------------------------------------------------------------

async def run(question: str) -> str:
    """
    Answer a financial advisory question using the ReAct agent.

    Args:
        question: User question, possibly prefixed with conversation history
                  or knowledge context from finance_agent.py.

    Returns:
        Final natural-language answer as a plain string.
    """
    import logging
    _log = logging.getLogger(__name__)
    llm   = ChatGroq(model=GROQ_MODEL, temperature=0, max_retries=0)
    agent = create_react_agent(llm, _ALL_TOOLS, state_modifier=_SYSTEM_PROMPT)
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        return result["messages"][-1].content
    except Exception as exc:
        _log.error("tool_agent ReAct error: %s", exc, exc_info=True)
        return (
            "I couldn't complete the requested analysis. "
            "Please verify the inputs and try again."
        )
