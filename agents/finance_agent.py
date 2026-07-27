"""
finance_agent.py — TechMart India Finance Agent

10 tools across 3 categories. Python/ML do ALL arithmetic — the LLM only
picks which tool to call and extracts parameter values.

Category A — Company Math (deterministic Python):
  1. calculate_selling_price    — base/margin/GST breakdown
  2. calculate_bulk_quote       — unit price + bulk discount + shipping
  3. calculate_loyalty_price    — tier-based loyalty discount
  4. calculate_profit_margin    — revenue/cost/GST/profit per unit
  5. generate_invoice           — formatted invoice with all line items

Category B — ML Predictions:
  6. predict_demand             — LinearRegression demand vs stock check

Category C — Analytics (SQL + Python):
  7. customer_lifetime_value    — CLV from transaction history
  8. category_performance       — revenue/growth/returns per category
  9. monthly_trend_analysis     — full revenue trend, best/worst, MoM growth

_direct_calc is retained for generic math (GST%, CAGR, ROI) that do not
require a DB lookup — these bypass create_react_agent entirely.

Revenue forecasting (ARIMA) and churn-risk prediction (RandomForest) used to
live here but are now owned by dedicated agents (agents/forecast_agent.py,
agents/risk_agent.py) — see Phase 2 of the multi-agent expansion. The
underlying tool functions (forecast_revenue, predict_customer_risk,
compare_customer_risk) still physically live in this module and are
imported/reused by those agents rather than duplicated, but this module's
own _fast_dispatch no longer routes to them — that routing now belongs to
the Forecast/Risk agents' own dispatchers.
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
from datetime import date

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_BASE)
DB_PATH  = os.path.join(_ROOT, "data",   "company.db")
CFG_PATH = os.path.join(_ROOT, "config", "company_rates.json")
MDL_DIR  = os.path.join(_ROOT, "models")
_MODEL   = os.getenv("FINANCE_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

# ── Module-level: load config and ML models once ──────────────────────────────
import sys as _sys
_sys.path.insert(0, _ROOT)
from models import ForecastWrapper   # noqa: F401 — required for joblib.load

import joblib

COMPANY = json.load(open(CFG_PATH))

def _safe_load(filename):
    try:
        return joblib.load(os.path.join(MDL_DIR, filename))
    except Exception:
        return None

def _safe_json(filename):
    try:
        return json.load(open(os.path.join(MDL_DIR, filename)))
    except Exception:
        return {}

FORECAST_MODEL   = _safe_load("sales_forecast.pkl")
FORECAST_METRICS = _safe_json("forecast_metrics.json")
CHURN_MODEL      = _safe_load("churn_classifier.pkl")
CHURN_ENCODER    = _safe_load("churn_label_encoder.pkl")
CHURN_FEATURES   = _safe_load("churn_feature_columns.pkl")
CHURN_METRICS    = _safe_json("churn_metrics.json")
DEMAND_MODELS    = _safe_load("demand_models.pkl")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _query_db(sql: str, params: tuple = ()) -> list[dict]:
    """Execute SQL against company.db and return list of row dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
    return rows


def _chart(data: dict) -> str:
    return f"<CHART_DATA>\n{json.dumps(data, ensure_ascii=False)}\n</CHART_DATA>\n\n"


def _bullet_summary(card: dict) -> str:
    """
    Plain-text bullet recap of a card's own data, appended after the card +
    table in every multi-item tool. Cards are for visual scanning; this is
    for users who also want a quick plain-text answer they can read without
    the widget — deterministic from the same data already in the card, so
    it can't drift from or contradict the numbers shown above it.
    """
    lines = ["**Summary:**"]
    metrics = card.get("metrics")
    if isinstance(metrics, list):
        for m in metrics:
            if isinstance(m, dict) and "label" in m and "value" in m:
                lines.append(f"- {m['label']}: {m['value']}")
    elif isinstance(metrics, dict):
        for k, v in metrics.items():
            lines.append(f"- {k.replace('_', ' ').title()}: {v}")
    if card.get("result_label") and card.get("result_value"):
        lines.append(f"- **{card['result_label']}: {card['result_value']}**")
    return "\n".join(lines)


def _num(s: str) -> float:
    return float(re.sub(r"[₹₨,\s]", "", s))


# Confirmed live (see project chat history): when NO real tool genuinely
# fits a question, the ReAct model sometimes reaches for the
# closest-sounding tool anyway, with a made-up argument that doesn't match
# its actual schema (e.g. calling prioritize_collections with a
# "customer_id" it doesn't even accept), and emits this as a raw pseudo
# tool-call STRING in the message content rather than a real tool_calls
# entry -- confirmed via tool_calls=[] on that exact AIMessage. Since no
# ToolMessage exists for a call that was never actually registered/
# executed, it would otherwise flow straight through as the "final
# answer" and leak this garbage directly to the user.
_LEAKED_TOOL_CALL_RE = re.compile(r'^\s*<(\w+)>\s*\{.*?\}\s*</\1>\s*$', re.DOTALL)


def _looks_like_leaked_tool_call(text: str) -> bool:
    """True if `text` is a hallucinated pseudo tool-call that was never
    actually executed, not a genuine final answer."""
    return bool(_LEAKED_TOOL_CALL_RE.match((text or "").strip()))


# Standard cost-of-capital assumption for an Indian retail business -- not
# fit to this data, just a reasonable annual discount rate for NPV math.
_CLV_DISCOUNT_RATE = 0.10


def _npv(monthly_cashflow: float, months: int, annual_rate: float) -> float:
    """
    Present value of a constant monthly cash flow received for `months`
    months, discounted at `annual_rate` (annual, converted to a compounding
    monthly rate). Used by customer_lifetime_value to turn "aov * frequency
    * N months" from a spreadsheet sum into a genuine NPV figure -- money
    expected 3 years from now is worth less today than money expected next
    month.
    """
    monthly_rate = (1 + annual_rate) ** (1 / 12) - 1
    if monthly_rate == 0:
        return monthly_cashflow * months
    return sum(monthly_cashflow / (1 + monthly_rate) ** t for t in range(1, months + 1))


def _normalize_id(raw_id: str, prefix: str) -> str:
    """
    Zero-pad a product/customer id to the DB's PRD001/CUST001 format.
    Every id ultimately reaches _get_product/_get_customer from three very
    different sources — a fast-dispatch regex match, a ReAct tool-call
    argument the LLM typed out, or the multi-product resolver — and none of
    them can be trusted to zero-pad consistently. "prd1", "PRD1", and
    "PRD001" must all resolve to the same row, or a perfectly-routed
    question (finance correctly picked) still answers "not found" because
    the lookup string didn't match the DB's fixed-width id.
    """
    m = re.match(rf'{prefix}0*(\d+)$', raw_id.strip().upper())
    if m:
        return f"{prefix}{int(m.group(1)):03d}"
    return raw_id.strip().upper()


def _get_product(product_id: str) -> dict | None:
    """Fetch product row from DB using absolute DB_PATH."""
    pid = _normalize_id(product_id, "PRD")
    print(f"[finance] _get_product: DB_PATH={DB_PATH}")
    print(f"[finance] _get_product: looking for {pid}")
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT product_id, product_name, category, base_cost, "
            "margin_pct, tax_pct, selling_price, mrp, stock_quantity "
            "FROM products WHERE product_id = ?",
            (pid,)
        ).fetchone()
        conn.close()
        print(f"[finance] _get_product: result={row}")
        if not row:
            return None
        return dict(zip(
            ["product_id","product_name","category","base_cost",
             "margin_pct","tax_pct","selling_price","mrp","stock_quantity"], row
        ))
    except Exception as e:
        logger.error("[finance] DB error fetching %s: %s", product_id, e)
        return None


def _get_customer(customer_id: str) -> dict | None:
    """Fetch customer row from DB using absolute DB_PATH."""
    cid = _normalize_id(customer_id, "CUST")
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT customer_id, customer_name, tier, total_spend, "
            "total_orders, days_inactive "
            "FROM customers WHERE customer_id = ?",
            (cid,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        return dict(zip(
            ["customer_id","customer_name","tier","total_spend",
             "total_orders","days_inactive"], row
        ))
    except Exception as e:
        logger.error("[finance] DB error fetching %s: %s", customer_id, e)
        return None


# ── CATEGORY A — Company Math ─────────────────────────────────────────────────

@tool
def calculate_selling_price(product_id: str) -> str:
    """Calculate selling price using internal company rates.
    Use when asked: selling price, price of PRD, cost of product, how much is PRD."""
    prod = _get_product(product_id)
    if not prod:
        return f"Product {product_id} was not found in the catalogue."

    base       = prod["base_cost"]
    margin     = prod["margin_pct"]
    gst        = prod["tax_pct"]
    subtotal   = round(base * (1 + margin / 100), 2)
    sell       = round(subtotal * (1 + gst / 100), 2)
    margin_amt = round(base * margin / 100, 2)
    gst_amt    = round(subtotal * gst / 100, 2)

    card = {
        "type":         "calculation",
        "title":        f"Selling Price — {prod['product_name']}",
        "result_value": f"₹{sell:,.2f}",
        "result_label": "Selling Price",
        "metrics": [
            {"label": "Base Cost",             "value": f"₹{base:,.2f}"},
            {"label": f"Margin ({margin}%)",   "value": f"₹{margin_amt:,.2f}"},
            {"label": "Subtotal",              "value": f"₹{subtotal:,.2f}"},
            {"label": f"GST ({gst}%)",         "value": f"₹{gst_amt:,.2f}"},
        ],
    }
    return (
        _chart(card)
        + f"**Selling Price — {product_id} ({prod['product_name']})**\n\n"
        f"- Base Cost: ₹{base:,.2f}\n"
        f"- Margin ({margin}%): ₹{margin_amt:,.2f}\n"
        f"- Subtotal: ₹{subtotal:,.2f}\n"
        f"- GST ({gst}%): ₹{gst_amt:,.2f}\n"
        f"- **Selling Price: ₹{sell:,.2f}**\n\n"
        f"Category: {prod['category']} | GST Rate: {gst}% | Margin: {margin}%"
        + "\n\n" + _bullet_summary(card)
    )


@tool
def calculate_bulk_quote(product_id: str, quantity: int) -> str:
    """Calculate a bulk order quote. Applies 8% bulk discount if order total > ₹50,000.
    Use when asked: 'bulk order', 'quote for N units', 'wholesale price', 'order quantity'."""
    prod = _get_product(product_id)
    if not prod:
        return f"Product {product_id} not found."

    unit_price = prod["selling_price"]
    subtotal   = round(unit_price * quantity, 2)

    bulk_threshold = COMPANY["discounts"]["bulk_threshold"]
    bulk_pct       = COMPANY["discounts"]["bulk_discount_pct"]
    bulk_eligible  = subtotal > bulk_threshold
    discount_amt   = round(subtotal * bulk_pct / 100, 2) if bulk_eligible else 0.0
    after_discount = round(subtotal - discount_amt, 2)

    shipping_free_above = COMPANY["shipping"]["free_above"]
    shipping_flat       = COMPANY["shipping"]["flat_rate"]
    shipping            = 0.0 if after_discount > shipping_free_above else shipping_flat
    grand_total         = round(after_discount + shipping, 2)

    discount_note = (
        f"**Bulk discount applied: {bulk_pct}% off** (order value ₹{subtotal:,.2f} > ₹{bulk_threshold:,})"
        if bulk_eligible else
        f"No bulk discount (order value ₹{subtotal:,.2f} < threshold ₹{bulk_threshold:,})"
    )
    card = {
        "type":         "calculation",
        "title":        f"Bulk Quote — {prod['product_name']} × {quantity}",
        "result_value": f"₹{grand_total:,.2f}",
        "result_label": "Grand Total",
        "metrics": [
            {"label": "Unit Price",                       "value": f"₹{unit_price:,.2f}"},
            {"label": f"Subtotal ({quantity} units)",      "value": f"₹{subtotal:,.2f}"},
            {"label": "Bulk Discount",                     "value": f"₹{discount_amt:,.2f}"},
            {"label": "Shipping",                          "value": f"₹{shipping:,.2f}"},
        ],
    }
    return (
        _chart(card)
        + f"**Bulk Quote — {prod['product_name']} × {quantity} units**\n\n"
        f"- Unit Price: ₹{unit_price:,.2f}\n"
        f"- Subtotal ({quantity} units): ₹{subtotal:,.2f}\n"
        f"- Bulk Discount: ₹{discount_amt:,.2f}\n"
        f"- After Discount: ₹{after_discount:,.2f}\n"
        f"- Shipping: ₹{shipping:,.2f}\n"
        f"- **Grand Total: ₹{grand_total:,.2f}**\n\n"
        + discount_note
        + "\n\n" + _bullet_summary(card)
    )


@tool
def calculate_loyalty_price(customer_id: str, product_id: str | None = None) -> str:
    """Calculate a customer's loyalty tier discount, and the discounted price if applied
    to a specific product. Use ONLY for: discount, loyalty price, tier discount, 'what
    discount/price does CUST get'. NOT for churn, risk, retention, or 'will they leave'
    questions — use predict_customer_risk for those instead. product_id is optional —
    omit it to answer a plain discount/tier question with no specific product."""
    cust = _get_customer(customer_id.upper())
    if not cust:
        return f"Customer {customer_id} not found."

    tier        = cust["tier"]
    loyalty_pct = COMPANY["discounts"]["loyalty"].get(tier, 0)

    # No product named — plain "what's my tier/discount" answer, not a price
    # breakdown for a product nobody asked about.
    if not product_id:
        card = {
            "type":         "calculation",
            "title":        f"Loyalty Tier — {cust['customer_name']}",
            "result_value": f"{loyalty_pct}%",
            "result_label": "Loyalty Discount",
            "metrics": [{"label": "Tier", "value": tier}],
        }
        return (
            _chart(card)
            + f"**{cust['customer_name']}** ({customer_id.upper()}) is a **{tier} tier** customer.\n\n"
            f"- **Loyalty Discount: {loyalty_pct}%**\n\n"
            f"Loyalty tiers: Bronze 0% | Silver 5% | Gold 8% | Platinum 12%"
            + "\n\n" + _bullet_summary(card)
        )

    prod = _get_product(product_id.upper())
    if not prod:
        return f"Product {product_id} not found."

    unit_price    = prod["selling_price"]
    discount_amt  = round(unit_price * loyalty_pct / 100, 2)
    loyalty_price = round(unit_price - discount_amt, 2)

    shipping_free = COMPANY["shipping"]["free_above"]
    shipping      = 0.0 if loyalty_price > shipping_free else COMPANY["shipping"]["flat_rate"]
    total         = round(loyalty_price + shipping, 2)

    card = {
        "type":         "calculation",
        "title":        f"Loyalty Price — {prod['product_name']} ({cust['customer_name']})",
        "result_value": f"₹{total:,.2f}",
        "result_label": "Total",
        "metrics": [
            {"label": "Standard Price",                    "value": f"₹{unit_price:,.2f}"},
            {"label": f"Loyalty Discount ({loyalty_pct}%)", "value": f"−₹{discount_amt:,.2f}"},
            {"label": "Loyalty Price",                      "value": f"₹{loyalty_price:,.2f}"},
            {"label": "Shipping",                           "value": f"₹{shipping:,.2f}"},
        ],
    }
    return (
        _chart(card)
        + f"**{cust['customer_name']}** ({customer_id.upper()}) is a **{tier} tier** customer.\n\n"
        f"- Product: {prod['product_name']}\n"
        f"- Standard Price: ₹{unit_price:,.2f}\n"
        f"- Loyalty Discount ({loyalty_pct}%): −₹{discount_amt:,.2f}\n"
        f"- **Loyalty Price: ₹{loyalty_price:,.2f}**\n"
        f"- Shipping: ₹{shipping:,.2f}\n"
        f"- **Total: ₹{total:,.2f}**\n\n"
        f"Loyalty tiers: Bronze 0% | Silver 5% | Gold 8% | Platinum 12%"
        + "\n\n" + _bullet_summary(card)
    )


@tool
def calculate_profit_margin(product_id: str) -> str:
    """Calculate profit margin for a product.
    Use when asked: profit margin, how much we make, gross margin, markup on PRD."""
    prod = _get_product(product_id)
    if not prod:
        return f"Product {product_id} not found."

    base       = prod["base_cost"]
    margin_pct = prod["margin_pct"]
    sell       = prod["selling_price"]
    gst        = prod["tax_pct"]
    margin_amt = round(base * margin_pct / 100, 2)
    gst_amt    = round(sell * gst / (100 + gst), 2)

    card = {
        "type":         "calculation",
        "title":        f"Profit Margin — {prod['product_name']}",
        "result_value": f"{margin_pct}%",
        "result_label": "Margin",
        "metrics": [
            {"label": "Selling Price", "value": f"₹{sell:,.2f}"},
            {"label": "Base Cost",     "value": f"₹{base:,.2f}"},
            {"label": "Gross Profit",  "value": f"₹{margin_amt:,.2f}"},
            {"label": f"GST to Govt ({gst}%)", "value": f"₹{gst_amt:,.2f}"},
        ],
    }
    return (
        _chart(card)
        + f"**Profit Margin — {product_id} ({prod['product_name']})**\n\n"
        f"- Selling Price: ₹{sell:,.2f}\n"
        f"- Base Cost: ₹{base:,.2f}\n"
        f"- Gross Profit: ₹{margin_amt:,.2f}\n"
        f"- GST to Govt ({gst}%): ₹{gst_amt:,.2f}\n"
        f"- Net Profit/unit: ₹{margin_amt:,.2f}\n"
        f"- **Margin: {margin_pct}%**"
        + "\n\n" + _bullet_summary(card)
    )


@tool
def generate_invoice(customer_id: str, product_id: str, quantity: int) -> str:
    """Generate a formatted invoice for a customer order. Applies loyalty + bulk discounts.
    Use when asked: 'invoice', 'bill for', 'generate order', 'purchase receipt'."""
    product_id = _normalize_id(product_id, "PRD")
    customer_id = _normalize_id(customer_id, "CUST")
    prod_rows = _query_db("SELECT * FROM products WHERE product_id = ?", (product_id,))
    cust_rows = _query_db("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    if not prod_rows:
        return f"Product {product_id} not found."
    if not cust_rows:
        return f"Customer {customer_id} not found."

    p         = prod_rows[0]
    c         = cust_rows[0]
    tier      = c["tier"]
    unit_price = p["selling_price"]
    subtotal   = round(unit_price * quantity, 2)

    # Apply loyalty discount first, then bulk if still eligible
    loyalty_pct   = COMPANY["discounts"]["loyalty"].get(tier, 0)
    loyalty_disc  = round(subtotal * loyalty_pct / 100, 2)
    after_loyalty = round(subtotal - loyalty_disc, 2)

    bulk_threshold = COMPANY["discounts"]["bulk_threshold"]
    bulk_pct       = COMPANY["discounts"]["bulk_discount_pct"]
    bulk_eligible  = after_loyalty > bulk_threshold
    bulk_disc      = round(after_loyalty * bulk_pct / 100, 2) if bulk_eligible else 0.0
    after_bulk     = round(after_loyalty - bulk_disc, 2)

    shipping = 0.0 if after_bulk > COMPANY["shipping"]["free_above"] else COMPANY["shipping"]["flat_rate"]
    total    = round(after_bulk + shipping, 2)

    inv_num = f"INV-TM-{date.today().strftime('%Y%m%d')}-{customer_id[-4:]}{product_id[-3:]}"
    today   = date.today().isoformat()

    # The invoice's own table format is kept (a line-itemed table is how a
    # real invoice conventionally reads) alongside the card+bullets.
    card = {
        "type":         "calculation",
        "title":        f"Invoice {inv_num}",
        "result_value": f"₹{total:,.2f}",
        "result_label": "TOTAL",
        "metrics": [
            {"label": "Unit Price",                        "value": f"₹{unit_price:,.2f}"},
            {"label": "Subtotal",                           "value": f"₹{subtotal:,.2f}"},
            {"label": f"Loyalty Discount ({loyalty_pct}%)", "value": f"−₹{loyalty_disc:,.2f}"},
            {"label": f"Bulk Discount ({bulk_pct}%)",       "value": f"−₹{bulk_disc:,.2f}"},
            {"label": "Shipping",                           "value": f"₹{shipping:,.2f}"},
        ],
    }
    return (
        _chart(card)
        + f"**TechMart India — Invoice #{inv_num}**\n"
        f"**Date:** {today}\n\n"
        f"| Bill To | {c['customer_name']} ({customer_id}) — {tier} Tier |\n"
        f"|---|---|\n"
        f"| Item | {p['product_name']} ({product_id}) |\n"
        f"| Unit Price | ₹{unit_price:,.2f} |\n"
        f"| Quantity | {quantity} |\n"
        f"| Subtotal | ₹{subtotal:,.2f} |\n"
        f"| Loyalty Discount ({loyalty_pct}%) | −₹{loyalty_disc:,.2f} |\n"
        f"| Bulk Discount ({bulk_pct}%) | −₹{bulk_disc:,.2f} |\n"
        f"| Shipping | ₹{shipping:,.2f} |\n"
        f"| **TOTAL** | **₹{total:,.2f}** |\n\n"
        f"_Return window: {COMPANY['operations']['return_window_days']} days from delivery_"
        + "\n\n" + _bullet_summary(card)
    )


# ── CATEGORY B — ML Predictions ───────────────────────────────────────────────

@tool
def forecast_revenue(months_ahead: int = 3) -> str:
    """Forecast total revenue for the next N months using the trained ARIMA model.
    Use when asked: 'predict revenue', 'forecast', 'next month sales', 'future revenue'."""
    if FORECAST_MODEL is None:
        return "Revenue forecast model not available — run models/train_models.py first."

    preds = list(np.array(FORECAST_MODEL.forecast(months_ahead)).flatten())
    mape  = FORECAST_METRICS.get("mape", "N/A")
    order = tuple(FORECAST_METRICS.get("best_order", []))
    last  = FORECAST_METRICS.get("last_month", "2026-06")

    # Month labels continue from the model's own training cutoff (FORECAST_METRICS
    # last_month), not from today's real-world date — "next 3 months" means the 3
    # months after the data the model was trained on, which may not be "now".
    year, mon = int(last[:4]), int(last[5:])
    future_months = []
    for _ in range(months_ahead):
        mon += 1
        if mon > 12:
            mon = 1
            year += 1
        future_months.append(f"{year}-{mon:02d}")

    ci_lo = [round(p * 0.85, 0) for p in preds]
    ci_hi = [round(p * 1.15, 0) for p in preds]

    card = {
        "type":              "forecast",
        "total_forecast":    round(sum(preds)),
        "projected_savings": 0,
        "income":            round(sum(preds)),
        "chart_data": [
            {"name": m, "forecast": round(p), "trend": "rising", "change": None}
            for m, p in zip(future_months, preds)
        ],
    }

    rows = "\n".join(
        f"| {m} | ₹{p:,.0f} | ₹{lo:,.0f} – ₹{hi:,.0f} |"
        for m, p, lo, hi in zip(future_months, preds, ci_lo, ci_hi)
    )
    # Separate bullet-summary dict, not the chart JSON above -- `card`'s
    # "type": "forecast" shape (chart_data time-series) is what the frontend
    # widget consumes and must stay unchanged; _bullet_summary needs a
    # "metrics" list instead.
    bullet_data = {
        "metrics": [{"label": m, "value": f"₹{p:,.0f}"} for m, p in zip(future_months, preds)],
        "result_label": "Total Forecast",
        "result_value": f"₹{round(sum(preds)):,}",
    }
    return (
        _chart(card)
        + f"**Revenue Forecast — Next {months_ahead} Month(s)**\n\n"
        f"| Month | Forecast | Range (±15%) |\n|---|---|---|\n"
        + rows
        + f"\n\n**Model:** ARIMA{order}  |  **Test MAPE:** {mape}%\n"
        f"> MAPE reflects variance from mixed product prices in training data. "
        f"Forecasts indicate central tendency around ₹{round(sum(preds)/len(preds)):,}/month."
        + "\n\n" + _bullet_summary(bullet_data)
    )


@tool
def predict_customer_risk(customer_id: str) -> str:
    """Predict if a customer will stop buying (churn risk), using the trained RandomForest classifier.
    Use ONLY for: churn, at risk, will they leave, retention, inactive.
    NOT for discount or pricing questions — use calculate_loyalty_price for those instead."""
    if CHURN_MODEL is None:
        return "Churn model not available — run models/train_models.py first."

    customer_id = _normalize_id(customer_id, "CUST")
    cust_rows = _query_db("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    if not cust_rows:
        return f"Customer {customer_id} was not found."
    c = cust_rows[0]

    # Compute the same features used during training
    txns = _query_db("SELECT * FROM transactions WHERE customer_id = ?", (customer_id,))
    completed   = [t for t in txns if t["status"] == "Completed"]
    total_spend = sum(t["final_amount"] for t in completed)
    order_count = len(completed)
    aov         = total_spend / order_count if order_count > 0 else 0.0
    days_inact  = int(c["days_inactive"])
    return_rate = (sum(1 for t in txns if t["status"] == "Returned") / max(len(txns), 1))
    uniq_cats   = len({t["category"] for t in txns})
    uniq_prods  = len({t["product_id"] for t in txns})
    avg_qty     = sum(t["quantity"] for t in txns) / max(len(txns), 1)

    # days_inactive/return_rate are deliberately excluded here -- they're
    # what the training-time label rule thresholds on (see train_models.py),
    # so the model only ever sees behavioral/purchase-pattern features.
    # Both are still shown in the output below, just not fed to the model.
    feature_values = [total_spend, order_count, aov,
                      uniq_cats, uniq_prods, avg_qty]
    X = np.array([feature_values])

    risk_label = CHURN_ENCODER.inverse_transform(CHURN_MODEL.predict(X))[0]
    proba      = CHURN_MODEL.predict_proba(X)[0]
    proba_dict = dict(zip(CHURN_ENCODER.classes_, np.round(proba, 2)))

    top_importances = sorted(
        zip(CHURN_FEATURES, CHURN_MODEL.feature_importances_),
        key=lambda x: -x[1],
    )[:3]

    risk_emoji = {
        "Days Inactive": "🔴 High" if days_inact > 90 else "🟡 Medium" if days_inact > 30 else "🟢 Low",
        "Return Rate":   "🔴 High" if return_rate > 0.2 else "🟢 Low",
        "Orders Made":   "🟢 Active" if order_count >= 20 else "🟡 Light",
        "Lifetime Spend": "🟢 High" if total_spend > 1_000_000 else "🟡 Medium",
    }

    card = {
        "type":         "calculation",
        "title":        f"Churn Risk — {c['customer_name']}",
        "result_value": str(risk_label),
        "result_label": "Risk Level",
        "metrics": [
            {"label": "Days Inactive",   "value": str(days_inact)},
            {"label": "Return Rate",     "value": f"{return_rate*100:.1f}%"},
            {"label": "Orders Made",     "value": str(order_count)},
            {"label": "Lifetime Spend",  "value": f"₹{total_spend:,.0f}"},
        ],
    }
    return (
        _chart(card)
        + f"**Churn Risk — {c['customer_name']}** ({customer_id}, {c['tier']} tier)\n\n"
        f"**Risk Level: {risk_label}**  (Low Risk probability: {proba_dict.get('Low Risk', 0):.0%})\n\n"
        f"- Days Inactive: {days_inact} ({risk_emoji['Days Inactive']})\n"
        f"- Return Rate: {return_rate*100:.1f}% ({risk_emoji['Return Rate']})\n"
        f"- Orders Made: {order_count} ({risk_emoji['Orders Made']})\n"
        f"- Lifetime Spend: ₹{total_spend:,.0f} ({risk_emoji['Lifetime Spend']})\n\n"
        f"Top risk drivers: {', '.join(f[0] for f in top_importances)}"
        + "\n\n" + _bullet_summary(card)
    )


@tool
def predict_demand(product_id: str) -> str:
    """Predict next-month demand for a product and compare to current stock.
    Use when asked: 'demand for PRD', 'restock', 'will we run out', 'inventory check', 'how many will sell'."""
    if DEMAND_MODELS is None:
        return "Demand models not available — run models/train_models.py first."
    pid = _normalize_id(product_id, "PRD")
    if pid not in DEMAND_MODELS:
        return f"No demand model trained for {pid} (insufficient transaction history)."

    d            = DEMAND_MODELS[pid]
    pred         = d["next_month_demand"]
    stock        = d["current_stock"]
    reorder      = d["reorder_needed"]
    reorder_days = COMPANY["operations"]["reorder_lead_days"]

    # Recommended order quantity: cover 2 months of predicted demand
    rec_order = max(0, pred * 2 - stock)

    prod_rows = _query_db("SELECT * FROM products WHERE product_id = ?", (pid,))
    pname = prod_rows[0]["product_name"] if prod_rows else pid

    status_emoji = "🔴 REORDER NEEDED" if reorder else "🟢 Stock Sufficient"
    card = {
        "type":         "calculation",
        "title":        f"Demand Forecast — {pname}",
        "result_value": "Reorder Needed" if reorder else "Stock Sufficient",
        "result_label": "Status",
        "metrics": [
            {"label": "Predicted Next-Month Demand", "value": f"{pred} units"},
            {"label": "Current Stock",                "value": f"{stock} units"},
        ],
    }
    return (
        _chart(card)
        + f"**Demand Forecast — {pname}** ({pid})\n\n"
        f"- Predicted Next-Month Demand: {pred} units\n"
        f"- Current Stock: {stock} units\n"
        f"- Coverage: {'%.1f' % (stock/pred if pred else 0)}× demand\n"
        f"- **Status: {status_emoji}**\n"
        + (f"- Recommended Order: {rec_order} units (covers 2 months)\n" if reorder else "")
        + f"\n_Lead time: {reorder_days} days from order to receipt._"
        + "\n\n" + _bullet_summary(card)
    )


# ── CATEGORY C — Analytics ────────────────────────────────────────────────────

@tool
def customer_lifetime_value(customer_id: str) -> str:
    """Calculate a customer's lifetime value (CLV) and spending trend.
    Use when asked: 'lifetime value', 'CLV', 'how valuable is customer', 'total spend', 'customer value'."""
    customer_id = _normalize_id(customer_id, "CUST")
    cust_rows = _query_db("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    if not cust_rows:
        return f"Customer {customer_id} was not found."
    c = cust_rows[0]

    txns = _query_db(
        "SELECT * FROM transactions WHERE customer_id = ? AND status = 'Completed' ORDER BY date",
        (customer_id,),
    )
    if not txns:
        return f"No completed transactions found for {customer_id}."

    total_spend = sum(t["final_amount"] for t in txns)
    order_count = len(txns)
    aov         = round(total_spend / order_count, 2)

    # Monthly frequency: orders / months_active
    first_txn    = date.fromisoformat(txns[0]["date"])
    last_txn     = date.fromisoformat(txns[-1]["date"])
    months_active = max(((last_txn - first_txn).days / 30), 1)
    freq_per_mo  = round(order_count / months_active, 2)

    # CLV projection: 3 year (undiscounted, i.e. rupees valued at face
    # value regardless of when they land) -- kept alongside the NPV figure
    # below so the discounting effect itself is visible.
    clv_3y = round(aov * freq_per_mo * 36, 2)

    # NPV versions: a rupee earned in month 36 is worth less TODAY than one
    # earned next month, so the undiscounted totals above overstate a
    # customer's true present value -- this is what makes the 3-year figure
    # a genuine financial metric rather than a spreadsheet sum. Monthly cash
    # flow is assumed constant at aov * freq_per_mo (the customer's current
    # run-rate); _CLV_DISCOUNT_RATE is a standard cost-of-capital assumption
    # for an Indian retail business, not fit to this data.
    monthly_cashflow = aov * freq_per_mo
    npv_1y = round(_npv(monthly_cashflow, 12, _CLV_DISCOUNT_RATE), 2)
    npv_3y = round(_npv(monthly_cashflow, 36, _CLV_DISCOUNT_RATE), 2)

    # Tier comparison: average CLV for same tier
    tier_avg_rows = _query_db(
        "SELECT AVG(total_spend) as avg_spend FROM customers WHERE tier = ?",
        (c["tier"],),
    )
    tier_avg = round(tier_avg_rows[0]["avg_spend"] or 0, 2)
    vs_tier  = round((total_spend - tier_avg) / tier_avg * 100, 1) if tier_avg else 0

    # Monthly spend breakdown (last 6 months)
    monthly = _query_db(
        "SELECT month, SUM(final_amount) as revenue FROM transactions "
        "WHERE customer_id = ? AND status = 'Completed' "
        "GROUP BY month ORDER BY month DESC LIMIT 6",
        (customer_id,),
    )
    monthly_rows = list(reversed(monthly))

    # The monthly breakdown stays a small table alongside the card+bullets
    # since it's genuinely multi-row time-series data, not a single fact.
    recent_rows = "\n".join(f"| {r['month']} | ₹{r['revenue']:,.0f} |" for r in monthly_rows)
    card = {
        "type":         "calculation",
        "title":        f"Lifetime Value — {c['customer_name']}",
        "result_value": f"₹{npv_1y:,.0f}",
        "result_label": "1-Year CLV (NPV)",
        "metrics": [
            {"label": "Total Lifetime Spend",  "value": f"₹{total_spend:,.2f}"},
            {"label": "Orders",                 "value": str(order_count)},
            {"label": "Avg Order Value",        "value": f"₹{aov:,.2f}"},
            {"label": "3-Year CLV (NPV)",        "value": f"₹{npv_3y:,.0f}"},
            {"label": "3-Year CLV (undiscounted)", "value": f"₹{clv_3y:,.0f}"},
        ],
    }
    return (
        _chart(card)
        + f"**CLV — {c['customer_name']}** ({customer_id}, {c['tier']} Tier)\n\n"
        f"- Total Lifetime Spend: ₹{total_spend:,.2f}\n"
        f"- Orders: {order_count}\n"
        f"- Avg Order Value: ₹{aov:,.2f}\n"
        f"- Purchase Frequency: {freq_per_mo:.2f}/month\n"
        f"- **1-Year CLV (NPV @ {_CLV_DISCOUNT_RATE*100:.0f}%): ₹{npv_1y:,.0f}**\n"
        f"- **3-Year CLV (NPV @ {_CLV_DISCOUNT_RATE*100:.0f}%): ₹{npv_3y:,.0f}** "
        f"(₹{clv_3y:,.0f} undiscounted)\n"
        f"- vs {c['tier']} tier avg: {vs_tier:+.1f}%\n\n"
        f"**Monthly Spend (last 6 months):**\n| Month | Revenue |\n|---|---|\n"
        + recent_rows
        + "\n\n" + _bullet_summary(card)
    )


@tool
def category_performance(category: str) -> str:
    """Analyse sales performance for a product category.
    Use when asked: 'how is Electronics doing', 'category performance', 'category sales', 'Electronics revenue'.
    Valid categories: Electronics, Clothing, Home & Kitchen, Beauty, Sports."""
    rows = _query_db(
        "SELECT month, SUM(final_amount) as revenue, COUNT(*) as orders "
        "FROM transactions WHERE category = ? AND status = 'Completed' "
        "GROUP BY month ORDER BY month",
        (category,),
    )
    if not rows:
        return f"No data found for category '{category}'."

    total_rev = sum(r["revenue"] for r in rows)
    total_ord = sum(r["orders"]  for r in rows)
    aov       = round(total_rev / total_ord, 2) if total_ord else 0

    # Return rate
    ret_rows = _query_db(
        "SELECT COUNT(*) as cnt FROM transactions WHERE category = ? AND status = 'Returned'",
        (category,),
    )
    all_rows = _query_db(
        "SELECT COUNT(*) as cnt FROM transactions WHERE category = ?", (category,),
    )
    return_rate = round(ret_rows[0]["cnt"] / max(all_rows[0]["cnt"], 1) * 100, 1)

    # MoM growth (last 3 months)
    last3 = rows[-3:] if len(rows) >= 3 else rows
    growth_pct = 0.0
    if len(last3) >= 2:
        growth_pct = round((last3[-1]["revenue"] - last3[0]["revenue"]) / max(last3[0]["revenue"], 1) * 100, 1)

    # Top 3 products by revenue
    top_prods = _query_db(
        "SELECT product_name, SUM(final_amount) as rev "
        "FROM transactions WHERE category = ? AND status = 'Completed' "
        "GROUP BY product_name ORDER BY rev DESC LIMIT 3",
        (category,),
    )

    card = {
        "type":         "forecast",
        "total_forecast": round(total_rev),
        "projected_savings": 0,
        "income": round(total_rev),
        "chart_data": [
            {"name": r["month"], "forecast": round(r["revenue"]), "trend": "rising", "change": None}
            for r in rows[-9:]   # last 9 months for chart
        ],
    }
    top_lines = "\n".join(f"| {p['product_name']} | ₹{p['rev']:,.0f} |" for p in top_prods)
    n_months = len(rows)
    bullet_data = {
        "metrics": [
            {"label": f"Total Revenue ({n_months} months)", "value": f"₹{total_rev:,.0f}"},
            {"label": "Total Orders",                "value": str(total_ord)},
            {"label": "Avg Order Value",             "value": f"₹{aov:,.2f}"},
            {"label": "Return Rate",                 "value": f"{return_rate}%"},
        ],
        "result_label": "3-Month Growth",
        "result_value": f"{growth_pct:+.1f}%",
    }
    return (
        _chart(card)
        + f"**{category} — Category Performance**\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total Revenue ({n_months} months) | ₹{total_rev:,.0f} |\n"
        f"| Total Orders | {total_ord} |\n"
        f"| Avg Order Value | ₹{aov:,.2f} |\n"
        f"| Return Rate | {return_rate}% |\n"
        f"| 3-Month Growth | {growth_pct:+.1f}% |\n\n"
        f"**Top 3 Products by Revenue:**\n| Product | Revenue |\n|---|---|\n"
        + top_lines
        + "\n\n" + _bullet_summary(bullet_data)
    )


@tool
def monthly_trend_analysis() -> str:
    """Analyse the full revenue trend across all months of data: MoM growth, best/worst months, trajectory.
    Use when asked: 'monthly trend', 'how are we doing', 'sales trend', 'overall performance',
    'business performance', 'revenue over time'."""
    rows = _query_db(
        "SELECT month, total_revenue, total_orders, avg_order_value, top_category "
        "FROM monthly_sales ORDER BY month"
    )
    if not rows:
        return "No monthly sales data found."

    revenues = [r["total_revenue"] for r in rows]
    best_idx = revenues.index(max(revenues))
    worst_idx = revenues.index(min(revenues))

    # MoM growth rates
    mom_rates = []
    for i in range(1, len(rows)):
        prev = rows[i-1]["total_revenue"]
        curr = rows[i]["total_revenue"]
        mom_rates.append(round((curr - prev) / prev * 100, 1) if prev else 0)

    avg_mom    = round(sum(mom_rates) / len(mom_rates), 1) if mom_rates else 0
    avg_rev    = round(sum(revenues) / len(revenues), 0)
    total_rev  = round(sum(revenues), 0)

    # Overall trend: compare first half to second half
    mid       = len(rows) // 2
    first_avg = sum(revenues[:mid]) / mid
    second_avg = sum(revenues[mid:]) / max(len(rows) - mid, 1)
    trend_dir = "growing" if second_avg > first_avg * 1.05 else ("declining" if second_avg < first_avg * 0.95 else "stable")

    card = {
        "type":         "forecast",
        "total_forecast": total_rev,
        "projected_savings": 0,
        "income": total_rev,
        "chart_data": [
            {"name": r["month"], "forecast": round(r["total_revenue"]), "trend": "rising", "change": None}
            for r in rows
        ],
    }

    mom_lines = "\n".join(
        f"| {rows[i+1]['month']} | ₹{rows[i+1]['total_revenue']:,.0f} | {mom_rates[i]:+.1f}% |"
        for i in range(len(mom_rates))
    )
    n_months = len(rows)
    bullet_data = {
        "metrics": [
            {"label": f"Total Revenue ({n_months} months)", "value": f"₹{total_rev:,.0f}"},
            {"label": "Monthly Average",            "value": f"₹{avg_rev:,.0f}"},
            {"label": "Best Month",                 "value": f"{rows[best_idx]['month']} (₹{revenues[best_idx]:,.0f})"},
            {"label": "Worst Month",                "value": f"{rows[worst_idx]['month']} (₹{revenues[worst_idx]:,.0f})"},
        ],
        "result_label": "Overall Trajectory",
        "result_value": trend_dir.upper(),
    }
    return (
        _chart(card)
        + f"**TechMart India — {n_months}-Month Revenue Trend**\n\n"
        f"| Summary Metric | Value |\n|---|---|\n"
        f"| Total Revenue ({n_months} months) | ₹{total_rev:,.0f} |\n"
        f"| Monthly Average | ₹{avg_rev:,.0f} |\n"
        f"| Best Month | {rows[best_idx]['month']} (₹{revenues[best_idx]:,.0f}) |\n"
        f"| Worst Month | {rows[worst_idx]['month']} (₹{revenues[worst_idx]:,.0f}) |\n"
        f"| Avg MoM Growth | {avg_mom:+.1f}% |\n"
        f"| Overall Trajectory | {trend_dir.upper()} |\n\n"
        f"**Month-over-Month Growth:**\n| Month | Revenue | MoM Growth |\n|---|---|---|\n"
        f"| {rows[0]['month']} | ₹{rows[0]['total_revenue']:,.0f} | — |\n"
        + mom_lines
        + "\n\n" + _bullet_summary(bullet_data)
    )


# ── CATEGORY D — Comparisons (history-driven) ─────────────────────────────────
# Reached only via the supervisor's history-compare fast path (orchestration/
# supervisor.py): the user asks "which one is cheapest?" after discussing
# specific products/customers earlier in the conversation, with no PRD/CUST id
# in the current question at all — the ids come from conversation history.

@tool
def compare_products(product_ids: list[str], comparison_type: str = "price") -> str:
    """Compare multiple products by price, margin, or base cost.
    Use when asked which of several already-discussed products is cheapest,
    most expensive, or has the best margin. comparison_type: 'price', 'margin', 'base_cost'."""
    results = [p for p in (_get_product(pid.upper()) for pid in product_ids) if p]
    if not results:
        return "No products found to compare."

    # Sort direction and labels depend on what's being compared — a margin
    # comparison's "best" is the HIGHEST value, but a price/cost comparison's
    # "best" (cheapest) is the LOWEST. Using "CHEAPEST"/"MOST EXPENSIVE" for
    # every comparison type mislabeled margin results (results[0] was the
    # highest margin, correctly sorted, but printed as "← CHEAPEST") and the
    # summary lines always printed selling_price even when comparing margin.
    if comparison_type == "margin":
        results.sort(key=lambda x: x["margin_pct"], reverse=True)
        field, label = "margin_pct", "Margin %"
        best_word, worst_word = "Highest Margin", "Lowest Margin"
    elif comparison_type == "base_cost":
        results.sort(key=lambda x: x["base_cost"])
        field, label = "base_cost", "Base Cost"
        best_word, worst_word = "Lowest Cost", "Highest Cost"
    else:
        results.sort(key=lambda x: x["selling_price"])
        field, label = "selling_price", "Selling Price"
        best_word, worst_word = "Cheapest", "Most Expensive"

    def _fmt(p: dict) -> str:
        val = p[field]
        return f"₹{val:,.2f}" if field in ("selling_price", "base_cost") else f"{val}%"

    best, worst = results[0], results[-1]
    rows = []
    for i, p in enumerate(results):
        marker = f" ← {best_word.upper()}" if i == 0 else (f" ← {worst_word.upper()}" if i == len(results) - 1 else "")
        rows.append(f"| {p['product_id']} | {p['product_name']} | {_fmt(p)} |{marker}")

    card = {
        "type":         "calculation",
        "title":        f"Product Comparison — {label}",
        "result_value": _fmt(best),
        "result_label": f"{best_word} — {best['product_name']}",
        "metrics": [{"label": p["product_name"], "value": _fmt(p)} for p in results],
    }
    return (
        _chart(card)
        + f"**Product Comparison — {label}**\n\n"
        f"| Product ID | Name | {label} | |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + f"\n\n**{best_word}: {best['product_name']} ({_fmt(best)})**\n"
        f"**{worst_word}: {worst['product_name']} ({_fmt(worst)})**"
        + "\n\n" + _bullet_summary(card)
    )


@tool
def calculate_multi_product_price(product_ids: list[str]) -> str:
    """Calculate selling price + GST breakdown for MULTIPLE products in one order,
    with a grand total. Use when asked the combined cost/price/GST for two or more
    products at once, e.g. 'how much for PRD001, PRD004 and PRD006 including GST'
    or 'I want to buy prod1, 4, 6, 8, how much and what's the GST'."""
    rows: list[dict] = []
    missing: list[str] = []
    for pid in product_ids:
        prod = _get_product(pid.upper())
        if not prod:
            missing.append(pid.upper())
            continue
        base     = prod["base_cost"]
        margin   = prod["margin_pct"]
        gst_pct  = prod["tax_pct"]
        subtotal = round(base * (1 + margin / 100), 2)
        sell     = round(subtotal * (1 + gst_pct / 100), 2)
        gst_amt  = round(subtotal * gst_pct / 100, 2)
        rows.append({
            "product_id": prod["product_id"], "name": prod["product_name"],
            "sell": sell, "gst_amt": gst_amt, "gst_pct": gst_pct,
        })

    if not rows:
        return f"None of the requested products were found: {', '.join(missing) or product_ids}."

    grand_total = round(sum(r["sell"] for r in rows), 2)
    total_gst   = round(sum(r["gst_amt"] for r in rows), 2)
    avg_price   = round(grand_total / len(rows), 2)

    card = {
        "type":         "calculation",
        "title":        "Multi-Product Order Summary",
        "result_value": f"₹{grand_total:,.2f}",
        "result_label": "Grand Total (incl. GST)",
        "metrics": [
            {"label": r["name"], "value": f"₹{r['sell']:,.2f}"} for r in rows
        ] + [
            {"label": "Total GST",       "value": f"₹{total_gst:,.2f}"},
            {"label": "Average Price",   "value": f"₹{avg_price:,.2f}"},
        ],
    }
    table_rows = "\n".join(
        f"| {r['product_id']} | {r['name']} | ₹{r['sell']:,.2f} | {r['gst_pct']}% (₹{r['gst_amt']:,.2f}) |"
        for r in rows
    )
    missing_note = f"\n\n_Not found in catalogue: {', '.join(missing)}_" if missing else ""
    return (
        _chart(card)
        + f"**Multi-Product Order Summary — {len(rows)} item(s)**\n\n"
        f"| Product | Name | Price (incl. GST) | GST |\n|---|---|---|---|\n"
        + table_rows
        + f"\n| **Grand Total** | | **₹{grand_total:,.2f}** | **₹{total_gst:,.2f}** |"
        + f"\n\n**Average price per product: ₹{avg_price:,.2f}**"
        + missing_note
        + "\n\n" + _bullet_summary(card)
    )


@tool
def calculate_multi_product_loyalty_price(customer_id: str, product_ids: list[str]) -> str:
    """Calculate a customer's loyalty tier discount applied across MULTIPLE products
    bought together, with a combined total. Use when asked the discount/loyalty
    price for a specific customer buying several products in one purchase, e.g.
    'what discount does CUST003 get on Smartphone and Laptop combined'."""
    cust = _get_customer(customer_id.upper())
    if not cust:
        return f"Customer {customer_id} not found."

    tier        = cust["tier"]
    loyalty_pct = COMPANY["discounts"]["loyalty"].get(tier, 0)

    rows: list[dict] = []
    missing: list[str] = []
    for pid in product_ids:
        prod = _get_product(pid.upper())
        if not prod:
            missing.append(pid.upper())
            continue
        unit_price    = prod["selling_price"]
        discount_amt  = round(unit_price * loyalty_pct / 100, 2)
        loyalty_price = round(unit_price - discount_amt, 2)
        rows.append({
            "product_id": prod["product_id"], "name": prod["product_name"],
            "standard": unit_price, "discount": discount_amt, "loyalty_price": loyalty_price,
        })

    if not rows:
        return f"None of the requested products were found: {', '.join(missing) or product_ids}."

    standard_total = round(sum(r["standard"] for r in rows), 2)
    total_discount = round(sum(r["discount"] for r in rows), 2)
    loyalty_total  = round(standard_total - total_discount, 2)

    shipping_free = COMPANY["shipping"]["free_above"]
    shipping      = 0.0 if loyalty_total > shipping_free else COMPANY["shipping"]["flat_rate"]
    grand_total   = round(loyalty_total + shipping, 2)

    card = {
        "type":         "calculation",
        "title":        f"Combined Loyalty Price — {cust['customer_name']} ({tier})",
        "result_value": f"₹{grand_total:,.2f}",
        "result_label": f"{tier} Loyalty Total (incl. shipping)",
        "metrics": [
            {"label": r["name"], "value": f"₹{r['loyalty_price']:,.2f}"} for r in rows
        ] + [
            {"label": "Total Discount", "value": f"₹{total_discount:,.2f}"},
            {"label": "Shipping",       "value": f"₹{shipping:,.2f}"},
        ],
    }
    table_rows = "\n".join(
        f"| {r['product_id']} | {r['name']} | ₹{r['standard']:,.2f} | −₹{r['discount']:,.2f} | ₹{r['loyalty_price']:,.2f} |"
        for r in rows
    )
    missing_note = f"\n\n_Not found in catalogue: {', '.join(missing)}_" if missing else ""
    return (
        _chart(card)
        + f"**{cust['customer_name']}** ({customer_id.upper()}) is a **{tier} tier** customer "
        f"— {loyalty_pct}% loyalty discount.\n\n"
        f"**Combined Loyalty Price — {len(rows)} item(s)**\n\n"
        f"| Product | Name | Standard | Discount | Loyalty Price |\n|---|---|---|---|---|\n"
        + table_rows
        + f"\n| | **Subtotal** | ₹{standard_total:,.2f} | −₹{total_discount:,.2f} | ₹{loyalty_total:,.2f} |"
        + f"\n\nShipping: ₹{shipping:,.2f}\n\n**Grand Total: ₹{grand_total:,.2f}**"
        + missing_note
        + "\n\n" + _bullet_summary(card)
    )


@tool
def calculate_multi_product_bulk_quote(product_ids: list[str], quantities: list[int]) -> str:
    """Calculate a combined bulk quote for MULTIPLE products, each with its OWN
    quantity, with one combined bulk discount and shipping charge. Use when asked
    for a bulk quote covering several DIFFERENT products with different quantities
    each, e.g. '5 units of Smartphone and 3 units of Tablet'. product_ids and
    quantities must be the same length and in the same order (quantities[i] is the
    quantity for product_ids[i])."""
    rows: list[dict] = []
    missing: list[str] = []
    for pid, qty in zip(product_ids, quantities):
        prod = _get_product(pid.upper())
        if not prod:
            missing.append(pid.upper())
            continue
        unit_price = prod["selling_price"]
        line_total = round(unit_price * qty, 2)
        rows.append({
            "product_id": prod["product_id"], "name": prod["product_name"],
            "qty": qty, "unit_price": unit_price, "line_total": line_total,
        })

    if not rows:
        return f"None of the requested products were found: {', '.join(missing) or product_ids}."

    subtotal = round(sum(r["line_total"] for r in rows), 2)

    bulk_threshold = COMPANY["discounts"]["bulk_threshold"]
    bulk_pct       = COMPANY["discounts"]["bulk_discount_pct"]
    bulk_eligible  = subtotal > bulk_threshold
    discount_amt   = round(subtotal * bulk_pct / 100, 2) if bulk_eligible else 0.0
    after_discount = round(subtotal - discount_amt, 2)

    shipping_free_above = COMPANY["shipping"]["free_above"]
    shipping_flat       = COMPANY["shipping"]["flat_rate"]
    shipping            = 0.0 if after_discount > shipping_free_above else shipping_flat
    grand_total         = round(after_discount + shipping, 2)

    card = {
        "type":         "calculation",
        "title":        "Multi-Product Bulk Quote",
        "result_value": f"₹{grand_total:,.2f}",
        "result_label": "Grand Total",
        "metrics": [
            {"label": f"{r['name']} × {r['qty']}", "value": f"₹{r['line_total']:,.2f}"} for r in rows
        ] + [
            {"label": f"Bulk Discount ({bulk_pct}%)" if bulk_eligible else "Bulk Discount",
             "value": f"₹{discount_amt:,.2f}" if bulk_eligible else f"₹0 (below ₹{bulk_threshold:,} threshold)"},
            {"label": "Shipping", "value": f"₹{shipping:,.2f}"},
        ],
    }
    table_rows = "\n".join(
        f"| {r['product_id']} | {r['name']} | {r['qty']} | ₹{r['unit_price']:,.2f} | ₹{r['line_total']:,.2f} |"
        for r in rows
    )
    discount_note = (
        f"**Bulk discount applied: {bulk_pct}% off** (order value ₹{subtotal:,.2f} > ₹{bulk_threshold:,})"
        if bulk_eligible else
        f"No bulk discount (order value ₹{subtotal:,.2f} < threshold ₹{bulk_threshold:,})"
    )
    missing_note = f"\n\n_Not found in catalogue: {', '.join(missing)}_" if missing else ""
    return (
        _chart(card)
        + f"**Multi-Product Bulk Quote — {len(rows)} item(s)**\n\n"
        f"| Product | Name | Qty | Unit Price | Line Total |\n|---|---|---|---|---|\n"
        + table_rows
        + f"\n| | | | **Subtotal** | **₹{subtotal:,.2f}** |"
        + f"\n\n{discount_note}\n"
        f"Shipping: ₹{shipping:,.2f}\n\n**Grand Total: ₹{grand_total:,.2f}**"
        + missing_note
        + "\n\n" + _bullet_summary(card)
    )


@tool
def explain_gst_impact(product_id: str) -> str:
    """Explain how GST affects profit and pricing for a product.
    Use when asked: GST impact, how does GST affect profit,
    net profit after GST, tax impact on margin, what goes to government."""
    prod = _get_product(product_id)
    if not prod:
        return f"Product {product_id} not found."

    base       = prod["base_cost"]
    margin_pct = prod["margin_pct"]
    gst_pct    = prod["tax_pct"]
    sell       = prod["selling_price"]

    margin_amt = round(base * margin_pct / 100, 2)
    subtotal   = round(base + margin_amt, 2)
    gst_amt    = round(subtotal * gst_pct / 100, 2)
    net_profit = margin_amt  # profit before GST — GST is collected on top, not from this

    card = {
        "type":         "calculation",
        "title":        f"GST Impact — {prod['product_name']}",
        "result_value": f"Rs.{net_profit:,.2f}",
        "result_label": "Net Profit/Unit (unaffected by GST)",
        "metrics": [
            {"label": "Base Cost",              "value": f"Rs.{base:,.2f}"},
            {"label": f"Gross Margin ({margin_pct}%)", "value": f"Rs.{margin_amt:,.2f}"},
            {"label": f"GST ({gst_pct}%)",       "value": f"Rs.{gst_amt:,.2f}"},
            {"label": "Selling Price",          "value": f"Rs.{sell:,.2f}"},
        ],
    }
    return (
        _chart(card)
        + f"GST Impact Analysis — {product_id.upper()} ({prod['product_name']})\n\n"
        f"Base Cost:          Rs.{base:,.2f}  (paid to supplier)\n"
        f"Gross Margin ({margin_pct}%): Rs.{margin_amt:,.2f}  (TechMart's revenue)\n"
        f"Subtotal:           Rs.{subtotal:,.2f}\n"
        f"GST ({gst_pct}%):          Rs.{gst_amt:,.2f}  (collected and paid to govt)\n"
        f"Selling Price:      Rs.{sell:,.2f}\n\n"
        f"Net profit per unit: Rs.{net_profit:,.2f}\n"
        f"GST does NOT reduce profit — TechMart collects GST from the customer\n"
        f"and passes it directly to the government.\n"
        f"The Rs.{gst_amt:,.2f} GST is not TechMart's income or expense."
        + "\n\n" + _bullet_summary(card)
    )


@tool
def explain_multi_product_gst_impact(product_ids: list[str]) -> str:
    """Explain how GST affects pricing and profit across MULTIPLE or ALL
    products, grouped by GST rate. Use when asked how/why GST impacts
    pricing or margin for several products,
    a whole category, or "all products", e.g. 'how does GST impact the
    pricing of all products' or 'how does GST affect Clothing in general'."""
    rows: list[dict] = []
    missing: list[str] = []
    for pid in product_ids:
        prod = _get_product(pid.upper())
        if not prod:
            missing.append(pid.upper())
            continue
        base       = prod["base_cost"]
        margin_pct = prod["margin_pct"]
        gst_pct    = prod["tax_pct"]
        margin_amt = round(base * margin_pct / 100, 2)
        subtotal   = round(base + margin_amt, 2)
        gst_amt    = round(subtotal * gst_pct / 100, 2)
        rows.append({
            "product_id": prod["product_id"], "name": prod["product_name"],
            "category": prod["category"], "gst_pct": gst_pct,
            "gst_amt": gst_amt, "sell": prod["selling_price"],
        })

    if not rows:
        return f"None of the requested products were found: {', '.join(missing) or product_ids}."

    by_rate: dict[float, list[dict]] = {}
    for r in rows:
        by_rate.setdefault(r["gst_pct"], []).append(r)

    total_gst   = round(sum(r["gst_amt"] for r in rows), 2)
    avg_gst_pct = round(sum(r["gst_pct"] for r in rows) / len(rows), 1)

    lines = [f"**GST Impact Across {len(rows)} Product(s)**\n"]
    for rate in sorted(by_rate.keys()):
        group = by_rate[rate]
        cats  = sorted({g["category"] for g in group})
        lines.append(f"\n**{rate}% GST** ({', '.join(cats)}) — {len(group)} product(s)")
        for g in group[:5]:
            lines.append(f"  - {g['name']}: GST Rs.{g['gst_amt']:,.2f} on Rs.{g['sell']:,.2f} selling price")
        if len(group) > 5:
            lines.append(f"  - ...and {len(group) - 5} more at {rate}% GST")

    lines.append(
        f"\n**Summary**: Average GST rate {avg_gst_pct}% across {len(rows)} product(s); "
        f"total GST collected Rs.{total_gst:,.2f}.\n"
        f"GST does NOT reduce TechMart's margin — it's collected from customers "
        f"and passed directly to the government on top of the selling price; "
        f"products in higher-GST categories cost customers more but earn "
        f"TechMart no extra profit from that difference."
    )
    missing_note = f"\n\n_Not found in catalogue: {', '.join(missing)}_" if missing else ""

    card = {
        "type":         "calculation",
        "title":        f"GST Impact Across {len(rows)} Product(s)",
        "result_value": f"{avg_gst_pct}%",
        "result_label": "Average GST Rate",
        "metrics": [
            {"label": f"{rate}% GST group", "value": f"{len(group)} product(s)"}
            for rate, group in sorted(by_rate.items())
        ] + [{"label": "Total GST Collected", "value": f"Rs.{total_gst:,.2f}"}],
    }
    return _chart(card) + "\n".join(lines) + missing_note + "\n\n" + _bullet_summary(card)


@tool
def compare_gst_by_category(category1: str = "Clothing", category2: str = "Electronics") -> str:
    """Compare GST rates and pricing between two product categories, with a real
    example product from each pulled live from the database.
    Use when asked: how does GST differ, compare GST between categories,
    GST for clothing vs electronics, tax difference between categories."""
    conn = sqlite3.connect(DB_PATH)
    try:
        results = {}
        for cat in [category1, category2]:
            rows = conn.execute(
                "SELECT product_name, base_cost, margin_pct, tax_pct, selling_price "
                "FROM products WHERE category = ? LIMIT 1",
                (cat,),
            ).fetchall()
            if rows:
                results[cat] = rows[0]
    finally:
        conn.close()

    if not results:
        return f"No data found for categories '{category1}' / '{category2}'."

    output = [f"GST Comparison: {category1} vs {category2}\n"]
    metrics = []
    for cat, r in results.items():
        name, base, margin, gst, sell = r
        margin_amt = round(base * margin / 100, 2)
        gst_amt    = round((base + margin_amt) * gst / 100, 2)

        output.append(f"\n{cat} (GST: {gst}%)")
        output.append(f"Example: {name}")
        output.append(f"  Base Cost:    Rs.{base:,.2f}")
        output.append(f"  Margin:       Rs.{margin_amt:,.2f} ({margin}%)")
        output.append(f"  GST ({gst}%):   Rs.{gst_amt:,.2f}")
        output.append(f"  Sell Price:   Rs.{sell:,.2f}")
        output.append(f"  Net Profit:   Rs.{margin_amt:,.2f} (GST goes to govt)")
        metrics.append({"label": f"{cat} GST ({name})", "value": f"{gst}% — Rs.{gst_amt:,.2f}"})

    gst_rates = COMPANY["pricing"]["gst_rates"]
    gst1 = gst_rates.get(category1)
    gst2 = gst_rates.get(category2)
    diff_label = ""
    if gst1 is not None and gst2 is not None:
        diff = gst2 - gst1
        output.append(f"\nKey Difference:")
        output.append(f"  {category1} GST: {gst1}%")
        output.append(f"  {category2} GST: {gst2}%")
        higher_cat = category2 if diff > 0 else category1
        diff_label = f"{abs(diff)}% higher for {higher_cat}"
        output.append(f"  Difference: {diff_label}")

    output.append(f"\nNote: GST is collected from the customer and paid to the government.")
    output.append(f"It does NOT reduce TechMart's profit margin.")

    card = {
        "type":         "calculation",
        "title":        f"GST Comparison — {category1} vs {category2}",
        "result_value": diff_label or "N/A",
        "result_label": "GST Difference",
        "metrics": metrics,
    }
    return _chart(card) + "\n".join(output) + "\n\n" + _bullet_summary(card)


@tool
def compare_customer_risk(customer_ids: list[str]) -> str:
    """Compare churn risk across multiple already-discussed customers.
    Use when asked which of several customers is most/least at risk of churning.
    Ranks by days_inactive — the single strongest churn predictor in this model
    (feature importance 0.27, see models/churn_metrics.json)."""
    results = [c for c in (_get_customer(cid.upper()) for cid in customer_ids) if c]
    if not results:
        return "No customers found to compare."

    results.sort(key=lambda x: x["days_inactive"], reverse=True)
    most_at_risk, least_at_risk = results[0], results[-1]

    rows = []
    for i, c in enumerate(results):
        marker = " ← MOST AT RISK" if i == 0 else (" ← LEAST AT RISK" if i == len(results) - 1 else "")
        rows.append(
            f"| {c['customer_id']} | {c['customer_name']} | {c['tier']} | "
            f"{c['days_inactive']} days |{marker}"
        )

    card = {
        "type":         "calculation",
        "title":        "Customer Risk Comparison — Days Inactive",
        "result_value": f"{most_at_risk['days_inactive']} days",
        "result_label": f"Most At Risk — {most_at_risk['customer_name']}",
        "metrics": [
            {"label": c["customer_name"], "value": f"{c['days_inactive']} days"}
            for c in results
        ],
    }
    return (
        _chart(card)
        + "**Customer Risk Comparison — Days Inactive**\n\n"
        f"| Customer ID | Name | Tier | Days Inactive | |\n|---|---|---|---|---|\n"
        + "\n".join(rows)
        + f"\n\n**Most at risk: {most_at_risk['customer_name']} "
        f"({most_at_risk['days_inactive']} days inactive)**\n"
        f"**Least at risk: {least_at_risk['customer_name']} "
        f"({least_at_risk['days_inactive']} days inactive)**"
        + "\n\n" + _bullet_summary(card)
    )


# ── Legacy direct calc (generic math without DB) ──────────────────────────────

def _direct_calc(question: str) -> str | None:
    """
    Handle deterministic financial formulas in pure Python before hitting the LLM.
    Covers: GST%, CAGR, percentage-of-amount, growth rate, ROI, profit margin.
    Returns None when no pattern matches — falls through to the ReAct agent.
    """
    q    = question
    _CUR = r"(?:[₹₨]|Rs\.?\s*)?\s*"

    # GST on amount
    m = re.search(
        rf"(\d+(?:\.\d+)?)\s*%\s*gst\s+(?:on|of)\s+{_CUR}([\d,]+(?:\.\d+)?)"
        rf"|gst\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%\s+(?:on|of)\s+{_CUR}([\d,]+(?:\.\d+)?)",
        q, re.IGNORECASE,
    )
    if m:
        pairs = [(m.group(1), m.group(2)), (m.group(3), m.group(4))]
        rate_s, amt_s = next((r, a) for r, a in pairs if r is not None)
        rate, amount  = float(rate_s), float(re.sub(r"[₹₨,\s]", "", amt_s))
        gst = round(amount * rate / 100, 2)
        return (
            f"**GST @ {rate}% on ₹{amount:,.2f}**\n"
            f"GST = ₹{gst:,.2f} | Total = ₹{amount+gst:,.2f} "
            f"(CGST ₹{gst/2:,.2f} + SGST ₹{gst/2:,.2f})"
        )

    # CAGR
    m = re.search(
        r"(?:cagr|compound\s+annual)[^₹\d]*[₹₨]?\s*([\d,]+(?:\.\d+)?)"
        r"\s+to\s+[₹₨]?\s*([\d,]+(?:\.\d+)?)[^₹\d]*?(\d+(?:\.\d+)?)\s*year",
        q, re.IGNORECASE,
    )
    if m:
        start, end, years = float(re.sub(r"[,\s]", "", m.group(1))), float(re.sub(r"[,\s]", "", m.group(2))), float(m.group(3))
        if start > 0 and years > 0:
            rate = (end / start) ** (1 / years) - 1
            return f"**CAGR = {rate*100:.2f}% p.a.** (₹{start:,.0f} → ₹{end:,.0f} over {years:.1f} years)"

    # Percentage of amount
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+[₹₨]?\s*([\d,]+(?:\.\d+)?)", q, re.IGNORECASE)
    if m:
        rate, amount = float(m.group(1)), float(re.sub(r"[,\s]", "", m.group(2)))
        return f"**{rate}% of ₹{amount:,.2f} = ₹{amount*rate/100:,.2f}**"

    return None


# ── System prompt ─────────────────────────────────────────────────────────────

_PRODUCTS_MAP = """
PRD001=Laptop (Electronics), PRD002=Wireless Earbuds (Electronics),
PRD003=Smartphone (Electronics), PRD004=Tablet (Electronics),
PRD005=Cotton Shirt (Clothing), PRD006=Denim Jeans (Clothing),
PRD007=Kurta Set (Clothing), PRD008=Running Shoes (Clothing),
PRD009=Air Purifier (Home & Kitchen), PRD010=Mixer Grinder (Home & Kitchen),
PRD011=Pressure Cooker (Home & Kitchen), PRD012=Water Purifier (Home & Kitchen),
PRD013=Face Serum (Beauty), PRD014=Sunscreen SPF50 (Beauty),
PRD015=Hair Oil Set (Beauty), PRD016=Skincare Kit (Beauty),
PRD017=Yoga Mat (Sports), PRD018=Dumbbells Set (Sports),
PRD019=Cricket Bat (Sports), PRD020=Fitness Tracker (Sports)
"""

_CUSTOMERS_MAP = """
CUST001=Arjun Mehta (Gold), CUST002=Priya Patel (Silver),
CUST003=Vikram Singh (Platinum), CUST004=Deepa Nair (Bronze),
CUST005=Rahul Gupta (Gold), CUST006=Ananya Sharma (Silver),
CUST007=Karthik Rajan (Bronze), CUST008=Sneha Reddy (Gold),
CUST009=Amit Joshi (Bronze), CUST010=Meera Iyer (Silver)
"""

_SYSTEM_PROMPT = f"""\
You are TechMart India's Finance Agent. Use the tools to answer questions about
pricing, discounts, and business analytics. Revenue forecasting is handled by
the Forecast agent and churn/fraud risk by the Risk agent -- not you.

PRODUCT IDs:
{_PRODUCTS_MAP}

CUSTOMER IDs:
{_CUSTOMERS_MAP}

TOOL SELECTION:
- calculate_selling_price  → price of product, how much does X cost
- calculate_bulk_quote     → bulk order, wholesale, N units of PRD
- calculate_loyalty_price  → discount for CUST, loyalty price
- calculate_profit_margin  → profit on product, margin, how much we make
- generate_invoice         → invoice / bill for customer + product
- predict_demand           → restock, demand for product, inventory
- customer_lifetime_value  → CLV, lifetime value, total spend
- category_performance     → how is Electronics/Clothing doing
- monthly_trend_analysis   → monthly trend, overall performance, sales trend
- explain_gst_impact       → how does GST affect profit/margin, tax impact, net profit after GST
- compare_gst_by_category  → compare GST/tax between two categories (e.g. Clothing vs Electronics)
- calculate_multi_product_price → combined price + GST for TWO OR MORE products in one order

RULES:
1. Call EXACTLY ONE tool per question. Stop after the tool returns.
2. Map product names to PRD IDs and customer names to CUST IDs before calling.
3. The tool output contains both markdown AND a <CHART_DATA> block.
   Include the tool output VERBATIM in your final answer — do NOT paraphrase,
   summarise, or strip the <CHART_DATA> block.
4. Do NOT do arithmetic yourself — the tools handle all computation.
5. For generic math (GST%, CAGR, ROI without DB) answer directly in markdown.
6. If NO tool genuinely fits the question — e.g. "summarise our conversation",
   "what did I ask before", anything about the chat itself rather than
   TechMart's data — do NOT force a mismatched tool call. Reply in plain text
   that this isn't something you can look up, or ask what specific data they
   want. Forcing an irrelevant tool (e.g. a customer lifetime value lookup for
   "summarise what we discussed") is worse than declining.
7. If product_id is not specified, state you need a product ID in your response.
"""

_ALL_TOOLS = [
    calculate_selling_price,
    calculate_bulk_quote,
    calculate_loyalty_price,
    calculate_profit_margin,
    generate_invoice,
    predict_demand,
    customer_lifetime_value,
    category_performance,
    monthly_trend_analysis,
    compare_products,
    explain_gst_impact,
    explain_multi_product_gst_impact,
    compare_gst_by_category,
    calculate_multi_product_price,
    calculate_multi_product_loyalty_price,
    calculate_multi_product_bulk_quote,
]


# ── Fast-path tool dispatcher (no LLM routing needed) ────────────────────────

_PRD_RE  = re.compile(r'\b(PRD\d{3})\b', re.IGNORECASE)
_CUST_RE = re.compile(r'\b(CUST\d{3})\b', re.IGNORECASE)
_CAT_RE  = re.compile(
    r'\b(Electronics|Clothing|Sports|Beauty|Home\s*&?\s*Kitchen|Groceries|'
    r'Books\s*&?\s*Stationery|Toys\s*&?\s*Games)\b', re.IGNORECASE
)

# Natural-language product name → PRD id — covers all 20 products plus common
# variants, so "cotton shirts" / "5 laptops" resolve without an explicit PRD id.
_NAME_TO_ID = {
    "laptop": "PRD001", "laptops": "PRD001",
    "wireless earbuds": "PRD002", "earbuds": "PRD002", "airpods": "PRD002",
    "smartphone": "PRD003", "smartphones": "PRD003", "phone": "PRD003", "mobile": "PRD003",
    "tablet": "PRD004", "tablets": "PRD004", "ipad": "PRD004",
    "cotton shirt": "PRD005", "cotton shirts": "PRD005", "shirt": "PRD005", "shirts": "PRD005",
    "denim jeans": "PRD006", "jeans": "PRD006", "denim": "PRD006",
    "kurta set": "PRD007", "kurta": "PRD007", "kurtas": "PRD007",
    "running shoes": "PRD008", "shoes": "PRD008", "sneakers": "PRD008",
    "air purifier": "PRD009", "purifier": "PRD009", "purifiers": "PRD009",
    "mixer grinder": "PRD010", "mixer": "PRD010", "grinder": "PRD010", "grinders": "PRD010",
    "pressure cooker": "PRD011", "cooker": "PRD011", "cookers": "PRD011",
    "water purifier": "PRD012", "water filter": "PRD012",
    "face serum": "PRD013", "serum": "PRD013",
    "sunscreen spf50": "PRD014", "sunscreen": "PRD014", "spf": "PRD014",
    "hair oil set": "PRD015", "hair oil": "PRD015",
    "skincare kit": "PRD016", "skincare": "PRD016",
    "yoga mat": "PRD017", "yoga mats": "PRD017", "mat": "PRD017", "mats": "PRD017",
    "dumbbells set": "PRD018", "dumbbells": "PRD018", "dumbbell": "PRD018", "weights": "PRD018",
    "cricket bat": "PRD019", "bat": "PRD019",
    "fitness tracker": "PRD020", "tracker": "PRD020", "smartwatch": "PRD020",
}
# Longest name first, so "cotton shirts" matches before the bare "shirts".
_NAME_TO_ID_BY_LENGTH = sorted(_NAME_TO_ID.keys(), key=len, reverse=True)


def _find_name(text_lower: str, name: str, last: bool = False) -> int:
    """
    Word-boundary-safe search for a product name — returns its start index,
    or -1 if not found. A plain substring check (`name in text`, `.find()`)
    matches short product names inside unrelated words in ordinary prose:
    "mat" (Yoga Mat) inside "inforMATion", "bat" (Cricket Bat) inside "comBAT"
    or "dataBAse". Conversation history is full of normal English sentences
    (SQL/finance answers, assistant replies), so this false-match risk is real
    and was observed live — a customer lookup answer containing the word
    "Information" silently pulled "Yoga Mat" into an unrelated comparison.
    """
    matches = list(re.finditer(rf'\b{re.escape(name)}\b', text_lower))
    if not matches:
        return -1
    return matches[-1].start() if last else matches[0].start()


def _resolve_product_id(text: str) -> str | None:
    """
    Find a single product reference anywhere in text — a strict PRD### id, a
    shorthand digit form ("prod1", "prd 12"), or a natural-language product
    name. Delegates to _resolve_multi_product_ids (which already handles all
    three forms with overlap-safe matching) rather than duplicating that
    logic — this function used to only understand strict ids and full names,
    silently missing shorthand digits entirely. That gap meant a current
    message like "prod1" resolved to nothing here, which then fell through
    to searching the ENTIRE conversation history for any product mention —
    including truncated SQL result tables from earlier turns — and grabbed
    whatever id happened to land at the truncation boundary. Callers using
    _resolve_single_product() (below) now correctly resolve "prod1" from
    the CURRENT message alone and never reach that noisy history-wide
    fallback at all.

    Returns the LAST-appearing match, not the first — a short follow-up
    ("what about its margin?") must resolve against the most recently
    discussed product, not the first one ever mentioned in a long session.
    """
    ids = _resolve_multi_product_ids(text)
    return ids[-1] if ids else None


def _looks_like_bare_number_list(text: str) -> bool:
    """
    True if text has 2+ bare numbers joined by "and"/","/"&" with NO
    "prod"/"prd" prefix at all — e.g. "margin for 2 and 15 and 19".
    _resolve_multi_product_ids requires a prod/prd prefix before a number
    list, so this shape resolves to nothing there; it means "several
    products, no prefix used," not "no product mentioned at all."
    """
    nums = re.findall(r'\b\d{1,3}\b', text)
    return len(nums) >= 2 and bool(re.search(r'\band\b|,|&', text, re.IGNORECASE))


def _resolve_single_product(q: str, question: str) -> str | None:
    """
    Safe replacement for the `_resolve_product_id(q) or
    _resolve_product_id(question)` pattern used across _fast_dispatch's
    single-product branches (via _resolve_single_product_for_dispatch,
    below, which delegates here). Skips the HISTORY fallback when
    q itself looked like an unprefixed multi-number list ("margin for 2 and
    15 and 19") — that shape means the CURRENT message is asking about
    several products the fast-path regex can't parse, not "no product here,
    check history." Falling back to history for it used to silently return
    an unrelated product mentioned earlier in the conversation with full
    confidence, completely ignoring the numbers the user actually typed —
    reproduced live: "2 and 15 and 19" returned PRD007, a product from two
    turns earlier that isn't even one of the three numbers requested.
    """
    pid = _resolve_product_id(q)
    if pid:
        return pid
    if _looks_like_bare_number_list(q):
        return None
    return _resolve_product_id(question)


_PRD_SHORT_LIST_RE = re.compile(
    r'\b(?:prod(?:uct)?s?|prd)\.?\s*(?:no\.?|number|ids?|of)?\s*[:#]?\s*'
    r'(\d{1,3}(?:\s*(?:,|and|&)\s*\d{1,3})*)',
    re.IGNORECASE,
)


def _resolve_multi_product_ids(text: str) -> list[str]:
    """
    Find every distinct product referenced in text, including shorthand forms
    that _resolve_all_product_ids doesn't cover: "prod1", "prd 4", "product 1,
    4 and 8" (users often drop the PRD00-prefix and zero-padding entirely).

    The returned list is ordered by each product's LAST occurrence in the
    text — ids[-1] is always the most recently mentioned product, which is
    what every "most recent product" caller (_resolve_single_product_for_
    dispatch, etc.) relies on. Real bug this fixes: three separate detection
    passes run over the WHOLE text each (strict "PRDxxx" ids, short comma/
    and-joined lists, then bare product names) — a naive "append if not
    already found" scheme orders products by which PASS found them, not by
    where they actually appear in the text. Across multi-turn history,
    "laptop and wireless earbuds ... what is the price of PRD001" put
    PRD001 (found by the first, regex pass) ahead of Wireless Earbuds (found
    by the later, name pass) even though PRD001's mention is textually the
    LATEST one — so a bare follow-up like "its profit margin too?" resolved
    to the wrong, stale product. Tracking each match's character position
    directly and sorting by that at the end fixes this regardless of which
    pass or textual form found a given product.

    Tracks the character span each match claims and skips any later match
    that overlaps one already claimed — "water purifier" contains "purifier"
    as a literal substring, and the two map to DIFFERENT products (PRD012 vs
    PRD009), so without this a single mention of "Water Purifier" resolved as
    TWO products and single-product questions were wrongly routed to the
    multi-product tools.
    """
    consumed: list[tuple[int, int]] = []
    last_pos: dict[str, int] = {}

    def _overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in consumed)

    def _record(pid: str, pos: int) -> None:
        if pid not in last_pos or pos > last_pos[pid]:
            last_pos[pid] = pos

    for m in re.finditer(r'\bPRD(\d{3})\b', text, re.IGNORECASE):
        _record(f"PRD{m.group(1)}", m.start())
        consumed.append(m.span())

    # finditer (not search) — "prd1 or prd5" is two SEPARATE prefixed mentions,
    # not one comma/and-joined list, so only scanning the first match would
    # silently drop every id after the first "or"/space-separated one.
    for m in _PRD_SHORT_LIST_RE.finditer(text):
        if _overlaps(*m.span()):
            continue
        numbers = re.findall(r'\d{1,3}', m.group(1))
        if not numbers:
            continue
        for n in numbers:
            _record(f"PRD{int(n):03d}", m.start())
        consumed.append(m.span())

    text_lower = text.lower()
    for name in _NAME_TO_ID_BY_LENGTH:  # longest name first
        idx = _find_name(text_lower, name)
        if idx == -1:
            continue
        end = idx + len(name)
        if _overlaps(idx, end):
            continue
        _record(_NAME_TO_ID[name], idx)
        consumed.append((idx, end))

    return sorted(last_pos, key=lambda pid: last_pos[pid])


def _resolve_all_product_ids(text: str) -> list[str]:
    """
    Find every distinct PRD id / product name mentioned in text, in order of
    first appearance — used for comparison questions that name two products
    directly ("why is laptop more expensive than cotton shirt") rather than
    referencing them from conversation history.
    """
    text_lower = text.lower()
    found: list[str] = []
    for m in re.finditer(r'\bPRD\d{3}\b', text, re.IGNORECASE):
        pid = m.group().upper()
        if pid not in found:
            found.append(pid)

    name_hits: list[tuple[int, str]] = []
    for name in _NAME_TO_ID_BY_LENGTH:
        idx = _find_name(text_lower, name)
        if idx == -1:
            continue
        pid = _NAME_TO_ID[name]
        if pid in found or pid in (p for _, p in name_hits):
            continue
        name_hits.append((idx, pid))
    name_hits.sort(key=lambda h: h[0])
    found.extend(pid for _, pid in name_hits)
    return found


# Keywords that mark a message as an order/bulk-quote request rather than a
# generic sentence that happens to contain a number.
_ORDER_INTENT_RE = re.compile(
    r'\b(order|buy|purchase|need|want|would\s+like|looking\s+for|quote|'
    r'units?|pieces?|instead|how\s+about)\b',
    re.IGNORECASE,
)
_BARE_QTY_RE = re.compile(r'\b(\d+)\b')

# A number in a question is NOT always a quantity to order — "GST (18%)",
# "last 3 months", "5 customers returned it" are percentages/time periods/counts,
# not "buy N units". _ORDER_INTENT_RE alone isn't enough to rule these out: "per
# unit" contains the bare word "unit", which matches _ORDER_INTENT_RE's `units?`
# even in a question that has nothing to do with ordering (e.g. "impact on net
# profit per unit" was being read as an 18-unit bulk order because of the "(18%)"
# nearby). When the question also contains an analysis/explanation keyword, treat
# any number found as NOT a quantity, regardless of what _ORDER_INTENT_RE matched.
_ANALYSIS_INTENT_RE = re.compile(
    r'\b(gst|tax|percent|%|impact|affect|margin|profit|rate|'
    r'how\s+does|what\s+is\s+the|explain|difference|compare|'
    r'calculate\s+the|analysis|breakdown)\b',
    re.IGNORECASE,
)

# GST/profit impact — an analysis question, not a calculation for a specific
# order. Checked before quantity/bulk detection: "GST (18%)" would otherwise be
# misread as "18 units", and "impact"/"per unit" match _ORDER_INTENT_RE's bare
# `units?` keyword even though this has nothing to do with placing an order.
_GST_IMPACT_RE = re.compile(
    r'\b(gst|tax)\b.*\b(impact|affect|profit|margin|net|govt|government)\b|'
    r'\b(impact|affect)\b.*\b(gst|tax)\b|'
    r'\bnet\s+profit\b.*\b(gst|tax)\b|'
    r'\b(gst|tax)\b.*\b(reduce|increase|change)\b',
    re.IGNORECASE,
)

# GST comparison BETWEEN categories — "how does GST differ for clothing vs
# electronics?" — distinct from _GST_IMPACT_RE (single-product GST/profit
# analysis); checked first since a comparison question can also contain
# "impact"/"affect"-adjacent wording that would otherwise match _GST_IMPACT_RE
# and try to resolve a single product id instead of two categories.
_GST_COMPARE_RE = re.compile(
    r'\b(gst|tax)\b.*(differ|compare|difference|vs|versus|between|'
    r'clothing.*electronics|electronics.*clothing)\b|'
    r'\bcompare\b.*(gst|tax).*(categor|product)\b',
    re.IGNORECASE,
)
_CATEGORY_WORDS = {
    "clothing":       "Clothing",
    "electronics":    "Electronics",
    "home":           "Home & Kitchen",
    "kitchen":        "Home & Kitchen",
    "beauty":         "Beauty",
    "sports":         "Sports",
    "groceries":      "Groceries",
    "grocery":        "Groceries",
    "books":          "Books & Stationery",
    "stationery":     "Books & Stationery",
    "toys":           "Toys & Games",
    "games":          "Toys & Games",
}

# Discount/tier-only question — "what discount does CUST003 get?", "what tier is
# CUST003?" — must be checked before the SHORT FOLLOW-UP block, which otherwise
# treats any short CUST-containing question as a churn-risk question by default.
_DISCOUNT_ONLY_RE = re.compile(
    r'\bwhat\s+discount\s+does\s+CUST\d+|'
    r'\bCUST\d+.*\bdiscount\b|'
    r'\bdiscount\s+for\s+CUST\d+|'
    r'\bwhat\s+tier\s+is\s+CUST\d+|'
    r'\bCUST\d+.*\btier\b',
    re.IGNORECASE,
)

# Rule/policy question — asking about the bulk-discount policy itself, not a
# calculation for a specific order quantity.
_RULE_RE = re.compile(
    r'\bwhat\s+(quantity|amount|threshold|minimum|limit)\s+(does|is|for)\s+bulk|'
    r'\bwhen\s+does\s+bulk\s+discount|'
    r'\bbulk\s+discount\s+(rule|threshold|policy)|'
    r'\bhow\s+much\s+for\s+bulk|'
    r'\bwhat\s+is\s+the\s+bulk\s+threshold',
    re.IGNORECASE,
)


def _resolve_products_for_dispatch(current: str, full: str, min_count: int = 1) -> list[str]:
    """Product ids for _fast_dispatch: current message first (recency-
    correct — a short follow-up must resolve against what was JUST said),
    falling back to the full history-inclusive text only when the current
    message doesn't itself look like a bare, unprefixed number list (e.g.
    'cost of 2 and 15 and 19'). Falling back for that shape used to grab an
    unrelated product from earlier in the conversation with full confidence,
    ignoring the numbers the user actually typed this turn — see
    _resolve_single_product's docstring for the real bug this guards
    against (reproduced live: "2 and 15 and 19" returned an unrelated
    product from two turns earlier).

    min_count: for comparison-style callers (pass 2) that need MULTIPLE
    products, a current message like "compare that with PRD004" names only
    PRD004 explicitly plus a referential pronoun for the other -- returning
    early with just that one match (the default single-product behavior)
    left the comparison branch with too few products, so it declined and
    fell all the way through to the 60s ReAct/Groq fallback for what should
    be an instant regex answer (reproduced live: this exact phrasing timed
    out at 71s). When under min_count, merge in additional products from
    the full history text instead of returning early."""
    ids = _resolve_multi_product_ids(current)
    if len(ids) >= min_count:
        return ids
    if _looks_like_bare_number_list(current):
        return ids
    merged = list(ids)
    for pid in _resolve_multi_product_ids(full):
        if pid not in merged:
            merged.append(pid)
    return merged


def _resolve_single_product_for_dispatch(current: str, full: str) -> str | None:
    """Single-product convenience wrapper around _resolve_products_for_dispatch
    — last-appearing match wins (recency), same rationale as _resolve_product_id."""
    ids = _resolve_products_for_dispatch(current, full)
    return ids[-1] if ids else None


def _resolve_products_scoped(current: str, full: str, *, allow_history_merge_as_multi: bool) -> list[str]:
    """
    Product ids for a _fast_dispatch branch that has BOTH a single- and a
    multi-product tool (margin, loyalty, ...).

    Real bug this fixes: "Want to see its profit margin too?" names ZERO
    products in the current message, so _resolve_products_for_dispatch fell
    back to merging in every product id mentioned ANYWHERE earlier in the
    conversation (Laptop and Wireless Earbuds, from a bulk quote several
    turns before) -- 2 products found, so the margin branch confidently ran
    a product COMPARISON nobody asked for, instead of the profit margin for
    the ONE product actually just under discussion. A follow-up like
    "loyalty discount instead?" hit the same bug in the loyalty branch,
    silently combining 3 unrelated products from across the whole
    conversation into one "combined loyalty price" quote.

    If the CURRENT message itself explicitly names 2+ products, that's real
    intent -- always honored. Otherwise, only allow the history-merge
    fallback to produce a MULTI result when the caller has already detected
    explicit comparison/"all of these"-style language in the current
    question (allow_history_merge_as_multi=True); otherwise collapse to the
    single most-recent product (recency-based, same as every other
    single-product branch) rather than silently guessing a multi-item action.
    """
    current_only = _resolve_multi_product_ids(current)
    if len(current_only) >= 2:
        return current_only
    if allow_history_merge_as_multi:
        return _resolve_products_for_dispatch(current, full)
    pid = _resolve_single_product_for_dispatch(current, full)
    return [pid] if pid else []


# Explicit admission of uncertainty about WHICH entity -- "not sure which
# product", "don't know which customer" -- must never be silently resolved
# by guessing from history; it must ask, not guess. Real bug: "i want to buy
# customer CUST001 a bulk order but not sure which product" silently
# defaulted to two unrelated products left over from several turns earlier
# instead of asking what the user just said they didn't know.
_UNCERTAIN_ENTITY_RE = re.compile(
    r"\b(not\s+sure|don'?t\s+know|no\s+idea)\s+(which|what)\s+(product|customer|item|one)\b",
    re.IGNORECASE,
)


# ── Customer id resolution (mirrors the product resolvers above) ────────────

_CUST_SHORT_RE = re.compile(
    r'\bcust(?:omer)?\.?\s*(?:id)?\s*[:#]?\s*(\d{1,3})\b', re.IGNORECASE
)


def _resolve_multi_customer_ids(text: str) -> list[str]:
    """Every distinct customer referenced in text — strict CUST### ids plus
    shorthand ("cust3", "customer 3", "customer id 3"). Overlap-safe like
    _resolve_multi_product_ids so a strict match isn't double-counted by
    the shorthand pass."""
    ids: list[str] = []
    consumed: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in consumed)

    for m in re.finditer(r'\bCUST\d{3}\b', text, re.IGNORECASE):
        cid = m.group().upper()
        if cid not in ids:
            ids.append(cid)
        consumed.append(m.span())

    for m in _CUST_SHORT_RE.finditer(text):
        if _overlaps(*m.span()):
            continue
        cid = f"CUST{int(m.group(1)):03d}"
        if cid not in ids:
            ids.append(cid)
        consumed.append(m.span())

    return ids


def _resolve_customer_id(text: str) -> str | None:
    """Last-appearing customer reference — same recency rationale as
    _resolve_product_id: a short follow-up must resolve against the most
    recently discussed customer, not the first one ever mentioned."""
    ids = _resolve_multi_customer_ids(text)
    return ids[-1] if ids else None


def _resolve_customers_for_dispatch(current: str, full: str, min_count: int = 1) -> list[str]:
    """Customer ids for _fast_dispatch — same current-first, merge-when-
    under-min_count, bare-number-list guarded behavior as
    _resolve_products_for_dispatch (see its docstring for the real bug
    both guard against)."""
    ids = _resolve_multi_customer_ids(current)
    if len(ids) >= min_count:
        return ids
    if _looks_like_bare_number_list(current):
        return ids
    merged = list(ids)
    for cid in _resolve_multi_customer_ids(full):
        if cid not in merged:
            merged.append(cid)
    return merged


def _resolve_single_customer_for_dispatch(current: str, full: str) -> str | None:
    ids = _resolve_customers_for_dispatch(current, full)
    return ids[-1] if ids else None


# ── Quantity extraction ──────────────────────────────────────────────────────

_QTY_UNIT_RE = re.compile(r'\b(\d{1,5})\s*(?:units?|pieces?|pcs?|nos?|items?|qty|x)\b', re.IGNORECASE)


def _extract_all_quantities(text: str) -> list[int]:
    """Every 'N units/pieces/items/x' quantity mentioned, in order of
    appearance. "items" was missing here — real bug: "bulk order each 20
    items" matched zero quantities, so _extract_quantities_for_products'
    "not enough quantities found" fallback silently defaulted every product
    to quantity 1 instead of the 20 actually asked for."""
    return [int(n) for n in _QTY_UNIT_RE.findall(text)]


def _extract_quantity(text: str) -> int | None:
    """A single quantity for a single-product order. Prefers an explicit
    'N units' mention; falls back to a bare number directly before a
    product name/id ('buy 5 laptops') only when the sentence has clear
    order intent, so a stray number in an unrelated sentence ('GST 18%')
    is never misread as a quantity."""
    found = _extract_all_quantities(text)
    if found:
        return found[0]
    m = re.search(r'\b(\d{1,5})\s+(?:x\s+)?(?:PRD\d{3}|[a-zA-Z]+)', text)
    if m and _ORDER_INTENT_RE.search(text):
        return int(m.group(1))
    return None


def _extract_quantities_for_products(text: str, product_ids: list[str]) -> list[int]:
    """Best-effort per-product quantities for a multi-product order, e.g.
    '5 units of Smartphone and 3 units of Tablet' -> [5, 3] in the same
    order as product_ids (resolved separately, in the same left-to-right
    scan order). If the count of explicit quantities found doesn't match
    the product count, falls back to a single shared quantity (or 1) for
    every product rather than guessing a pairing that could be wrong."""
    found = _extract_all_quantities(text)
    if len(found) == len(product_ids):
        return found
    default = found[0] if len(found) == 1 else 1
    return [default] * len(product_ids)


# ── Category extraction ──────────────────────────────────────────────────────

_CANON_CATEGORY = {"electronics": "Electronics", "clothing": "Clothing", "sports": "Sports", "beauty": "Beauty"}


def _canon_category(raw: str) -> str:
    stripped = re.sub(r'\s*&?\s*', '', raw.lower())
    if stripped.startswith("home") or "kitchen" in stripped:
        return "Home & Kitchen"
    return _CANON_CATEGORY.get(raw.lower(), raw.title())


def _extract_category(text: str) -> str | None:
    m = _CAT_RE.search(text)
    if m:
        return _canon_category(m.group(1))
    text_lower = text.lower()
    for word, canon in _CATEGORY_WORDS.items():
        if re.search(rf'\b{word}\b', text_lower):
            return canon
    return None


def _extract_all_categories(text: str) -> list[str]:
    """Every distinct category mentioned, in order of appearance — used for
    'compare GST between Clothing and Electronics'-style two-category
    questions."""
    found: list[str] = []
    for m in _CAT_RE.finditer(text):
        canon = _canon_category(m.group(1))
        if canon not in found:
            found.append(canon)
    if len(found) < 2:
        text_lower = text.lower()
        for word, canon in _CATEGORY_WORDS.items():
            if canon not in found and re.search(rf'\b{word}\b', text_lower):
                found.append(canon)
    return found


# ── Fast-path dispatcher: pure regex/keyword intent + entity resolution ─────
# No LLM call for tool SELECTION. Reconstructed for production: deterministic,
# free, and immune to Groq rate limits / occasional malformed tool-call JSON
# that the LLM-based dispatch tiers hit under load. The entity resolvers this
# calls into (_resolve_product_id, _resolve_multi_product_ids, _normalize_id,
# etc.) already carry this session's real bug fixes (word-boundary safety,
# recency-based resolution, overlap-safe multi-product matching) — only the
# intent -> tool mapping below is new/reconstructed.
#
# Branches are ordered most-specific-first so a question that could match
# multiple patterns (e.g. "GST impact" containing the bare word "margin")
# is caught by the intent it actually expresses, not an earlier looser match.

def _fast_dispatch(question: str) -> str | None:
    """
    Deterministic tool dispatch. `question` may be the full contextual
    string built by orchestration/graph.py::_contextual_question (history +
    current question) — current-message text is extracted first via the
    "[Current question]" marker so entity resolution prefers what the user
    JUST said, falling back to the full history-inclusive text only when
    the current message alone doesn't name an entity (recency-correct
    follow-up handling, e.g. "what about its margin?").

    Returns None when no pattern confidently matches — callers fall through
    to the ReAct agent for genuinely novel phrasings.
    """
    marker = "[Current question]\n"
    current = question.split(marker, 1)[1].strip() if marker in question else question.strip()
    full = question
    q_lower = current.lower()

    # ── Explicit uncertainty about which entity — must ask, never guess ────
    if _UNCERTAIN_ENTITY_RE.search(current):
        return (
            "Sure — which product (or products) would you like, and how many "
            "units? Let me know and I'll put the quote together."
        )

    # ── GST comparison between categories — checked before the single-
    # product GST-impact branch, since "compare GST" also contains
    # "impact"/"affect"-adjacent wording that branch looks for.
    if _GST_COMPARE_RE.search(current):
        cats = _extract_all_categories(current) or _extract_all_categories(full)
        if len(cats) >= 2:
            return compare_gst_by_category.invoke({"category1": cats[0], "category2": cats[1]})
        return compare_gst_by_category.invoke({})

    # ── GST impact analysis (single product, multi product, or category) ──
    if _GST_IMPACT_RE.search(current):
        if re.search(r'\ball\s+products?\b', q_lower):
            all_rows = _query_db("SELECT product_id FROM products")
            all_ids  = [r["product_id"] for r in all_rows]
            return explain_multi_product_gst_impact.invoke({"product_ids": all_ids})
        cat = _extract_category(current)
        if cat:
            rows = _query_db("SELECT product_id FROM products WHERE category = ?", (cat,))
            ids = [r["product_id"] for r in rows]
            if ids:
                return explain_multi_product_gst_impact.invoke({"product_ids": ids})
        ids = _resolve_products_for_dispatch(current, full)
        if len(ids) >= 2:
            return explain_multi_product_gst_impact.invoke({"product_ids": ids})
        if len(ids) == 1:
            return explain_gst_impact.invoke({"product_id": ids[0]})
        return None

    # ── Bulk-order policy/rule question (not a specific-order calculation) ─
    if _RULE_RE.search(current):
        threshold = COMPANY["discounts"]["bulk_threshold"]
        pct = COMPANY["discounts"]["bulk_discount_pct"]
        return (
            f"**Bulk Discount Policy**\n\n"
            f"- Orders over ₹{threshold:,} qualify for a **{pct}% bulk discount**.\n"
            f"- The discount is applied automatically once the order subtotal crosses this threshold."
        )

    # ── Discount/tier-only question (customer, no product) ───────────────
    # Checked only when no product is also named -- "what discount does
    # CUST003 get on PRD001 and PRD004" must fall through to the general
    # loyalty branch below (which handles the multi-product case), not stop
    # here and answer with just the customer's bare tier discount.
    if _DISCOUNT_ONLY_RE.search(current) and not _resolve_products_for_dispatch(current, full):
        cid = _resolve_single_customer_for_dispatch(current, full)
        return calculate_loyalty_price.invoke({"customer_id": cid}) if cid else None

    # ── Invoice ────────────────────────────────────────────────────────
    if re.search(r'\b(invoice|bill\s+for|generate\s+(an?\s+)?order|purchase\s+receipt)\b', q_lower):
        cid = _resolve_single_customer_for_dispatch(current, full)
        pid = _resolve_single_product_for_dispatch(current, full)
        if cid and pid:
            qty = _extract_quantity(current) or 1
            return generate_invoice.invoke({"customer_id": cid, "product_id": pid, "quantity": qty})
        return None

    # ── Monthly trend / overall performance ───────────────────────────
    if re.search(
        r'\b(monthly\s+trend|sales\s+trend|overall\s+performance|business\s+performance|'
        r'how\s+are\s+we\s+doing|revenue\s+over\s+time)\b',
        q_lower,
    ):
        return monthly_trend_analysis.invoke({})

    # ── Category performance ──────────────────────────────────────────
    if re.search(r'\bcategory\s+performance\b|\bhow\s+is\s+\w+.*\bdoing\b', q_lower):
        cat = _extract_category(current) or _extract_category(full)
        return category_performance.invoke({"category": cat}) if cat else None

    # ── Demand / restock ───────────────────────────────────────────────
    if re.search(r'\b(demand|restock|run\s+out|inventory\s+check|how\s+many\s+will\s+sell)\b', q_lower):
        pid = _resolve_single_product_for_dispatch(current, full)
        return predict_demand.invoke({"product_id": pid}) if pid else None

    # ── Customer lifetime value ─────────────────────────────────────────
    if re.search(r'\b(lifetime\s+value|\bclv\b|customer\s+value|total\s+spend)\b', q_lower):
        cid = _resolve_single_customer_for_dispatch(current, full)
        return customer_lifetime_value.invoke({"customer_id": cid}) if cid else None

    # ── Loyalty / discount price (customer, optional product) ────────────
    # wants_multi gates whether a history-merged multi-product result is
    # trusted as "the user wants a combined quote." Real bug: "Want a
    # loyalty discount quote for a specific customer instead?" names no
    # product at all, and used to merge in 3 unrelated products scattered
    # across the whole conversation into one bloated "combined loyalty
    # price" quote, instead of the single product actually just discussed.
    if re.search(r'\b(loyalty|discount)\b', q_lower) and not _ANALYSIS_INTENT_RE.search(q_lower):
        cids = _resolve_customers_for_dispatch(current, full)
        if cids:
            cid  = cids[-1]
            wants_multi = bool(re.search(r'\b(all|each|every|combined?|these|those|both)\b', q_lower))
            pids = _resolve_products_scoped(current, full, allow_history_merge_as_multi=wants_multi)
            if len(pids) >= 2:
                return calculate_multi_product_loyalty_price.invoke({"customer_id": cid, "product_ids": pids})
            pid = pids[-1] if pids else None
            return calculate_loyalty_price.invoke({"customer_id": cid, "product_id": pid})
        return None

    # ── Bulk order / quote ────────────────────────────────────────────
    if _ORDER_INTENT_RE.search(current) and not _ANALYSIS_INTENT_RE.search(q_lower):
        pids = _resolve_products_for_dispatch(current, full)
        if pids:
            if len(pids) >= 2:
                qtys = _extract_quantities_for_products(current, pids)
                return calculate_multi_product_bulk_quote.invoke({"product_ids": pids, "quantities": qtys})
            qty = _extract_quantity(current) or 1
            return calculate_bulk_quote.invoke({"product_id": pids[-1], "quantity": qty})
        return None

    # ── Profit margin ──────────────────────────────────────────────────
    # Bare "margin" is enough -- GST-flavored margin questions ("how does
    # GST affect margin") are already routed to the GST-impact branch
    # earlier, so by the time we reach here "margin" reliably means a plain
    # profit-margin question, including a bare follow-up like "what about
    # its margin?" with no other keyword.
    #
    # wants_comparison gates whether a history-merged multi-product result
    # is trusted as "the user wants a comparison." Real bug: "Want to see
    # its profit margin too?" names no product at all, and used to merge in
    # 2 stale products from several turns earlier and run a comparison
    # nobody asked for, instead of the margin for the ONE product actually
    # just under discussion.
    if re.search(r'\b(margin|how\s+much\s+(do\s+)?we\s+make|markup)\b', q_lower):
        wants_comparison = bool(re.search(r'\b(compare|between|versus|vs\.?|both)\b', q_lower))
        pids = _resolve_products_scoped(current, full, allow_history_merge_as_multi=wants_comparison)
        if len(pids) >= 2:
            return compare_products.invoke({"product_ids": pids, "comparison_type": "margin"})
        if len(pids) == 1:
            return calculate_profit_margin.invoke({"product_id": pids[0]})
        return None

    # ── Product comparison (cheapest / most expensive / better value) ───
    if re.search(r'\b(compare|cheapest|most\s+expensive|better\s+value|which\s+(one|is)|versus|vs\.?)\b', q_lower):
        # min_count=2 -- this branch's own trigger regex already requires
        # comparison language, so a current message naming only one product
        # plus a referential pronoun for the other ("compare that with
        # PRD004") must merge in the missing product from history rather
        # than declining and falling through to the slow ReAct/Groq
        # fallback (reproduced live: this exact phrasing timed out at 71s).
        pids = _resolve_products_for_dispatch(current, full, min_count=2)
        if len(pids) >= 2:
            comparison_type = (
                "margin" if "margin" in q_lower else
                "base_cost" if "base cost" in q_lower or "cost" in q_lower else
                "price"
            )
            return compare_products.invoke({"product_ids": pids, "comparison_type": comparison_type})
        return None

    # ── Selling price (default / fallback intent) ─────────────────────
    if re.search(r'\b(price|cost|how\s+much\s+(is|does|for))\b', q_lower):
        pids = _resolve_products_for_dispatch(current, full)
        if len(pids) >= 2:
            return calculate_multi_product_price.invoke({"product_ids": pids})
        if len(pids) == 1:
            return calculate_selling_price.invoke({"product_id": pids[0]})
        return None

    return None


_TOOLS_BY_NAME = {t.name: t for t in _ALL_TOOLS}


async def run(
    question: str,
    knowledge_result: dict | None = None,
    messages: list | None = None,
) -> str:
    """
    Answer a TechMart finance question.

    Priority order:
    1. _direct_calc  -- pure arithmetic (GST%, CAGR) -- no DB, no LLM
    2. _fast_dispatch -- deterministic regex/keyword intent + entity
       resolution -- no LLM call for tool selection at all. Production
       default: free, instant, and immune to Groq rate limits or malformed
       tool-call JSON, at the cost of only covering phrasings the patterns
       actually anticipate.
    3. create_react_agent -- full ReAct loop, last resort for phrasing
       _fast_dispatch's patterns don't recognize.

    knowledge_result is accepted for API compatibility (graph.py passes it)
    but is not used -- the tools query company.db directly. messages is
    accepted for the same reason (graph.py passes it) but currently unused
    -- _fast_dispatch resolves history via the "[Current question]"-prefixed
    contextual string in `question` itself, not a separate messages list.
    """
    from langchain_core.messages import ToolMessage  # local import avoids circular

    try:
        direct = _direct_calc(question)
        if direct:
            return direct
    except Exception:
        pass

    dispatched = _fast_dispatch(question)
    if dispatched is not None:
        return dispatched

    # Full ReAct agent for complex questions
    # max_tokens caps the RESERVED completion budget Groq counts toward its
    # TPM rate limit -- left unset, Groq reserves the model's full max
    # output allowance regardless of actual answer length, confirmed live
    # to cause repeated 413 "rate_limit_exceeded" errors this session.
    llm   = ChatGroq(model=_MODEL, temperature=0, max_tokens=1024)
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
        # Return the first tool output directly -- preserves CHART_DATA blocks
        # and prevents the LLM from paraphrasing the structured answer.
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        if tool_msgs:
            return tool_msgs[0].content
        final_text = result["messages"][-1].content
        if _looks_like_leaked_tool_call(final_text):
            logger.warning("finance_agent: model hallucinated an unexecuted tool call for: %s", question)
            return (
                "I don't have a specific tool for that request. Could you rephrase it, "
                "or ask about a specific product/category/customer?"
            )
        return final_text
    except asyncio.TimeoutError:
        logger.error("finance_agent timed out for: %s", question)
        return "Request timed out. Please try again."
    except Exception as exc:
        logger.error("finance_agent error: %s", exc, exc_info=True)
        return "Unable to complete the financial analysis. Please rephrase the question."
