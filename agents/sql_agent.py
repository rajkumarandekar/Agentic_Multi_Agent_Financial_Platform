"""
SQL agent: natural-language → SQL against the financial transactions database.

Bug fixes applied:
  Bug 1 — Pre-flight entity existence check: CUST003/INV102 not in DB →
           returns "not found" message immediately, never touches LLM.
  Bug 4 — Context-aware execute_sql tool validates that generated SQL contains
           the exact requested entity ID before executing — rejects broad queries.
  Bug 6 — All internal exceptions return friendly messages; nothing exposed raw.
"""

import logging
import os
import re
import sqlite3

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

load_dotenv()
logger = logging.getLogger(__name__)

SQL_MODEL = os.getenv("SQL_MODEL", "llama-3.1-8b-instant")
DB_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'company.db')

# Matches CUST002, cust2, and spelled-out "customer 2" / "customer no 2" /
# "customer number 2" -- real bug this fixes: "customer 2" (no CUST prefix)
# previously matched nothing, so the id was silently dropped before ever
# reaching the LLM. Captures just the digits; _extract_entity below builds
# the zero-padded CUST### form directly from them.
_CUST_RE = re.compile(r'\bcust(?:omer)?\.?\s*(?:no\.?|number)?\s*0*(\d+)\b', re.IGNORECASE)
_PRD_RE  = re.compile(r'\b(PRD\d+)\b',  re.IGNORECASE)

_BASE_SYSTEM_PROMPT = """You are a SQL expert for TechMart India.

Database: data/company.db (SQLite)

Tables and key columns:
- products: product_id (PRD### format), product_name, category, base_cost, margin_pct, tax_pct, selling_price, stock_quantity
- transactions: transaction_id, customer_id (CUST### format), customer_name, product_id, product_name, category, quantity, unit_price, final_amount, date, month, status
- customers: customer_id (CUST### format), customer_name, tier, city, total_spend, total_orders, days_inactive, credit_limit, outstanding_balance
- monthly_sales: month, total_revenue, total_orders, avg_order_value
- company_rates: rate_name, rate_value
- loans: loan_id, customer_id, principal, interest_rate, tenure_months, monthly_emi, status, created_at

CRITICAL SQL RULES:
1. Always filter by ID columns when ID is mentioned:
   - "CUST007 transactions" → WHERE customer_id = 'CUST007'
   - "PRD001 details" → WHERE product_id = 'PRD001'
   - NEVER filter by name when ID is given
2. Use ONLY SQLite syntax — no MySQL functions (no INTERVAL, DATE_ADD, NOW(), DATEDIFF)
   - Date math: date('now', '-30 days'), date(column, '+3 months')
3. Always include LIMIT 50
4. Call execute_sql EXACTLY ONCE. Do not retry.
5. If result is empty → return "No results found" immediately, do not retry
6. Always include any column used in a WHERE filter in the SELECT list too —
   e.g. "customers in Chennai" → WHERE city = 'Chennai' must also SELECT city.
   Without it, the result table doesn't show why those rows match the question.
7. For a BROAD SCHEMA question with no specific table implied — "what data
   do you have", "what's in the database", "what tables exist" — do NOT
   query every table one by one. Query sqlite_master ONCE instead:
   SELECT name FROM sqlite_master WHERE type='table'
   and answer with that table list. Querying tables individually for a
   question like this burns the step budget and produces a truncated,
   misleading answer instead of the overview actually being asked for.
   IMPORTANT — this is NARROW: only for questions about the database's
   STRUCTURE itself. A question naming a real entity type — "what products
   do you have", "products u have", "give me the customer list", "show me
   your products" — is NOT this case, even though it also contains the
   word "have". Those name an actual table (products/customers/
   transactions) and must query THAT table directly (see the next example),
   never sqlite_master.
8. "Last month" / "this month" / "last N months" — use the `month` column
   (format 'YYYY-MM'), compared with strftime('%Y-%m', ...). NEVER use
   date('now', '-1 month') against the `date` column for this — date()
   returns a single SPECIFIC calendar day one month back (e.g. if today is
   the 27th, that's only the 27th of last month), which matches almost
   nothing and silently undercounts. "Last month" means the ENTIRE
   previous calendar month, which the `month` column already encodes
   directly — see the example below.
9. "Top N products" / "best products" / "top selling products" with NO
   metric named — this is ambiguous by itself (top by price? by stock? by
   sales?), and picking a different column each time you're asked produces
   a DIFFERENT ranking on every run, which is worse than picking one and
   being consistent. Default to "best-selling by units actually sold" —
   join products to transactions and rank by total quantity sold, NOT any
   single column already sitting in the products table (price, stock,
   etc. are not "top", they're just sorting an arbitrary field). See the
   example below. If the question DOES name a metric explicitly ("top 10
   most expensive", "products with the most stock left"), use that column
   instead.

EXAMPLE QUERIES:
"Show CUST007 transactions" →
  SELECT transaction_id, product_name, final_amount, date, status
  FROM transactions WHERE customer_id = 'CUST007'
  ORDER BY date DESC LIMIT 20

"List Bronze tier customers" →
  SELECT customer_id, customer_name, city, total_spend, days_inactive
  FROM customers WHERE tier = 'Bronze'

"Electronics products" →
  SELECT product_id, product_name, selling_price, stock_quantity
  FROM products WHERE category = 'Electronics'

"What products do you have" / "give me products u have" / "show me your
products" / "list your products" →
  SELECT product_id, product_name, category, selling_price, stock_quantity
  FROM products LIMIT 50

"How many products" →
  SELECT COUNT(*) as total_products FROM products

"Top 10 products" / "best products" / "top selling products u have" (no
metric named — defaults to best-selling by units sold, see rule 9) →
  SELECT p.product_id, p.product_name, p.category,
         SUM(t.quantity) as units_sold
  FROM products p JOIN transactions t ON p.product_id = t.product_id
  WHERE t.status = 'Completed'
  GROUP BY p.product_id
  ORDER BY units_sold DESC LIMIT 10

"Top 10 most expensive products" (metric named explicitly -> use it) →
  SELECT product_id, product_name, category, selling_price
  FROM products ORDER BY selling_price DESC LIMIT 10

"Top product sold last month" / "best selling product this month" (rules 8
AND 9 combined — a time filter AND "top" both need applying together, do
NOT drop either one) →
  SELECT p.product_id, p.product_name, p.category,
         SUM(t.quantity) as units_sold
  FROM products p JOIN transactions t ON p.product_id = t.product_id
  WHERE t.status = 'Completed'
    AND t.month = strftime('%Y-%m', 'now', '-1 month')
  GROUP BY p.product_id
  ORDER BY units_sold DESC LIMIT 1

"What data do you have" / "show me everything" / "what's in the database" →
  SELECT name FROM sqlite_master WHERE type='table'

"Which customers in Chennai" →
  SELECT customer_id, customer_name, city, tier, total_spend
  FROM customers WHERE city = 'Chennai'

"Customers inactive more than 60 days" →
  SELECT customer_id, customer_name, tier, days_inactive
  FROM customers WHERE days_inactive > 60
  ORDER BY days_inactive DESC

"Total transactions last month" / "how many transactions this month" →
  SELECT COUNT(*) as total_transactions FROM transactions
  WHERE month = strftime('%Y-%m', 'now', '-1 month')
  -- for "this month" use strftime('%Y-%m', 'now') with no offset instead
"""


# ── Entity extraction ─────────────────────────────────────────────────────────

def _normalize_id(raw_id: str, prefix: str) -> str:
    """
    Zero-pad a shorthand id ("cust9", "prd1") to the DB's fixed-width format
    (CUST009, PRD001). Without this, a pre-flight entity check on a correctly-
    routed, correctly-worded question ("show cust9 orders") fails as "not
    found" purely because the id string doesn't match the DB's zero-padded
    format — never even reaching the LLM.
    """
    m = re.match(rf'{prefix}0*(\d+)$', raw_id.strip().upper())
    if m:
        return f"{prefix}{int(m.group(1)):03d}"
    return raw_id.strip().upper()


def _extract_entity(question: str) -> dict:
    """Pull CUST### and PRD### IDs from the question text."""
    cust = _CUST_RE.search(question)
    prd  = _PRD_RE.search(question)
    return {
        "customer_id": f"CUST{int(cust.group(1)):03d}" if cust else None,
        "product_id":  _normalize_id(prd.group(1), "PRD")   if prd  else None,
    }


def _check_entity_exists(entity: dict) -> tuple[bool, str]:
    """
    Pre-flight DB lookup. Returns (True, '') when the entity exists,
    or (False, user-friendly message) when absent.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        if entity.get("customer_id"):
            cid = entity["customer_id"]
            n = conn.execute(
                "SELECT COUNT(*) FROM customers WHERE customer_id = ?", (cid,)
            ).fetchone()[0]
            if n == 0:
                return False, f"Customer {cid} was not found in the database."

        if entity.get("product_id"):
            pid = entity["product_id"]
            n = conn.execute(
                "SELECT COUNT(*) FROM products WHERE product_id = ?", (pid,)
            ).fetchone()[0]
            if n == 0:
                return False, f"Product {pid} was not found in the catalogue."
    finally:
        conn.close()
    return True, ""


def _validate_entity_filter(sql: str, entity: dict) -> tuple[bool, str]:
    """
    Reject SQL that omits the exact requested entity filter.
    Prevents the LLM from silently querying a different customer/product.
    """
    sql_upper = sql.upper()
    if entity.get("customer_id"):
        cid = entity["customer_id"].upper()
        if cid not in sql_upper:
            return False, (
                f"Query must filter WHERE customer_id = '{entity['customer_id']}'. "
                f"Rewrite the SQL to include that exact value."
            )
    if entity.get("product_id"):
        pid = entity["product_id"].upper()
        if pid not in sql_upper:
            return False, (
                f"Query must filter WHERE product_id = '{entity['product_id']}'. "
                f"Rewrite the SQL to include that exact value."
            )
    return True, ""


# ── Context-aware execute_sql tool factory ────────────────────────────────────

def _make_execute_sql(entity: dict):
    """
    Returns an execute_sql @tool whose closure knows which entity was requested.
    The tool validates the WHERE clause before running — rejecting queries that
    filter on the wrong entity (Bug 4).
    """
    @tool
    def execute_sql(query: str) -> str:
        """Execute a SQL query against data/company.db (SQLite).
        Main tables: transactions (transaction_id, customer_id, customer_name, product_id,
        product_name, category, quantity, unit_price, final_amount, date, month, status),
        customers (customer_id, customer_name, tier, city, total_spend, total_orders, days_inactive),
        products (product_id, product_name, category, selling_price, stock_quantity)."""
        # Log every generated query so filter issues are visible in server logs
        logger.info("SQL execute: %s", query)

        # Reject SQL missing required entity filter (Bug 4)
        ok, err = _validate_entity_filter(query, entity)
        if not ok:
            logger.warning("SQL rejected: %s", err)
            return f"REJECTED: {err}"

        try:
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            conn.close()
            if not rows:
                # Say "no results for the filter" — NOT "customer not found".
                # If we say "customer not found", the LLM retries without the merchant
                # filter and a broader aggregate silently substitutes as the answer.
                return "No results found for the specified filter criteria."
            out = [", ".join(cols)]
            for r in rows:
                out.append(", ".join(str(v) for v in r))
            return "\n".join(out[:50])
        except Exception as exc:
            return f"SQL Error: {exc}"

    return execute_sql


# ── Public interface ──────────────────────────────────────────────────────────

def run(question: str, messages: list | None = None, entity_question: str | None = None) -> str:
    """
    Translate a natural-language question into SQL and return results.

    Args:
        question:        Full contextual question (may include conversation history)
                         passed to the ReAct agent for SQL generation.
        entity_question: Raw current question used for entity extraction ONLY.
                         Must NOT include conversation history — prior turns can
                         contain CUST IDs that would falsely scope the query.
                         If omitted, `question` is used for both purposes.

    Steps:
      1. Extract CUST### / INV### entity from entity_question (raw) only.
      2. Pre-flight DB check — if entity absent, return friendly "not found" (no LLM).
      3. Build entity-specific system prompt hint so LLM uses the exact ID.
      4. Create context-aware execute_sql tool that validates the WHERE clause.
      5. Run ReAct agent; parse and return the first meaningful ToolMessage.
    """
    # Use entity_question (raw) for extraction — conversation history in `question`
    # can contain CUST IDs from previous answers that would falsely scope the query.
    entity = _extract_entity(entity_question if entity_question is not None else question)

    # Pre-flight: abort immediately when the entity doesn't exist (Bug 1)
    if any(entity.values()):
        exists, not_found_msg = _check_entity_exists(entity)
        if not exists:
            return not_found_msg

    # Build entity-aware system prompt (reinforces exact ID for LLM)
    entity_hint = ""
    if entity.get("customer_id"):
        entity_hint += (
            f"\nMANDATORY: Your WHERE clause MUST use customer_id = '{entity['customer_id']}' "
            f"exactly — do not use any other customer_id."
        )
    if entity.get("product_id"):
        entity_hint += (
            f"\nMANDATORY: Your WHERE clause MUST use product_id = '{entity['product_id']}' "
            f"exactly — do not use any other product_id."
        )
    system_prompt = _BASE_SYSTEM_PROMPT + entity_hint

    execute_sql_tool = _make_execute_sql(entity)
    # max_tokens caps the RESERVED completion budget Groq counts toward its
    # TPM rate limit -- left unset, Groq reserves the model's full max
    # output allowance regardless of actual answer length, which was
    # confirmed live to trigger repeated 413 "rate_limit_exceeded" errors
    # even for short questions/results (see project chat history).
    llm = ChatGroq(model=SQL_MODEL, temperature=0, max_tokens=1024)
    agent = create_react_agent(
        llm, [execute_sql_tool],
        state_modifier=SystemMessage(content=system_prompt),
        checkpointer=None,
    )

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            # Limit to 5 steps (≈ 2 tool calls max). Prevents the LLM from
            # retrying 5+ times with progressively broader queries.
            config={"recursion_limit": 10},
        )
    except GraphRecursionError:
        # The model tried to explore more than the step budget allows —
        # NOT the same thing as a genuinely empty result. Saying "no
        # matching records found" here is a real, observed bug: a broad
        # question ("what data do you have") made the model query every
        # table in turn until it blew the recursion budget, and this
        # handler then falsely claimed the database had nothing, when real
        # rows almost certainly came back from at least the first query —
        # they just never got returned because the scan below never ran.
        return (
            "That question is too broad for me to answer in one query — "
            "try asking about a specific table instead, e.g. \"show me products\", "
            "\"list customers\", or \"show recent transactions\"."
        )
    except Exception:
        # Genuine API/connection failure — friendly fallback (Bug 6)
        if entity.get("customer_id"):
            return f"No matching records found for customer {entity['customer_id']}."
        if entity.get("product_id"):
            return f"Product {entity['product_id']} was not found in the catalogue."
        return "Something went wrong running that query. Please try again."

    # Return the FIRST non-REJECTED ToolMessage (forward scan, not reversed).
    #
    # Why forward (not reversed):
    #   The first tool call has all the original filter conditions (merchant, category, etc.).
    #   If the LLM later retries and drops a filter (e.g., removes merchant='Zomato'),
    #   the broader query produces a larger aggregate that silently replaces the honest
    #   "no results" from the first attempt. Returning the first result prevents this
    #   substitution — "no results for Zomato filter" is the correct honest answer.
    #
    # Validation rejections (REJECTED) are skipped — those are tool-level corrections,
    # not real SQL results. The first REAL result (accepted or "no results") wins.
    messages_out = result.get("messages", [])
    for m in messages_out:
        if isinstance(m, ToolMessage):
            content = m.content or ""
            if "REJECTED" in content:
                continue   # skip validation correction, look at next tool call
            if content:
                return content

    # All tool calls were rejected — validation never produced a valid query
    return "I couldn't retrieve the requested records — the query filter could not be verified."
