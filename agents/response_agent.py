"""
Response Agent: turns raw scratchpad entries into the final user-facing answer.

Runs ONLY after the supervisor emits "done" — never called mid-loop.

Pass-through vs LLM-call logic:
  - Empty scratchpad (chat/greeting)  → LLM chat reply (temp 0.8)
  - Finance present (ALWAYS,          → pass-through, unconditionally, even if
    regardless of what else ran)        SQL/RAG/research also ran this turn
  - SQL present (no finance)          → LLM format with _SQL_TEMPLATE
                                         (table, bold totals)
  - RAG-only (no finance, no research)→ pass-through (already LLM-synthesised)
  - Research-only (no RAG)            → pass-through
  - Remaining combos (RAG+research,   → LLM combine with _SYSTEM_COMBINE
    SQL+research, etc.)

Finance ALWAYS wins the pass-through, no matter what else is in the
scratchpad: finance's own tools already query the DB for whatever they need
(product/customer rows, GST rates, etc.), so a companion SQL/RAG/research
result in the same turn is usually the supervisor having explored a dead end
— not something finance's answer needs help from. Routing finance through
the generic LLM combiner used to hand the LLM finance's correct, fully-
computed answer alongside irrelevant SQL noise and ask it to merge them; it
would invent placeholder numbers to paper over whatever the other source
didn't actually answer. See run(), below, for the concrete case this fixed.

The SQL template produces consistent table formatting and a bold summary
line. RAG and Finance pass-throughs avoid a second LLM call that could
hallucinate or strip structured content (e.g. finance's <CHART_DATA> blocks).
"""
import os
import random
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from observability.trace import trace_llm_call

_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── SQL formatting template ───────────────────────────────────────────────────
# Used when SQL data is present (alone or combined with RAG).
# {sql_results} and {rag_results} are filled at call time.

_SQL_FORMAT_SYSTEM = SystemMessage(content=(
    "You format SQL query results into clean readable responses. "
    "CRITICAL RULES:\n"
    "1. If data rows are present → they exist. NEVER say 'not found' when rows are shown.\n"
    "2. Use exact values from the data — never invent or substitute any values.\n"
    "3. If SQL returns a COUNT=N → say 'N products' (or whatever the count is) — do NOT list them unless names are in the data.\n"
    "4. If SQL returns customer rows → show those exact customers with their exact names.\n"
    "5. Never add summary lines that contradict the data shown.\n"
    "6. Format as a clean table using the actual column values returned.\n"
    "7. Do NOT end with a follow-up question — risks inventing one about "
    "color/size/preference that isn't in the data.\n"
))

# Exact strings execute_sql (agents/sql_agent.py) returns when a query legitimately
# found nothing — checked in Python (see run(), below) BEFORE any LLM call, so the
# LLM never has to judge "is this empty?" itself. Asking it to make that judgment
# call (an earlier version of this prompt did) caused it to invoke the canned
# no-results message even when real rows were present but didn't literally repeat
# every filter term from the question (e.g. a Chennai-filtered customer table that
# doesn't include a city column) — a false "no results" is worse than an occasional
# missed one, so this must be a deterministic string match, not an LLM guess.
_SQL_EMPTY_SENTINELS = (
    "no results found for the specified filter criteria.",
    "no matching records found",
)

# Exact strings finance/credit/risk_agent.py's run() returns from their own
# except blocks when the underlying ReAct/tool call genuinely failed (Groq
# rate limit, timeout, malformed tool call, etc.) -- checked the same way as
# _SQL_EMPTY_SENTINELS above, and for the same reason: a real bug this
# guards against is sql fetching a perfectly good, complete answer (e.g. a
# plain "how many transactions last month" count), the supervisor's LLM
# router THEN unnecessarily chaining into finance/credit/risk anyway
# (misapplying the sql->finance "calculation on top of fetched data"
# pattern to a question that needed no further calculation at all), that
# second call failing, and the failure message silently overwriting sql's
# good answer because finance/credit/risk normally win pass-through
# priority unconditionally. If one of these agents failed AND something
# else in the scratchpad has real content, prefer that instead of the
# failure sentinel.
_AGENT_FAILURE_SENTINELS = (
    "unable to complete the financial analysis. please rephrase the question.",
    "unable to complete the credit analysis. please rephrase the question.",
    "unable to complete the risk analysis. please rephrase the question.",
    "request timed out. please try again.",
)

def _format_single_column_sql_result(sql_result: str) -> str | None:
    """
    Deterministic formatting for single-column SQL results (e.g. a
    'SELECT name FROM sqlite_master' table list, or any query that returns
    just one column) -- no LLM call, so there's nothing for a small model
    to hallucinate about.

    Real bug this fixes: asking "what data do you have" (which now
    correctly runs a single sqlite_master query, see agents/sql_agent.py)
    produced a real, correct 5-row single-column result (products,
    customers, transactions, monthly_sales, company_rates) -- but the LLM
    formatting step below invented an entirely fictional multi-column
    table (fake product names, fake customer names, fake amounts) instead
    of just listing the 5 real table names. A single-column result has no
    ambiguity to format -- it's always a plain list -- so there's no reason
    to risk an LLM pass over it at all.

    Returns None if the result isn't single-column (the exact format
    agents/sql_agent.py's execute_sql tool produces has no comma in the
    header/row for a 1-column result), letting the caller fall through to
    the LLM formatter for normal multi-column tabular data.
    """
    lines = [l for l in sql_result.split("\n") if l.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].strip()
    if "," in header:
        return None  # multi-column — not this path
    values = [l.strip() for l in lines[1:] if "," not in l]
    if len(values) != len(lines) - 1:
        return None  # a "row" had a comma — header lied about being 1 column
    bullets = "\n".join(f"- {v}" for v in values)
    label = header.replace("_", " ").title()
    return f"**{label} ({len(values)}):**\n\n{bullets}"


_SQL_FORMAT_TEMPLATE = """\
DATA FROM SQL ({row_count} data rows):
{sql_results}

DATA FROM DOCUMENTS (RAG):
{rag_results}

USER QUESTION:
{question}

FORMATTING RULES:
- STEP 1: The SQL data above contains EXACTLY {row_count} data rows (not counting the header).
  You MUST include ALL {row_count} rows in your table — do not omit any row.
- STEP 2: If rows are present, those records EXIST. NEVER say "not found" when rows are shown.
- STEP 3: Present all rows as ONE clean markdown table using the exact column values.
  If the data is only a COUNT/aggregate (no named rows), give a prose answer.
- STEP 4: After the table, add ONE bold summary line (e.g. "**3 customers are inactive > 60 days.**").
- STEP 5: Do NOT add a follow-up question of your own.
- Keep ₹ amounts exactly as given — two decimals, no rounding.
- Do NOT include a "Documents" section if RAG data is empty.

Write the answer now."""

# ── Generic combiner for non-SQL multi-source responses ──────────────────────
_SYSTEM_COMBINE = SystemMessage(content=(
    "You are a professional financial assistant. "
    "Format the provided information into a clear, well-structured markdown response. "
    "Use headers, bullet points, and tables where appropriate. "
    "Present numbers with ₹ symbol. Do not invent data — only format what you receive. "
    "Preserve any <CHART_DATA>...</CHART_DATA> blocks exactly as-is."
))

# ── Chat system message ───────────────────────────────────────────────────────
_SYSTEM_CHAT = SystemMessage(content=(
    "You are the conversational voice of TechMart India's AI assistant — a "
    "multi-agent system, not a general-purpose chatbot. Never claim "
    "capabilities this system doesn't have: no writing/essays, no "
    "translation, no image generation, no general knowledge unrelated to "
    "TechMart's own data.\n\n"
    "This system has exactly these specialists:\n"
    "- SQL Agent — raw data lookups: customers, products, transactions, order "
    "history. Example: 'show me all customers in Chennai'\n"
    "- Finance Agent — pricing, GST, bulk quotes, loyalty discounts, profit "
    "margins, demand forecasting. Example: 'what's the profit margin on PRD001?'\n"
    "- Credit Agent — credit eligibility, EMI calculation, loan proposals. "
    "Example: 'can CUST001 apply for a 50000 rupee loan?'\n"
    "- Risk Agent — churn/retention risk prediction, fraud/anomaly checks. "
    "Example: 'is CUST009 at risk of churning?'\n"
    "- Forecast Agent — revenue forecasts at any horizon (days, weeks, "
    "months, years). Example: 'forecast revenue for the next 6 months'\n"
    "- RAG Agent — answers questions about an uploaded PDF document. "
    "Example: 'summarise the uploaded report'\n"
    "- Research Agent — external market/industry benchmarks not in "
    "TechMart's own data\n\n"
    "If asked 'what can you do', 'help', or similar — briefly list these 7 "
    "specialists with one example each. Do NOT describe generic AI assistant "
    "capabilities.\n"
    "For plain greetings ('hi', 'how are you') — just reply warmly in 1-2 "
    "sentences, no capability listing. Vary your phrasing each time, never "
    "repeat the exact same sentence twice.\n"
    "If a 'Recent conversation' block is included below, you DO have real "
    "memory of it — use it directly to answer questions like 'summarise "
    "what we discussed' or 'what did I ask before' with the actual topics "
    "and figures from that history. Never claim the conversation 'just "
    "started' or that you don't retain context when history is provided."
))


# ── Greeting/meta-chat classification ────────────────────────────────────────
# Same rationale as supervisor.py's _GREETING_RE for the regexes themselves:
# narrow, whole-message-anchored patterns so a real question that happens to
# contain "help" or "hi" as part of a longer sentence is never misfired on --
# those still go through the LLM chat path, which also has real conversation
# history available (these classifications don't, by design, since they're
# meant only for content-free chit-chat).
#
# "thanks" and "bye" stay zero-LLM-call canned replies (a fixed reply is
# genuinely never wrong for those). "hi" and "who are you" are answered
# dynamically instead -- see _dynamic_reply below -- so the assistant isn't
# stuck repeating the same handful of stock sentences every time; that
# reintroduces one Groq call for exactly those two cases; falls back to a
# fixed reply only if the call itself fails.
_CANNED_HI_RE     = re.compile(r'^\s*(hi+|hello+|hey+|yo|good\s*(morning|afternoon|evening))\s*[!.]*\s*$', re.IGNORECASE)
_CANNED_THANKS_RE = re.compile(r'^\s*(thanks?|thank\s*you|thx|ty)\s*[!.]*\s*$', re.IGNORECASE)
_CANNED_BYE_RE    = re.compile(r'^\s*(bye|goodbye|see\s*ya|see\s*you|good\s*night)\s*[!.]*\s*$', re.IGNORECASE)
_CANNED_WHOAMI_RE = re.compile(
    r'^\s*(who\s+are\s+(you|u)|what\s+are\s+(you|u)|'
    r'what\s+(can|do|will)\s+(you|u)\s+(do|support|help)|'
    r'how\s+can\s+(you|u)\s+help|help)\s*[?!.]*\s*$',
    re.IGNORECASE,
)

_CANNED_THANKS_REPLIES = [
    "You're welcome! Let me know if you need anything else about TechMart's products, customers, or data.",
    "Happy to help — come back anytime for more pricing or analytics questions.",
]
_CANNED_BYE_REPLIES = [
    "Goodbye! Come back anytime you need TechMart pricing or analytics.",
    "See you next time — happy to help again whenever you need TechMart data.",
]

# Fallback replies ONLY for when the dynamic LLM call below fails -- same
# graceful-degradation contract as every other Groq call in this file.
_HI_FALLBACK_REPLIES = [
    "Hi! I'm TechMart India's assistant. Ask me about product pricing, bulk quotes, customer discounts, loans/EMI, churn risk, or revenue forecasts — or upload a PDF to ask about it.",
    "Hello! Ready to help with TechMart pricing, credit/loans, risk, forecasting, or customer data questions whenever you are.",
]
_WHOAMI_FALLBACK_REPLY = (
    "I'm TechMart India's assistant, backed by 7 specialists:\n"
    "- **SQL** — raw data lookups (customers, products, transactions)\n"
    "- **Finance** — pricing, GST, discounts, margins\n"
    "- **Credit** — credit eligibility, EMI, loan proposals\n"
    "- **Risk** — churn/retention prediction, fraud checks\n"
    "- **Forecast** — revenue forecasts at any horizon\n"
    "- **RAG** — answers questions about an uploaded PDF\n"
    "- **Research** — external market/industry benchmarks\n\n"
    "Just ask a question and I'll route it to the right one."
)

_HI_SYSTEM = SystemMessage(content=(
    "You are TechMart India's assistant. The user just greeted you (\"hi\", "
    "\"hello\", etc). Reply warmly in 1-2 sentences, conversational tone -- "
    "vary your phrasing every time, never repeat a stock sentence. Briefly "
    "mention you can help with product pricing, bulk quotes, customer "
    "discounts, loans/EMI, churn risk, revenue forecasts, or questions "
    "about an uploaded PDF. Do not format this as a bullet list -- keep it "
    "natural chat, not a capability menu."
))

_WHOAMI_SYSTEM = SystemMessage(content=(
    "You are TechMart India's assistant. The user asked who you are or what "
    "you can do. Explain -- in your own words, varying the phrasing every "
    "time, never the exact same sentence twice -- that you are backed by "
    "exactly these 7 specialists and nothing else:\n"
    "- SQL Agent — raw data lookups: customers, products, transactions, order history\n"
    "- Finance Agent — pricing, GST, bulk quotes, loyalty discounts, profit margins\n"
    "- Credit Agent — credit eligibility, EMI calculation, loan proposals\n"
    "- Risk Agent — churn/retention risk prediction, fraud/anomaly checks\n"
    "- Forecast Agent — revenue forecasts at any horizon (days/weeks/months/years)\n"
    "- RAG Agent — answers questions about an uploaded PDF document\n"
    "- Research Agent — external market/industry benchmarks\n"
    "Never claim capabilities beyond these seven (no essay writing, no "
    "translation, no image generation, no general knowledge unrelated to "
    "TechMart's own data). 3-4 short sentences or a short list, whichever "
    "reads more naturally."
))


def _dynamic_reply(system: SystemMessage, question: str, fallback: str) -> str:
    """Generate a fresh reply via a short Groq call; return `fallback`
    unchanged if the call itself fails (network/quota/etc)."""
    try:
        trace_llm_call("RESPONSE dynamic greeting", query=question, system=system.content)
        # max_tokens caps the RESERVED completion budget Groq counts toward
        # its TPM rate limit -- see agents/sql_agent.py's identical comment.
        llm      = ChatGroq(model=_MODEL, temperature=0.8, max_tokens=256, max_retries=0)
        response = llm.invoke([system, HumanMessage(content=question)])
        return response.content
    except Exception:
        return fallback


def _classify_greeting(question: str) -> str | None:
    """Pure regex classification -- no LLM call, no side effects. Returns
    'hi' | 'thanks' | 'bye' | 'whoami' | None."""
    q = question.strip()
    if _CANNED_HI_RE.match(q):
        return "hi"
    if _CANNED_THANKS_RE.match(q):
        return "thanks"
    if _CANNED_BYE_RE.match(q):
        return "bye"
    if _CANNED_WHOAMI_RE.match(q):
        return "whoami"
    return None


def _greeting_reply(question: str) -> str | None:
    """Return a reply for obvious greeting/meta-chat, or None if this isn't
    one (caller falls through to the real LLM chat path)."""
    kind = _classify_greeting(question)
    if kind == "hi":
        return _dynamic_reply(_HI_SYSTEM, question, random.choice(_HI_FALLBACK_REPLIES))
    if kind == "thanks":
        return random.choice(_CANNED_THANKS_REPLIES)
    if kind == "bye":
        return random.choice(_CANNED_BYE_REPLIES)
    if kind == "whoami":
        return _dynamic_reply(_WHOAMI_SYSTEM, question, _WHOAMI_FALLBACK_REPLY)
    return None


def _build_chat_history(messages: list, n: int = 10) -> str:
    """Recent conversation history for the chat/greeting path — needed for
    meta-conversation questions ('summarise what we discussed', 'what did
    I ask before') that have no scratchpad data to fall back on at all."""
    prior = (messages or [])[:-1][-n:]
    if not prior:
        return ""
    lines = []
    for m in prior:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content[:300]}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content[:200]}")
    return "\n".join(lines)


async def run(question: str, scratchpad: list[dict], messages: list | None = None) -> str:
    """
    Combine scratchpad results into the final answer.

    scratchpad entries are dicts produced by each agent node:
        {"agent": "sql"|"rag"|"finance"|"research", "result": str, ...}

    messages is the full conversation history — only used by the empty-
    scratchpad chat path, so a meta-conversation question ("summarise what
    we discussed") routed straight to "done" can actually answer from real
    history instead of the current question in isolation.
    """
    # Index by agent type — last entry wins if the same agent ran twice
    by_agent: dict[str, str] = {}
    for entry in scratchpad:
        name   = entry.get("agent", "")
        result = entry.get("result", "")
        if name and result:
            by_agent[name] = str(result)

    sql_result      = by_agent.get("sql", "")
    rag_result      = by_agent.get("rag", "")
    finance_result  = by_agent.get("finance", "")
    credit_result   = by_agent.get("credit", "")
    risk_result     = by_agent.get("risk", "")
    forecast_result = by_agent.get("forecast", "")
    research_result = by_agent.get("research", "")

    has_sql      = bool(sql_result)
    has_rag      = bool(rag_result)
    has_fin      = bool(finance_result)
    has_credit   = bool(credit_result)
    has_risk     = bool(risk_result)
    has_forecast = bool(forecast_result)
    has_research = bool(research_result)

    # ── Chat / greeting — empty scratchpad ────────────────────────────────────
    if not (has_sql or has_rag or has_fin or has_credit or has_risk or has_forecast or has_research):
        greeting = _greeting_reply(question)
        if greeting is not None:
            return greeting

        history = _build_chat_history(messages)
        chat_prompt = (
            f"Recent conversation:\n{history}\n\n[Current message]\n{question}"
            if history else question
        )
        try:
            trace_llm_call(
                "RESPONSE chat/greeting",
                query=question, history=history, system=_SYSTEM_CHAT.content,
            )
            # max_tokens caps the RESERVED completion budget Groq counts
            # toward its TPM rate limit -- see sql_agent.py's identical comment.
            llm      = ChatGroq(model=_MODEL, temperature=0.8, max_tokens=512, max_retries=0)
            response = llm.invoke([_SYSTEM_CHAT, HumanMessage(content=chat_prompt)])
            return response.content
        except Exception:
            return "Hi! I'm having trouble reaching the language model right now — please try again in a moment."

    # ── Finance/Credit/Risk/Forecast present → pass through directly, UNLESS
    # research also ran ──────────────────────────────────────────────────────
    # Each of these agents' own tools already query the DB for whatever they
    # need (product/customer rows, GST rates, credit limits, churn model,
    # ARIMA model, etc.) — a companion SQL call in the same turn is usually
    # the supervisor exploring a dead end, and combining through
    # _SYSTEM_COMBINE used to hand the LLM a correct, fully-computed answer
    # alongside SQL noise and ask it to merge them, inventing placeholder
    # numbers to paper over whatever SQL didn't actually answer. None of
    # these four ever need SQL's help to explain themselves.
    #
    # research is different: it's the ONLY source of external/industry data
    # in this system, and none of these four have any access to it either.
    # A compound question ("compare our margins to industry benchmarks")
    # that routed to research first and one of these four second needs BOTH
    # halves — falling through to the generic combiner below (which builds
    # its prompt from all present sources) instead of discarding research.
    #
    # Also skip the pass-through when the agent's own result IS one of its
    # known failure sentinels (Groq rate-limited, timed out, etc.) AND
    # something else in the scratchpad has real content -- real bug this
    # guards against: sql fetches a complete, correct answer (e.g. "how many
    # transactions last month"), the supervisor's router THEN unnecessarily
    # chains into finance anyway (over-applying the sql->finance "needs a
    # calculation on top" pattern to a question that needed none), that
    # second call fails, and finance's failure message silently overwrote
    # sql's good answer because finance normally wins pass-through
    # unconditionally. If nothing else has real content, still return the
    # failure message as before -- it's honest and there's nothing better.
    other_present = has_sql or has_rag or has_research
    fin_failed      = has_fin      and finance_result.strip().lower()  in _AGENT_FAILURE_SENTINELS
    credit_failed   = has_credit   and credit_result.strip().lower()   in _AGENT_FAILURE_SENTINELS
    risk_failed     = has_risk     and risk_result.strip().lower()     in _AGENT_FAILURE_SENTINELS
    forecast_failed = has_forecast and forecast_result.strip().lower() in _AGENT_FAILURE_SENTINELS

    if has_fin and not has_research and not (fin_failed and other_present):
        return finance_result
    if has_credit and not has_research and not (credit_failed and other_present):
        return credit_result
    if has_risk and not has_research and not (risk_failed and other_present):
        return risk_result
    if has_forecast and not has_research and not (forecast_failed and other_present):
        return forecast_result

    # ── SQL present (alone or with RAG, but NOT research) → SQL formatting
    # template. Real bug this guards against: a compound question (e.g.
    # "market trends AND compare with our TechMart data") can route to
    # research first, then sql for the internal half -- sql's own answer
    # (whether empty or just an unhelpful/irrelevant result for a vague
    # query like "market trends") must NOT discard research's perfectly
    # good answer to the OTHER half. Whenever research is ALSO present,
    # skip this whole SQL-only path and fall through to the generic
    # combiner below, which explicitly builds its prompt from sql_result
    # AND research_result together -- this is the only path in this
    # function that actually reconciles the two, and it's why the
    # combiner's own comment already lists "research + SQL" as a case it
    # handles, even though (before this fix) control flow could never
    # actually reach it while has_sql was true.
    #
    # The template handles the "no RAG" case via its own rule:
    #   "If no documents were retrieved, do NOT add a section about it."
    if has_sql and not has_research:
        # Genuinely empty result — decided here in Python, not by the LLM.
        # An earlier version asked the LLM to judge "is this empty?" from
        # the prompt, which made it invoke the canned no-results message
        # even when real rows were present (see _SQL_EMPTY_SENTINELS
        # comment above).
        if sql_result.strip().lower() in _SQL_EMPTY_SENTINELS:
            return ("No matching records found for your query. "
                    "Please check the product name or filter criteria.")

        # Single-column results (schema listings, id-only lists) format
        # deterministically -- no LLM call, nothing to hallucinate. Only
        # when there's no RAG data needing to be combined alongside it.
        if not has_rag:
            single_col = _format_single_column_sql_result(sql_result)
            if single_col is not None:
                return single_col

        # Count data rows (all non-empty lines minus the header line)
        sql_lines  = [l for l in sql_result.split("\n") if l.strip()]
        row_count  = max(0, len(sql_lines) - 1)  # first line is the header
        prompt = _SQL_FORMAT_TEMPLATE.format(
            sql_results = sql_result,
            rag_results = rag_result if has_rag else "None retrieved.",
            question    = question,
            row_count   = row_count,
        )
        try:
            trace_llm_call(
                "RESPONSE sql-format",
                query=question, system=_SQL_FORMAT_SYSTEM.content,
                extra=f"row_count={row_count}",
            )
            # max_tokens caps the RESERVED completion budget Groq counts
            # toward its TPM rate limit -- see sql_agent.py's identical
            # comment. Higher than the other response_agent calls since
            # formatting up to 50 rows (the SQL agent's own LIMIT) genuinely
            # needs more output room.
            llm      = ChatGroq(model=_MODEL, temperature=0.3, max_tokens=2048, max_retries=0)
            response = llm.invoke([_SQL_FORMAT_SYSTEM, HumanMessage(content=prompt)])
            return response.content
        except Exception:
            # Formatting failed, not the query itself — surface the raw SQL
            # result rather than a 500; still useful, just not pretty.
            return f"**Query results:**\n\n{sql_result}"

    # ── RAG-only pass-through ─────────────────────────────────────────────────
    # rag_agent already ran an LLM call; re-processing risks paraphrasing away
    # accurate citations or hallucinating from a second reformatting pass.
    # (has_fin is already False here — that case returned above; has_sql is
    # also already False here whenever has_research is False, since the SQL
    # branch above always returns in that case.)
    if has_rag and not has_research:
        return rag_result

    # ── Research-only pass-through ────────────────────────────────────────────
    # Must also require none of the other sources are present -- otherwise a
    # compound question that routed to research plus sql/finance/credit/risk/
    # forecast would return research alone and silently drop whatever that
    # other agent actually found. Any such combo goes to the generic
    # combiner below instead.
    if (
        has_research and not has_rag and not has_sql
        and not has_fin and not has_credit and not has_risk and not has_forecast
    ):
        return research_result

    # ── Generic LLM combine for remaining multi-source cases ──────────────────
    # (research + sql/finance/credit/risk/forecast/rag, in any combination
    # that reaches this point.)
    parts = [f"Question: {question}"]
    if has_sql:
        parts.append(f"Transaction data:\n{sql_result}")
    if has_fin:
        parts.append(f"Finance analysis:\n{finance_result}")
    if has_credit:
        parts.append(f"Credit analysis:\n{credit_result}")
    if has_risk:
        parts.append(f"Risk analysis:\n{risk_result}")
    if has_forecast:
        parts.append(f"Forecast:\n{forecast_result}")
    if has_rag:
        parts.append(f"Document content:\n{rag_result}")
    if has_research:
        parts.append(f"External research:\n{research_result}")

    try:
        trace_llm_call(
            "RESPONSE combine (multi-source)",
            query=question, system=_SYSTEM_COMBINE.content,
            extra=f"sources: sql={has_sql} rag={has_rag} research={has_research}",
        )
        # max_tokens caps the RESERVED completion budget Groq counts toward
        # its TPM rate limit -- see sql_agent.py's identical comment.
        llm      = ChatGroq(model=_MODEL, temperature=0.3, max_tokens=1536, max_retries=0)
        response = llm.invoke([_SYSTEM_COMBINE, HumanMessage(content="\n\n".join(parts))])
        return response.content
    except Exception:
        # Combining failed, not the underlying data — surface the raw parts
        # rather than a 500; still readable, just not unified into one voice.
        return "\n\n".join(parts)
