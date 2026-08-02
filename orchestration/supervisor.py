"""
supervisor.py — Dynamic LLM-powered routing.

One LLM call per routing decision: the router reads the raw question plus
conversation history and scratchpad, and decides which agent to call. No
separate reframing step — the router prompt carries full history/scratchpad
context, so it acts on state directly rather than a rewritten question.

Routing DECISIONS (which business agent -- sql/rag/finance/research --
handles a real question) were previously fully LLM-driven, no regex. That
decision is now layered rather than absolute: a regex + semantic-embedding
fast path (orchestration/semantic_router.py) runs BEFORE the LLM router for
high-confidence, unambiguous first-turn questions, saving an LLM call. It is
strictly a pre-filter -- if neither layer is confident, routing still falls
through to the LLM, which alone sees conversation history/scratchpad and
resolves ambiguous or context-dependent questions. The fast path only ever
runs on the very first supervisor iteration with no agents run yet, same
restriction _GREETING_RE (below) already uses -- a follow-up like "what
about that one" must never be fast-pathed on keyword/embedding similarity
alone.

_GREETING_RE is a separate, narrower pre-filter that skips the routing LLM
call entirely for messages that are obviously NOT a business question at
all ("hi", "who are you", "thanks") -- not a routing judgment call, just
avoiding paying for an LLM call whose answer is never in doubt. Added under
real Groq TPD-quota pressure (a single "hi" was costing a full router call
with the entire 20-product/10-customer catalog in the prompt).
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from observability.trace import trace_agent_start, trace_llm_call
from orchestration import followup
from orchestration.semantic_router import fast_route
from orchestration.state import AgentState

load_dotenv()

# The supervisor's routing/verification decisions get a STRONGER model than
# the rest of the app's cheap/fast calls -- this is the ONE place a bad
# judgment call cascades into every downstream agent, and it's a single
# short call per turn, so the cost of a bigger model here is negligible.
# llama-3.3-70b-versatile is Groq's actual available 70B-class model
# (verified via client.models.list(), not guessed) and is closer to how a
# real ChatGPT-style orchestrator is built: the strongest model available
# does the orchestrating, even when it delegates to cheaper specialist tools.
#
# Briefly swapped to Gemini (gemini-flash-latest) -- reverted after live
# testing on the deployed HF Space showed 30-120+s hangs/timeouts on
# nearly every request. Root cause: this project's Gemini API key is on
# the free tier, which rate-limits far more aggressively than Groq's, and
# ChatGoogleGenerativeAI silently retries with exponential backoff on
# 429s instead of failing fast -- since routing runs on every single
# message, that backoff stacked into the whole app hanging or timing out
# unpredictably. Don't re-attempt without capping max_retries/timeout and
# adding a fast fallback to Groq first.
_ROUTER_MODEL   = os.getenv("ROUTER_MODEL", "llama-3.3-70b-versatile")
_MAX_ITERATIONS = 3
_VALID_ROUTES   = {
    "sql", "rag", "finance", "research", "credit", "risk", "forecast",
    "clarify", "done",
}

# Obviously-not-a-business-question pre-filter -- see module docstring.
# Deliberately narrow: anchored to (near-)whole-message greetings/meta-chat,
# not a substring match, so it can never misfire on a real question that
# happens to contain "hi" or "help" as part of a longer sentence.
#
# NOTE: "what products/customers do you have" is deliberately NOT in this
# pattern — that phrasing names a real TechMart data request (products/
# customers), not a question about the assistant's own capabilities. It used
# to be lumped in with "who are you"/"what can you do" and got a canned
# capabilities reply instead of the actual product/customer list the user
# was asking for. Only genuine "what can you do"-style meta questions belong
# here.
_GREETING_RE = re.compile(
    r'^\s*(hi+|hello+|hey+|yo|good\s*(morning|afternoon|evening))\s*[!.]*\s*$|'
    r'^\s*(thanks?|thank\s*you|thx|ty)\s*[!.]*\s*$|'
    r'^\s*(bye|goodbye|see\s*ya|see\s*you|good\s*night)\s*[!.]*\s*$|'
    r'^\s*(who\s+are\s+(you|u)|what\s+are\s+(you|u)|'
    r'what\s+(can|do|will)\s+(you|u)\s+(do|support|help)|'
    r'how\s+can\s+(you|u)\s+help|help\s*[?!.]*)\s*[?!.]*\s*$',
    re.IGNORECASE,
)

# research has no TechMart data of its own -- a question asking to compare
# an external/industry figure against "our"/"TechMart's" own numbers needs a
# SECOND agent call after research to fetch the internal half. The router
# LLM (a small, fast model) doesn't reliably follow this as a prompted rule
# even when made very explicit (confirmed live: it re-picked "research" on
# turn 2 regardless of wording) -- this is a deterministic backstop, same
# philosophy as every other hard override in this file (PDF->never sql,
# same-agent-twice->done): don't trust the small model with a nuanced
# multi-step judgment call it has empirically already gotten wrong.
_COMPARE_OWN_DATA_RE = re.compile(
    r'\b(compare|versus|vs\.?)\b.*\b(our|techmart|my\s+own|us)\b|'
    r'\b(our|techmart|my\s+own)\b.*\b(compare|versus|vs\.?)\b',
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_history(state: AgentState, n: int = 8) -> str:
    """
    Build conversation history string from last N messages, plus an explicit
    note about any question -- suggested follow-up OR the chat LLM's own
    free-form clarifying question -- still outstanding from the last answer.

    That note matters because this text (e.g. "Want to see its profit
    margin too?" or "which product and quantity for the invoice?") is NOT
    part of the assistant's actual message content in `messages` -- it's
    tracked separately in state["pending_followup"] (see
    orchestration/state.py and orchestration/followup.py's
    pending_followup_for()). Without surfacing it here, a bare "ok"/"yes"/
    a direct answer has literally nothing in the router's context to
    resolve against -- it's not that the router judges badly, it's that
    it's judging blind. See STRICT RULE 12.
    """
    messages = state.get("messages", [])
    prior    = messages[:-1][-n:]
    lines = []
    for m in prior:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content[:300]}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content[:200]}")

    pending = state.get("pending_followup")
    if pending:
        lines.append(
            f'[Note: your previous answer ended with this pending question '
            f'to the user: "{pending}" — see RULE 12]'
        )

    return "\n".join(lines) if lines else "No prior conversation."


def _build_scratchpad(state: AgentState) -> str:
    """Summarise what agents have gathered so far."""
    scratchpad = state.get("scratchpad", [])
    if not scratchpad:
        return "Nothing gathered yet."
    lines = []
    for entry in scratchpad:
        agent  = entry.get("agent", "?").upper()
        result = str(entry.get("result", ""))[:300]
        lines.append(f"[{agent}]: {result}")
    return "\n\n".join(lines)


# ── LLM Call: Router ──────────────────────────────────────────────────────────

_ROUTER_SYSTEM_PROMPT = """You are the master routing agent for TechMart India AI.

====================================================================
CRITICAL LANGUAGE HANDLING RULE (FOR POOR ENGLISH & TYPOS):
- The user may write using heavily broken English, slang, typos, or rapid shortcuts (e.g., "gimme prodts", "wanna look buyers", "net not work yestday", "custmr balance").
- Do NOT search for exact match keywords from the rules below.
- Step 1: Translate the broken sentence mentally into its core business meaning.
- Step 2: Map that underlying meaning to the correct agent.
- Treat clear typos as their intended words (e.g., "prodts"/"prods" -> products, "custmr"/"cust" -> customer, "refnd" -> finance).
====================================================================

TechMart India Context:
- ~140 products (PRD001-PRD140+) across 8 categories: Electronics, Clothing, Home & Kitchen, Beauty, Sports, Groceries, Books & Stationery, Toys & Games.
- ~250 customers (CUST001-CUST250+) with Bronze/Silver/Gold/Platinum tiers.
- Each customer has a credit_limit, outstanding_balance, and potential loan records (principal, interest_rate, tenure_months, monthly_emi, status).

Use the conversation history to resolve vague or short messages ("it", "that", "what about PRD005", "I asked X not Y") — the current question may only make sense in light of past context.

AGENT SCOPE DEFINITIONS:
- sql: Fetching raw data rows or counts from TechMart's database (e.g., viewing, listing, counting customers, products, transactions, or order histories).
- finance: Pricing, discounts, bulk quotes, loyalty prices, profit margins, GST analysis, product price/margin comparisons, cheapest/most expensive items, restock/demand alerts, customer lifetime value, category/monthly trend analytics.
- credit: Credit eligibility, EMI/installment math, loan proposals/applications, available credit checks, credit limit validation.
- risk: Churn/retention risk prediction, fraud/anomaly/suspicious transaction checks. (NOT pricing).
- forecast: Revenue/sales predictions for any future horizon (days, weeks, months, years). (NOT customer churn).
- rag: Answering questions strictly about an uploaded PDF document.
- research: External market data, industry benchmarks, or general web information. (Has ZERO access to TechMart's internal database).
- done: Question already fully answered in scratchpad, casual greetings, or questions about the conversation itself (e.g., "summarise what we discussed").

STRICT ROUTING RULES:
1. PDF Active: Always route to `rag` first. Never route to `sql` when a PDF is actively selected.
2. Pricing/Calculations: Always route to `finance`. Never use `sql` to compute prices or margins.
3. Raw Internal Data Access: Any underlying intent to look at, list, view, count, or fetch rows from TechMart's internal database of products, customers, or transactions must route to `sql`. This applies regardless of messy phrasing (e.g., "gimme list", "show me who buyed", "i want see buyers details").
4. Bare Lists of Numbers/IDs: If the user provides just IDs/numbers ("prod1, 4, 6" or "12,3,1") and asks for cost/price/average/GST/discount/margin, route straight to `finance`. Never default to `sql` or `research` first for these.
5. Thread Continuity: If the history shows `finance` was just discussing pricing/GST, and the current message is short or numbers-only with no new topic, treat it as a continuation and prefer `finance`.
6. Customer Spending Breakdown: Requests to analyze or break down a customer's actual historical spending/purchases require raw transaction history. Route to `sql` FIRST to fetch rows, then `finance` will handle calculations. Never answer this from `finance` alone.
7. Iteration Cap: If iteration >= 3, always route to `done`.
8. Greetings/Politeness: Pure casual greetings or sign-offs (hi, hello, thanks, bye) route to `done`.
9. Conversational Meta-Questions: Questions asking about the chat itself ("summarise everything", "what did I ask before") have no database data to fetch. Route to `done`.
10. Short Affirmations / Context Replies: Short replies with no keywords ("yes", "ok", "sure", "do it", "prod5") must NEVER be routed to `done` on their own. Check history for a line starting with "[Note: your previous answer ended with this pending question...]". Combine that pending question with the current short reply to determine the true intent, then route accordingly (e.g., Offer pending + "ok" -> route to `finance` or `credit`).
11. Specialty Boundaries:
    - Churn/retention/fraud -> `risk` (never finance).
    - Future revenue horizons -> `forecast` (never finance).
    - Loan approvals/EMI math -> `credit` (never finance).
12. Loan Applications: A request naming a customer AND an amount (e.g., "apply loan 50k for CUST001") is `credit`, not finance.
13. External vs. Internal Comparison: If `research` was already called but the query also asked to compare web data against "our" or "TechMart's" internal metrics, route to `sql` or `finance` next to get the internal data. If no internal comparison is requested, route to `done`.

Reply with EXACTLY ONE WORD from this list: sql | rag | finance | credit | risk | forecast | research | clarify | done"""


def _route_question(
    question:   str,
    history:    str,
    scratchpad: str,
    agents_run: list[str],
    has_pdf:    bool,
    iteration:  int,
    llm,
) -> str:
    """
    Decide which agent to call based on the current question, using
    conversation history and scratchpad for context.
    Returns one word: sql | rag | finance | research | clarify | done
    """
    human_msg = (
        f"PDF active: {has_pdf}\n"
        f"Iteration: {iteration}/3\n"
        f"Agents already called: {agents_run or 'none'}\n\n"
        f"Conversation history:\n{history}\n\n"
        f"Current question: {question}\n\n"
        f"Results gathered so far:\n{scratchpad}\n\n"
        f"Which agent should handle this? Reply ONE WORD only:"
    )

    # Every other Groq call in this codebase (finance_agent.run, sql_agent.run)
    # is wrapped and degrades gracefully. A transient Groq connection error here
    # used to propagate uncaught through graph.ainvoke() in api/main.py and
    # surface as an HTTP 500 for the whole request. Fall back to "done" instead:
    # it's the only route guaranteed not to call another agent, so a flaky LLM
    # call degrades to "answer with whatever's in the scratchpad so far" rather
    # than crashing the request outright.
    try:
        trace_llm_call(
            "SUPERVISOR routing",
            query=question,
            history=history if history != "No prior conversation." else "",
            system=_ROUTER_SYSTEM_PROMPT,
            extra=f"Iteration: {iteration}/3 | Agents already run: {agents_run or 'none'} | PDF active: {has_pdf}\n\nScratchpad:\n{scratchpad}",
        )
        response = llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ])
        raw = response.content.strip().lower()
    except Exception as exc:
        print(f"[supervisor] router call failed ({exc}) -> done")
        return "done"

    route = raw.split()[0] if raw else "done"
    if route not in _VALID_ROUTES:
        route = "done"
    return route


# ── Verifier: "is the question already answered?" ────────────────────────────
# A dedicated, deliberately narrow yes/no check -- NOT part of _route_question
# above. Multi-way routing ("which of 7 agents should handle this") is a
# genuinely hard judgment call for a small/fast model; "does the gathered
# information already fully answer the question, yes or no" is a much easier
# one, and it targets the actual failure mode directly instead of reacting to
# specific bad phrasings after the fact (see supervisor.py's growing set of
# one-off deterministic overrides below, e.g. the sql-single-value check --
# this verifier is meant to catch the same class of case more generally).
_VERIFIER_SYSTEM_PROMPT = """You check whether a question has ALREADY been fully answered by the information gathered so far -- you do NOT decide what to do next, only whether anything more is needed at all.

Reply with EXACTLY ONE WORD: YES or NO.

YES -- the gathered information already contains everything needed to answer the question completely. No further data, calculation, or agent is needed.
NO -- the question genuinely still needs more data, a calculation on top of what's there, or a different agent's help before it can be answered.

When in doubt, prefer YES if the gathered information directly and completely addresses what was asked -- do not require extra polish or a nicer format, only completeness."""


def _is_answer_sufficient(question: str, scratchpad: str, llm) -> bool:
    """
    Ask the dedicated yes/no verifier question. Defaults to False (i.e.
    "not sufficient, keep the normal routing logic in play") on any
    failure -- if we can't tell, prefer falling through to the existing
    router/overrides rather than silently cutting the loop short.
    """
    try:
        trace_llm_call(
            "SUPERVISOR verifier",
            query=question, system=_VERIFIER_SYSTEM_PROMPT,
            extra=f"Scratchpad:\n{scratchpad}",
        )
        response = llm.invoke([
            SystemMessage(content=_VERIFIER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"Gathered so far:\n{scratchpad}\n\n"
                f"Is the question already fully answered? Reply YES or NO."
            )),
        ])
        return response.content.strip().upper().startswith("Y")
    except Exception as exc:
        print(f"[supervisor] verifier call failed ({exc}) -> assume not sufficient")
        return False


# ── Main supervisor node ──────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    question   = state.get("question", "")
    n          = (state.get("iteration_count") or 0) + 1
    scratchpad = state.get("scratchpad", [])
    agents_run = [e.get("agent") for e in scratchpad]
    has_pdf    = bool(
        state.get("source_document")
        and str(state.get("source_document")).strip() not in ("", "None")
    )

    trace_agent_start("SUPERVISOR", question)
    print(f"[supervisor] turn={n} q='{question[:60]}' pdf={has_pdf} agents={agents_run}")

    # ── Hard stops (no LLM needed) ─────────────────────────────────────────────

    if n > _MAX_ITERATIONS:
        print(f"[supervisor] max iterations -> done")
        return {"route": "done", "iteration_count": n}

    if "finance" in agents_run:
        # Human-in-the-loop purchase confirmation: an invoice is an actual
        # order, not just an informational calculation (a price/margin/bulk
        # quote) -- gate it behind an explicit approve/reject before treating
        # it as final. order_confirmed is set True by confirm_purchase_node
        # once the user has approved THIS turn; without that check, the
        # very next iteration (after resuming from the interrupt) would loop
        # right back into confirm_purchase forever.
        last_finance = next((e for e in reversed(scratchpad) if e.get("agent") == "finance"), None)
        result_text  = str(last_finance.get("result", "")) if last_finance else ""
        if "Invoice #" in result_text and not state.get("order_confirmed"):
            print(f"[supervisor] finance produced an invoice -> confirm_purchase")
            return {"route": "confirm_purchase", "iteration_count": n}
        print(f"[supervisor] finance ran -> done")
        return {"route": "done", "iteration_count": n}

    if "credit" in agents_run:
        # Same HITL gate as finance's invoice check, applied to a loan
        # proposal instead of a purchase invoice -- see
        # orchestration/confirm_loan.py. loan_confirmed is set True by
        # confirm_loan_node once the user approves THIS turn.
        last_credit = next((e for e in reversed(scratchpad) if e.get("agent") == "credit"), None)
        result_text = str(last_credit.get("result", "")) if last_credit else ""
        if "Loan Proposal #" in result_text and not state.get("loan_confirmed"):
            print(f"[supervisor] credit produced a loan proposal -> confirm_loan")
            return {"route": "confirm_loan", "iteration_count": n}
        print(f"[supervisor] credit ran -> done")
        return {"route": "done", "iteration_count": n}

    if "risk" in agents_run:
        print(f"[supervisor] risk ran -> done")
        return {"route": "done", "iteration_count": n}

    if "forecast" in agents_run:
        print(f"[supervisor] forecast ran -> done")
        return {"route": "done", "iteration_count": n}

    if len(agents_run) != len(set(agents_run)):
        print(f"[supervisor] duplicate agent -> done")
        return {"route": "done", "iteration_count": n}

    if n == 1 and not agents_run and _GREETING_RE.match(question.strip()):
        print(f"[supervisor] greeting/meta-chat pre-filter -> done (no LLM call)")
        return {"route": "done", "iteration_count": n}

    # ── Verifier: has this question already been fully answered? ─────────────
    # Only relevant once something has actually run (agents_run non-empty --
    # on a fresh question there's nothing to verify yet) and only for the
    # agents that DON'T already have their own hard "always done" stop above
    # (finance/credit/risk/forecast) -- by construction, reaching this point
    # with agents_run non-empty means only sql/rag/research have run so far,
    # which is exactly the category that used to fall through to the router
    # LLM every subsequent turn with no sufficiency check at all. A cheap,
    # dedicated yes/no call here (see _is_answer_sufficient's own docstring
    # for why this is a more reliable judgment than multi-way routing)
    # short-circuits straight to "done" instead of letting the router guess
    # at whether a second agent is still needed.
    if agents_run:
        scratchpad_text_for_verify = _build_scratchpad(state)
        # max_tokens caps the RESERVED completion budget Groq counts toward
        # the TPM limit, not just actual output length -- left unset
        # (None), Groq reserves the model's full max output allowance
        # regardless of how short the real answer is. Confirmed live: this
        # was the actual cause of repeated 413 "rate_limit_exceeded" errors
        # this session across multiple agents, not genuine traffic volume
        # -- a one-word YES/NO or routing answer was still being billed
        # against a multi-thousand-token reservation. 10 is generous for a
        # single word.
        verifier_llm = ChatGroq(model=_ROUTER_MODEL, temperature=0, max_tokens=10, max_retries=0)
        if _is_answer_sufficient(question, scratchpad_text_for_verify, verifier_llm):
            print(f"[supervisor] verifier: question already fully answered -> done")
            return {"route": "done", "iteration_count": n}

    # ── Fast path: regex + semantic embedding pre-filter (no LLM call) ────────
    # Only on a genuinely fresh question -- first iteration, nothing run yet,
    # no PDF override in play (has_pdf always forces rag regardless of what
    # this layer says, so skip it entirely rather than compute a wasted answer)
    # -- AND not one of our own suggested follow-up questions. Those are
    # deliberately context-dependent ("...instead?", "...one of these?") --
    # a real bug this guards against: clicking "Want an invoice for one of
    # these?" was matching the finance regex/semantic layer on "invoice" and
    # running finance with no product specified at all (silently defaulting
    # to PRD001), instead of the LLM router resolving "these" from history.
    route = None
    if n == 1 and not agents_run and not has_pdf and question.strip() not in followup.ALL_FOLLOWUPS:
        route = fast_route(question)
        if route:
            print(f"[supervisor] fast-path pre-filter -> route={route} (no LLM call)")

    # ── Build context ────────────────────────────────────────────────────────
    history         = _build_history(state)
    scratchpad_text = _build_scratchpad(state)

    # ── LLM Call: Route the question (skipped if the fast path was confident) ─
    if route is None:
        llm   = ChatGroq(model=_ROUTER_MODEL, temperature=0, max_tokens=10, max_retries=0)  # one-word route, see verifier_llm's comment above
        route = _route_question(
            question=question,
            history=history,
            scratchpad=scratchpad_text,
            agents_run=agents_run,
            has_pdf=has_pdf,
            iteration=n,
            llm=llm,
        )

    # ── Override: PDF active -> never SQL ────────────────────────────────────
    if route == "sql" and has_pdf:
        print(f"[supervisor] override: SQL blocked (PDF active) -> rag")
        route = "rag"

    # ── Override: research re-picked after already running on a question
    # that also asked to compare against TechMart's OWN data -> redirect to
    # sql instead of letting the "already ran" override below collapse this
    # straight to "done" with only the external half of the question answered.
    if (
        route == "research" and "research" in agents_run
        and "sql" not in agents_run and "finance" not in agents_run
        and _COMPARE_OWN_DATA_RE.search(question)
    ):
        # finance, not sql: a generic NL->SQL query has no idea what
        # "market trends" maps to as a TechMart column/metric and comes
        # back empty or useless (verified live). finance's
        # monthly_trend_analysis tool is a purpose-built "how are we doing
        # overall" summary -- an actual comparable figure, not a guess.
        print(f"[supervisor] override: research re-picked for an our-data comparison -> finance")
        route = "finance"

    # ── Override: sql already gave a complete, self-contained answer, but the
    # router wants to chain into finance/credit/risk/forecast anyway -> trust
    # sql's answer instead. Real bug this guards against: a plain "how many
    # transactions last month" question gets fully answered by sql (a bare
    # count, e.g. "total_transactions\n4"), but the router over-applies rule
    # 8's "sql then finance if a calculation is still needed on top" to a
    # question that needed no further calculation at all -- the LLM
    # (llama-3.1-8b-instant) doesn't reliably judge this distinction from
    # prompt wording alone (confirmed live, same class of miss as the
    # research-repick case above). Narrowly scoped to a SINGLE-VALUE sql
    # result (one header line + one value line, no commas) -- genuinely
    # multi-row/multi-column results (e.g. a customer's per-category
    # spending breakdown) are rule 8's real, legitimate use case and must
    # still be allowed to chain into finance for a calculation on top.
    if (
        route in ("finance", "credit", "risk", "forecast")
        and "sql" in agents_run and route not in agents_run
    ):
        last_sql  = next((e for e in reversed(scratchpad) if e.get("agent") == "sql"), None)
        sql_lines = [l for l in str(last_sql.get("result", "")).split("\n") if l.strip()] if last_sql else []
        if len(sql_lines) <= 2 and "," not in (sql_lines[0] if sql_lines else ""):
            print(f"[supervisor] override: sql already gave a complete single-value answer -> done")
            route = "done"

    # ── Override: agent already ran -> done ─────────────────────────────────
    if route in agents_run:
        print(f"[supervisor] override: {route} already ran -> done")
        route = "done"

    print(f"[supervisor] turn={n} -> route={route}")

    return {"route": route, "iteration_count": n}


def _route(state: AgentState) -> str:
    return state.get("route", "done")
