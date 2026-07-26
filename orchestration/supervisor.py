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

_MODEL          = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_MAX_ITERATIONS = 3
_VALID_ROUTES   = {"sql", "rag", "finance", "research", "clarify", "done"}

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

_ROUTER_SYSTEM_PROMPT = """You are a routing agent for TechMart India AI.

TechMart India context:
- 20 products: PRD001 (Laptop), PRD002 (Wireless Earbuds), PRD003 (Smartphone),
  PRD004 (Tablet), PRD005 (Cotton Shirt), PRD006 (Denim Jeans), PRD007 (Kurta Set),
  PRD008 (Running Shoes), PRD009 (Air Purifier), PRD010 (Mixer Grinder),
  PRD011 (Pressure Cooker), PRD012 (Water Purifier), PRD013 (Face Serum),
  PRD014 (Sunscreen SPF50), PRD015 (Hair Oil Set), PRD016 (Skincare Kit),
  PRD017 (Yoga Mat), PRD018 (Dumbbells Set), PRD019 (Cricket Bat), PRD020 (Fitness Tracker)
- 10 customers: CUST001-CUST010 with Bronze/Silver/Gold/Platinum tiers

Decide which specialist agent should handle the CURRENT question. Use the
conversation history to resolve vague or short messages ("it", "that", "what
about PRD005", "I asked X not Y") — the current question may only make sense
in light of what was discussed before.

AGENTS:
- finance: ANY calculation, pricing, discount, bulk quote, loyalty price,
           profit margin, GST analysis, revenue forecast (ARIMA model),
           churn risk prediction (ML model), product comparison by price/margin,
           most expensive/cheapest product, restock alerts
- sql: fetching raw data -- show/list/count customers, products, transactions,
       which customers in a city, how many orders, transaction history
- rag: questions about an uploaded PDF document
- research: external market data, industry benchmarks, web information
- done: question already answered in scratchpad, casual greeting, or a
        question about the CONVERSATION ITSELF (e.g. "summarise what we
        discussed", "what did I ask before") -- these have no TechMart data
        to look up, so finance/sql have nothing to do with them; route to
        done and let the chat reply use conversation history directly

STRICT RULES:
1. PDF active -> always rag first (never sql when PDF is selected)
2. Pricing/calculation -> always finance (never sql for prices)
3. Any request to see TechMart's own raw data (customers, products,
   transactions) -> sql, however it's phrased -- "show", "list", "count",
   "how many", "give me", "gimme", "what products/customers do you have",
   "I need/want the list of...". These all mean the same thing: fetch and
   show the real rows. Casual phrasing ("give me products u have") is NOT a
   question about the assistant's own capabilities -- it names a real
   TechMart entity (products/customers/transactions), so it is never `done`.
4. Never call same agent twice
5. If finance already ran -> done
6. A bare list of numbers or product/customer ids (e.g. "12,3,1,4,5,7",
   "prod1, 4, 6" or a follow-up like "what about those?") asking for
   cost/price/average/GST/discount/margin/total is a FINANCE question, even
   with no other keywords -- it is never sql or research. Route straight to
   finance; do not try sql or research first "just in case."
7. If the conversation history shows finance was just discussing
   pricing/GST/products, and the current question is short or numbers-only
   with no new topic, it is almost always a continuation of that SAME
   finance thread -- prefer finance over sql or research.
8. A question asking to analyse/breakdown a CUSTOMER's actual spending,
   purchases, or transaction history (by category, over time, by product)
   -- e.g. "analyse CUST001's spending by category" -- needs REAL
   transaction data finance does not have access to. Route to sql FIRST to
   fetch that data, THEN finance if a calculation is still needed on top of
   it. Never answer this kind of question from finance alone -- finance
   only knows product/customer master data (price, tier, base stats), not
   a customer's actual per-category purchase breakdown.
9. If iteration >= 3 -> done
10. Greetings (hi, hello, thanks, bye) -> done
11. A question about the conversation itself ("summarise everything we
    discussed", "what did I ask before") -> done, never finance/sql. There
    is no TechMart data to fetch for it -- forcing finance to answer this
    makes it pick an unrelated tool (e.g. a revenue forecast) just to have
    something to call, which is worse than a plain conversational reply.
12. A short reply with no keywords of its own -- an affirmation ("yes",
    "ok", "sure", "go ahead", "do it", "please resume", "continue") OR a
    short direct answer to a question that was just asked (a bare product/
    customer name or ID, a number, "prod5") -- is NEVER routed to done by
    itself. FIRST check the conversation history for a line starting with
    "[Note: your previous answer ended with this pending question...]" -- if
    present, that IS what the current reply is about, whether it's accepting
    an offer or answering the question directly. Combine the pending
    question with the current reply to figure out the real request (e.g.
    pending question "which product and quantity for the invoice?" + current
    reply "prod5" = generate an invoice for PRD005), and route to whichever
    agent that combined request needs (profit margin/GST/discount/invoice ->
    finance, showing/listing data -> sql, etc.). If no such note is present,
    fall back to whatever the assistant's previous message actually offered
    or described.

Reply with EXACTLY ONE WORD: sql | rag | finance | research | clarify | done"""


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

    if len(agents_run) != len(set(agents_run)):
        print(f"[supervisor] duplicate agent -> done")
        return {"route": "done", "iteration_count": n}

    if n == 1 and not agents_run and _GREETING_RE.match(question.strip()):
        print(f"[supervisor] greeting/meta-chat pre-filter -> done (no LLM call)")
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
        llm   = ChatGroq(model=_MODEL, temperature=0)
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

    # ── Override: agent already ran -> done ─────────────────────────────────
    if route in agents_run:
        print(f"[supervisor] override: {route} already ran -> done")
        route = "done"

    print(f"[supervisor] turn={n} -> route={route}")

    return {"route": route, "iteration_count": n}


def _route(state: AgentState) -> str:
    return state.get("route", "done")
