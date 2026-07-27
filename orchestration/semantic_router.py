"""
semantic_router.py — regex + embedding fast-path pre-filter for supervisor routing.

Supersedes the "routing decisions stay fully LLM-driven, no regex" note in
supervisor.py's old docstring (that decision covered ONLY the greeting
pre-filter). This module adds two more zero/low-cost layers BEFORE the LLM
router, in cheapest-first order:

  1. regex_route()    — near-zero cost, exact keyword/phrase patterns for the
                         most unambiguous phrasings ("show me all customers").
  2. semantic_route()  — cosine similarity against example utterances per
                         route, using the HuggingFace embedding model already
                         in this project's stack (ingestion/pdf_ingest.py) --
                         no new dependency. Catches paraphrases regex misses
                         ("what's on offer for laptops" ~ sql). Requires the
                         top route to both clear an absolute threshold AND
                         beat the runner-up route by a clear margin -- a
                         confident-looking top score that's nearly tied with
                         a second plausible route is exactly the ambiguous
                         case this must NOT guess on (see _MARGIN_THRESHOLD).

Both are ONLY a pre-filter, never a hard decision on their own: fast_route()
returns None whenever neither layer is confident, and the caller
(supervisor_node) falls through to the existing LLM router, which alone has
access to conversation history/scratchpad to resolve context-dependent or
ambiguous questions. Callers must only invoke fast_route() on a fresh
question (first supervisor iteration, no agents run yet, no PDF override in
play) -- exactly the same restriction supervisor.py's _GREETING_RE already
uses, for the same reason: a follow-up like "what about that one" must never
be fast-pathed on keyword/embedding similarity alone.
"""
import os
import re

import numpy as np
from dotenv import load_dotenv

# Previously unnecessary (HuggingFace's local model needs no API key), but
# the temporary Gemini embedding swap below reads GOOGLE_API_KEY from the
# environment -- this module can no longer silently rely on some other
# already-imported module having called load_dotenv() first.
load_dotenv()

_VALID_FAST_ROUTES = {"sql", "finance", "credit", "risk", "forecast", "research"}

# ── Layer 1: regex ────────────────────────────────────────────────────────────
# Deliberately narrow and anchored to unambiguous keyword combinations only --
# a miss falls through to semantic_route() or the LLM, so false negatives are
# cheap; a false positive would silently misroute, so precision > recall here.

_SQL_RE = re.compile(
    r'\b(show|list|display|count|how many|give\s*me|gimme|i\s*(need|want))\b.*\b'
    r'(customers?|products?|transactions?|orders?|records?)\b',
    re.IGNORECASE,
)

_FINANCE_RE = re.compile(
    r'\b(price|pricing|discount|margin|gst|quote|profit|invoice|bulk\s*order|'
    r'loyalty|cheapest|most\s+expensive)\b',
    re.IGNORECASE,
)

# churn/forecast/loan used to be finance keywords -- now dedicated agents
# (Phase 2 of the multi-agent expansion), checked BEFORE _FINANCE_RE so a
# question like "is this customer at risk" never falls through to finance
# just because a generic word also matches nearby.
_CREDIT_RE = re.compile(
    r'\b(loan|emi|installment|credit\s+eligib\w*|credit\s+limit|available\s+credit|'
    r'collections?\s+priorit\w*|overdue\s+account\w*|'
    r'(who|which\s+customers?)\s+should\s+we\s+chase)\b',
    re.IGNORECASE,
)

_RISK_RE = re.compile(
    r'\b(churn\w*|at\s+risk|retention|fraud|suspicious|anomaly|anomalous)\b',
    re.IGNORECASE,
)

_FORECAST_RE = re.compile(
    r'\b(forecast|predict(ed)?\s+revenue|future\s+revenue|revenue\s+(next|for))\b',
    re.IGNORECASE,
)

_RESEARCH_RE = re.compile(
    r'\b(market\s+trends?|industry\s+benchmarks?|competitors?|external\s+market|'
    r'industry\s+data|industry\s+reports?)\b',
    re.IGNORECASE,
)


def regex_route(question: str) -> str | None:
    """Return 'sql' | 'finance' | 'credit' | 'risk' | 'forecast' | 'research'
    | None (no confident match)."""
    q = question.strip()
    # research checked first: its keywords (competitor, industry benchmark,
    # market trend) are specific enough that they should win even when a
    # generic finance word like "price" also appears in the same sentence
    # (e.g. "how do competitors price similar products").
    if _RESEARCH_RE.search(q):
        return "research"
    # credit/risk/forecast checked before finance -- see comment above.
    if _CREDIT_RE.search(q):
        return "credit"
    if _RISK_RE.search(q):
        return "risk"
    if _FORECAST_RE.search(q):
        return "forecast"
    if _FINANCE_RE.search(q):
        return "finance"
    if _SQL_RE.search(q):
        return "sql"
    return None


# ── Layer 2: semantic (embedding similarity) ─────────────────────────────────

_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
_GOOGLE_EMBED_MODEL_NAME = os.getenv("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")
_SIMILARITY_THRESHOLD = 0.55

_ROUTE_EXAMPLES: dict[str, list[str]] = {
    "sql": [
        "what products do you have",
        "which customers live in Chennai",
        "how many transactions happened last month",
        "list all orders for CUST001",
        "show me the transaction history",
        "give me the full customer list",
    ],
    "finance": [
        "what is the profit margin on this product",
        "calculate the GST for this order",
        "give me a bulk quote for 50 units",
        "what discount does a gold tier customer get",
        "what's the loyalty price for this item",
    ],
    "credit": [
        "can this customer apply for a loan",
        "what's the EMI on a 50000 rupee loan",
        "check credit eligibility for this customer",
        "how much available credit does this customer have",
        "which customers should we chase for overdue payments",
    ],
    "risk": [
        "is this customer at risk of churning",
        "will this customer stop buying from us",
        "flag any suspicious transactions for this customer",
        "check for fraud on this account",
    ],
    "forecast": [
        "forecast revenue for the next 3 months",
        "predict next week's revenue",
        "what will our sales look like next year",
        "revenue forecast for the next 10 days",
    ],
    "research": [
        "what are the industry benchmarks for this",
        "how does this compare to market trends",
        "what are competitors charging for similar products",
        "any external market data on this category",
    ],
}

# Lazy singletons — never load the embedding model at import time (keeps
# this module importable in tests/scripts that never touch routing).
_embeddings = None
_route_vectors: dict[str, np.ndarray] | None = None


def _get_embeddings():
    """
    TEMPORARY: using Gemini embeddings instead of HuggingFace as an
    experiment -- see ingestion/pdf_ingest.py's _get_embeddings() for the
    full rationale/revert note. Same swap here: HuggingFace import/line
    commented out, not deleted.
    """
    global _embeddings
    if _embeddings is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        _embeddings = GoogleGenerativeAIEmbeddings(model=_GOOGLE_EMBED_MODEL_NAME)
        # from langchain_huggingface import HuggingFaceEmbeddings
        # _embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL_NAME)
    return _embeddings


def _get_route_vectors() -> dict[str, np.ndarray]:
    global _route_vectors
    if _route_vectors is None:
        emb = _get_embeddings()
        _route_vectors = {
            route: np.array(emb.embed_documents(examples))
            for route, examples in _ROUTE_EXAMPLES.items()
        }
    return _route_vectors


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return a_norm @ b_norm


# A confident-looking top score isn't enough on its own: "which customers get
# a discount" can score reasonably high against BOTH the sql and finance
# example sets. Requiring the top route to also beat the runner-up by a
# clear margin catches exactly that ambiguous case and defers it to the LLM
# router (which has conversation history to actually disambiguate it),
# instead of confidently guessing wrong on a coin-flip between two routes.
_MARGIN_THRESHOLD = 0.10


def semantic_route(question: str) -> tuple[str | None, float]:
    """
    Return (route, best_score). route is None -- caller must fall through to
    the LLM router -- unless BOTH hold:
      1. best_score >= _SIMILARITY_THRESHOLD (looks like a real match), and
      2. best_score beats the second-best route's score by >= _MARGIN_THRESHOLD
         (isn't a near-tie between two plausible routes).
    """
    vectors  = _get_route_vectors()
    q_vector = np.array(_get_embeddings().embed_query(question))

    scores = {
        route: float(np.max(_cosine_sim(examples, q_vector)))
        for route, examples in vectors.items()
    }
    ranked                    = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_route, best_score    = ranked[0]
    second_score              = ranked[1][1] if len(ranked) > 1 else -1.0
    margin                    = best_score - second_score

    if best_score >= _SIMILARITY_THRESHOLD and margin >= _MARGIN_THRESHOLD:
        return best_route, best_score

    if best_score >= _SIMILARITY_THRESHOLD:
        print(f"[semantic_router] rejected {best_route!r} (score={best_score:.2f}, "
              f"margin={margin:.2f} < {_MARGIN_THRESHOLD}) -> no fast route")
    return None, best_score


# ── Combined fast path ────────────────────────────────────────────────────────

def fast_route(question: str) -> str | None:
    """
    Try regex first (cheapest, highest precision), then semantic similarity.
    Returns a route in _VALID_FAST_ROUTES, or None if neither layer is
    confident -- caller must fall through to the LLM router.
    """
    route = regex_route(question)
    if route:
        return route

    try:
        route, _score = semantic_route(question)
    except Exception as exc:
        # Embedding model failed to load/run -- degrade to "no opinion",
        # same fallback contract as a failed LLM router call in supervisor.py.
        print(f"[semantic_router] semantic_route failed ({exc}) -> no fast route")
        return None

    return route
