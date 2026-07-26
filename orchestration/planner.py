"""
Planner: inspects the question and returns an execution plan.

Returns:
  {
    "knowledge": bool,  — True when any retrieval is needed (use_rag OR use_sql)
    "finance":   bool,  — True when Finance/Tool Agent should run
    "use_rag":   bool,  — Within Knowledge: query the uploaded PDF via ChromaDB
    "use_sql":   bool,  — Within Knowledge: query the transactions DB via SQL
  }

Zero LLM calls. Pure regex on the question text plus the active source_document.

Decision rules (in priority order):
  1. Both RAG and SQL signals present → run both
  2. Only RAG signal present           → RAG only
  3. Only SQL signal present           → SQL only
  4. source_document set, no signals   → RAG only (user is asking about their doc)
  5. No signals, no document           → neither (no knowledge needed)
"""

import re

# ── Signals that the user wants to query the uploaded PDF / document ──────────
_RAG_RE = re.compile(
    r"pdf|document|file|upload|according\s+to|summari[sz]|"
    r"what\s+does\s+it\s+say|from\s+the\s+(doc|file|pdf)|"
    r"in\s+the\s+(document|pdf|file)|"
    r"the\s+(document|pdf|file)\s+(says|mentions|states|shows|contains)",
    re.IGNORECASE,
)

# ── Signals that the user wants to query the structured transactions database ─
_SQL_RE = re.compile(
    r"transaction|expense[s]?|spent|spending|customer|CUST\d*|"
    r"invoice|INV\d*|order|receipt|payment|purchase|"
    r"merchant|category|balance|"
    r"list\s+(my|all)|"
    r"how\s+much\s+(did|have\s+i)|what\s+did\s+i\s+spend",
    re.IGNORECASE,
)

# ── Finance / calculation signals (independent of retrieval source) ────────────
_FINANCE_RE = re.compile(
    r"analys[ei]|health\s+score|financial\s+health|"
    r"forecast|predict|next\s+month|am\s+i\s+on\s+track|"
    r"calcul|cagr|roi|emi|compound|interest|tax|"
    r"investment|compare\s+(fd|sip)|budget|goal|"
    r"savings?\s+rate|recommend|how\s+am\s+i\s+doing|"
    r"anomaly|unusual\s+spend|where\s+did\s+my\s+money",
    re.IGNORECASE,
)


def plan(question: str, source_document: str = "") -> dict:
    """
    Return which agents and retrieval paths are needed for this question.

    The Knowledge Agent is told exactly which sources to query:
      use_rag=True  → query ChromaDB with the user's uploaded PDF
      use_sql=True  → query the transactions SQLite database

    When both signals are present the Knowledge Agent runs both sources
    so the Response Agent can synthesise across them.
    """
    has_rag = bool(_RAG_RE.search(question))
    has_sql = bool(_SQL_RE.search(question))
    has_doc = bool(source_document)
    needs_finance = bool(_FINANCE_RE.search(question))

    if has_rag and has_sql:
        # Explicit signals for both sources — user wants cross-source synthesis
        use_rag, use_sql = True, True
    elif has_rag:
        use_rag, use_sql = True, False
    elif has_sql:
        use_rag, use_sql = False, True
    elif has_doc:
        # Document is active but no explicit retrieval keyword: user is asking
        # about the document they uploaded, so default to RAG.
        use_rag, use_sql = True, False
    else:
        use_rag, use_sql = False, False

    needs_knowledge = use_rag or use_sql

    print(
        f"[planner] knowledge={needs_knowledge} use_rag={use_rag} use_sql={use_sql}"
        f" finance={needs_finance} source_doc={source_document!r}"
    )
    return {
        "knowledge": needs_knowledge,
        "finance":   needs_finance,
        "use_rag":   use_rag,
        "use_sql":   use_sql,
    }
