"""
Knowledge Agent: orchestrates RAG and SQL internally.
Returns structured data — never generates the final response.

Callers: graph.py knowledge_node (passes use_rag/use_sql from planner)
Imports: rag_agent.py, sql_agent.py  (unchanged)

Retrieval selection (in priority order):
  1. use_rag / use_sql provided by Planner → use those values directly
  2. Neither provided (direct call) → fall back to own regex detection
"""

import re

from agents.rag_agent import ask as _rag_ask
from agents.sql_agent import run as _sql_run

# Fallback patterns — only used when called without planner flags (e.g., tests)
_SQL_FALLBACK_RE = re.compile(
    r"transaction|expense[s]?|spent|spending|customer|CUST\d*|"
    r"invoice|INV\d*|order|receipt|payment|purchase|"
    r"merchant|category|balance|list\s+(my|all)|"
    r"how\s+much\s+(did|have\s+i)|what\s+did\s+i\s+spend",
    re.IGNORECASE,
)

_RAG_FALLBACK_RE = re.compile(
    r"pdf|document|file|upload|according\s+to|"
    r"summari[sz]|what\s+does\s+it\s+say|from\s+the\s+(doc|file|pdf)",
    re.IGNORECASE,
)

_CUST_RE = re.compile(r"\bCUST\d+\b", re.IGNORECASE)


def run(
    question: str,
    source_document: str = "",
    retrieval_query: str = "",
    use_rag: bool | None = None,
    use_sql: bool | None = None,
) -> dict:
    """
    Call the retrieval sources specified by the Planner, then return structured data.

    Args:
        question:         Full question (may include conversation history) — used for SQL.
        source_document:  Restrict RAG to this filename when set.
        retrieval_query:  Clean current question for RAG embedding (no history).
                          Defaults to question when omitted.
        use_rag:          Whether to run ChromaDB / RAG retrieval.
                          None = detect from question text (backward-compat fallback).
        use_sql:          Whether to run SQL retrieval against transactions DB.
                          None = detect from question text (backward-compat fallback).

    Returns:
        {
            "documents":    str,   # RAG answer (empty string when RAG not executed)
            "transactions": str,   # SQL raw rows (empty string when SQL not executed)
            "customer":     str,   # extracted customer ID e.g. "CUST001"
            "facts":        list,  # RAG source chunk metadata
            "sources":      list,  # same as facts, for state["sources"]
            "rag_executed": bool,  # whether RAG was actually called
            "sql_executed": bool,  # whether SQL was actually called
        }
    """
    result: dict = {
        "documents": "", "transactions": "",
        "customer": "", "facts": [], "sources": [],
        "rag_executed": False, "sql_executed": False,
    }

    # Use clean question for RAG so history tokens don't pollute embeddings
    rag_query = retrieval_query or question

    # Resolve retrieval flags: planner value → fallback detection
    if use_sql is None:
        need_sql = bool(_SQL_FALLBACK_RE.search(question))
    else:
        need_sql = use_sql

    if use_rag is None:
        need_rag = bool(_RAG_FALLBACK_RE.search(rag_query)) or bool(source_document)
    else:
        need_rag = use_rag

    # Extract customer ID if present (used by response agent for personalisation)
    cust = _CUST_RE.search(question)
    if cust:
        result["customer"] = cust.group(0).upper()

    print(
        f"[knowledge] need_rag={need_rag} need_sql={need_sql}"
        f" source_doc={source_document!r}"
        f" (planner_flags: use_rag={use_rag} use_sql={use_sql})"
    )

    if need_sql:
        try:
            result["transactions"] = _sql_run(question)
            result["sql_executed"] = True
        except Exception as exc:
            result["transactions"] = f"SQL error: {exc}"

    if need_rag:
        try:
            rag_out = _rag_ask(rag_query, source_document=source_document or None)
            result["documents"]    = rag_out["answer"]
            result["facts"]        = rag_out.get("sources", [])
            result["sources"]      = result["facts"]
            result["rag_executed"] = True
        except Exception as exc:
            result["documents"] = f"RAG error: {exc}"

    return result
