"""
RAG agent: retrieve top-k chunks from ChromaDB, then answer with Groq LLM.

No document selected → top 8 chunks across ALL docs, synthesis prompt.
Document selected    → top 4 chunks from that doc, focused prompt.
Both prompts enforce concise responses (max 5 bullet points).
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from ingestion.pdf_ingest import get_retriever

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K      = int(os.getenv("TOP_K", "4"))

_CONCISE = ("Be concise. Use at most 5 bullet points or a short paragraph. "
            "No lengthy explanations.")

_SINGLE_DOC_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "passages provided below. If the answer is not in the context, say "
    "\"I don't have enough information in the provided document to answer that.\" "
    + _CONCISE
)

_MULTI_DOC_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "passages provided below, which may come from multiple uploaded documents. "
    "Answer based on ALL uploaded documents. Synthesize information across "
    "documents where relevant. If the answer is not in the context, say "
    "\"I don't have enough information in the uploaded documents to answer that.\" "
    + _CONCISE
)


class RAGResult(TypedDict):
    answer: str
    sources: list[dict]


def _build_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        page   = doc.metadata.get("page", "?")
        src    = doc.metadata.get("source_document", "")
        label  = f"[Passage {i} | {src} p.{page}]" if src else f"[Passage {i} | p.{page}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(parts)


def ask(
    question: str,
    k: int = TOP_K,
    source_document: str | None = None,
) -> RAGResult:
    """
    Retrieve relevant chunks and generate an answer.

    Args:
        question:        User question (may include conversation context).
        k:               Base chunk count (overridden to 8 when no filter).
        source_document: Restrict to this filename, or None for all docs.
    """
    doc_filter = source_document if source_document and source_document != "all" else None
    print(f"[rag] source_document filter: {repr(doc_filter)}")

    effective_k   = k if doc_filter else max(k, 8)
    system_prompt = _SINGLE_DOC_PROMPT if doc_filter else _MULTI_DOC_PROMPT

    retriever  = get_retriever(k=effective_k, source_document=doc_filter)
    docs       = retriever.invoke(question)
    context    = _build_context(docs)

    # max_tokens caps the RESERVED completion budget Groq counts toward its
    # TPM rate limit -- see agents/sql_agent.py's identical comment.
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=1024, max_retries=1)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ])

    sources = [
        {
            "page":            doc.metadata.get("page", "?"),
            "snippet":         doc.page_content[:200],
            "source_document": doc.metadata.get("source_document", ""),
        }
        for doc in docs
    ]
    return RAGResult(answer=response.content, sources=sources)
