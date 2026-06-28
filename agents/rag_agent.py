"""
RAG agent: retrieve top-k chunks from ChromaDB, then answer with Groq LLM.
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from ingestion.pdf_ingest import get_retriever

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the context passages \
provided below. If the answer is not in the context, say "I don't have enough \
information in the provided document to answer that."
"""


class RAGResult(TypedDict):
    answer: str
    sources: list[dict]


def _build_context(docs: list) -> str:
    """Concatenate retrieved chunks into a single context string."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[Passage {i} | page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def ask(
    question: str,
    k: int = TOP_K,
    source_document: str | None = None,
) -> RAGResult:
    """
    Retrieve relevant chunks and generate an answer.

    Args:
        question:        The user's question (may include conversation context).
        k:               Number of chunks to retrieve.
        source_document: If set, restrict retrieval to chunks from this file only.

    Returns:
        RAGResult with 'answer' (str) and 'sources' (list of dicts with
        keys 'page', 'snippet', and 'source_document').
    """
    # 1. Retrieve — similarity search returns the k most relevant Document objects.
    retriever = get_retriever(k=k, source_document=source_document)
    docs = retriever.invoke(question)

    # 2. Build context — join chunks so the LLM sees them all in one prompt.
    context = _build_context(docs)

    # 3. Call Groq LLM — temperature=0 for deterministic, factual answers.
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)

    # 4. Build sources list for the API response.
    sources = [
        {
            "page":            doc.metadata.get("page", "?"),
            "snippet":         doc.page_content[:200],
            "source_document": doc.metadata.get("source_document", ""),
        }
        for doc in docs
    ]

    return RAGResult(answer=response.content, sources=sources)
