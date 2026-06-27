"""
FastAPI app — Phase 1: single RAG endpoint.

Start the server:
    uvicorn api.main:app --reload
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# Validate required keys at startup so failures are obvious immediately.
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

# Import after env validation so LangChain clients don't error on import.
from agents.rag_agent import ask  # noqa: E402

app = FastAPI(
    title="Agentic AI Platform",
    description="Phase 1 — RAG agent over ingested PDFs",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")


class SourceItem(BaseModel):
    page: int | str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    """
    Retrieve relevant document chunks and generate an answer with the Groq LLM.

    The ChromaDB collection must be populated first by running:
        python -m ingestion.pdf_ingest path/to/file.pdf
    """
    try:
        result = ask(question=body.question, k=body.k)
    except Exception as exc:
        # Surface the real error message to make debugging easy during development.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )
