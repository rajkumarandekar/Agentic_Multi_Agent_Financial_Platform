"""
FastAPI app — Phase 2: supervisor graph endpoint added alongside Phase 1 RAG endpoint.

Start the server:
    uvicorn api.main:app --reload
"""

import logging
import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate required key at startup so the error is obvious immediately.
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

# Import after env validation so LangChain clients don't error on import.
from agents.rag_agent import ask          # noqa: E402  (Phase 1)
from orchestration.graph import graph     # noqa: E402  (Phase 2)
from orchestration.state import AgentState  # noqa: E402

app = FastAPI(
    title="Agentic AI Platform",
    description="Phase 3 — Guardrails (input + output) + eval harness",
    version="0.3.0",
)


# ---------------------------------------------------------------------------
# Phase 1 — RAG endpoint (unchanged)
# ---------------------------------------------------------------------------

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

    The ChromaDB collection must be populated first:
        python -m ingestion.pdf_ingest path/to/file.pdf
    """
    try:
        result = ask(question=body.question, k=body.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )


# ---------------------------------------------------------------------------
# Phase 2 — Supervisor graph endpoint
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to route and answer")


class AgentResponse(BaseModel):
    agent_used: str
    answer: str
    sources: list[SourceItem]
    guardrails: dict


@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(body: AgentRequest) -> AgentResponse:
    """
    Route the question through the LangGraph supervisor to the appropriate agent.

    Phase 3 flow: input_guard → supervisor → agent → output_guard

    Guardrails applied automatically:
      - Input:  injection-pattern regex + LLM safety classifier
      - Output: PII masking (email/phone/SSN/card) + toxicity classifier
      The 'guardrails' field in the response shows pass/fail per check.

    Routing:
      - rag  → ChromaDB retrieval + Groq answer
      - sql  → SQLite shipments query + Groq answer
      - tool → native file-list / calculator tools + Groq answer
    """
    initial_state = AgentState(
        question=body.question,
        messages=[HumanMessage(content=body.question)],
        route="",
        answer="",
        sources=[],
        agent_used="",
        guardrail_results={},
    )
    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        # Log full traceback to the server console for diagnosis.
        logger.error("agent_endpoint failed:\n%s", traceback.format_exc())
        # Include exception type so detail is never an empty string.
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        raise HTTPException(status_code=500, detail=detail) from exc

    return AgentResponse(
        agent_used=result["agent_used"],
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result.get("sources", [])],
        guardrails=result.get("guardrail_results", {}),
    )
