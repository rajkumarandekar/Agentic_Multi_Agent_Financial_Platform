"""
FastAPI app — Phase 4: structured logging, in-process metrics, dashboard, LangSmith.

Start the server:
    uvicorn api.main:app --reload
"""

import logging
import os
import time
import traceback
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# --- Phase 4: configure JSON logging before anything else emits log lines ---
from observability.logger import configure_logging, new_request_id
configure_logging()

logger = logging.getLogger(__name__)

load_dotenv()

# Validate required key at startup so the error is obvious immediately.
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

# --- Phase 4: LangSmith must be set up BEFORE importing the graph so that
#     LangChain picks up LANGCHAIN_TRACING_V2 at import time. ---
from observability.langsmith_setup import setup_langsmith   # noqa: E402
setup_langsmith()

# Import agents and graph after env is ready.
from agents.rag_agent import ask                            # noqa: E402
from orchestration.graph import graph                       # noqa: E402
from orchestration.state import AgentState                  # noqa: E402
from observability.metrics import (                         # noqa: E402
    ObservabilityCallback, record_request, aggregate,
)
from observability.dashboard import DASHBOARD_HTML          # noqa: E402
from ingestion.pdf_ingest import ingest as pdf_ingest, list_documents  # noqa: E402

app = FastAPI(
    title="Agentic AI Platform",
    description="Phase 4.5 — Multi-turn memory, React frontend, PDF upload",
    version="0.5.0",
)

# Allow the React dev server (Vite default port 5173) to call the API.
# In production this list should be tightened to the actual frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
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
# Phase 2/3 — Supervisor graph endpoint (Phase 4: metrics + logging added)
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to route and answer")
    session_id: str | None = Field(None, description="Conversation session ID; generated if omitted")
    source_document: str | None = Field(None, description="Restrict RAG to this filename; omit to search all")


class AgentResponse(BaseModel):
    agent_used: str
    answer: str
    sources: list[SourceItem]
    guardrails: dict
    session_id: str  # echo back so the client can track the session


@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(body: AgentRequest) -> AgentResponse:
    """
    Route the question through the LangGraph supervisor to the appropriate agent.

    Phase 3 flow: input_guard → supervisor → agent → output_guard
    Phase 4 adds: structured request logging, per-stage timing, token counting.

    Guardrails:
      - Input:  injection-pattern regex + LLM safety classifier
      - Output: PII masking + toxicity classifier
    Routing:
      - rag  → ChromaDB retrieval + Groq
      - sql  → SQLite shipments query + Groq
      - tool → native file-list / calculator + Groq
    """
    # Use provided session_id or generate a new one for this conversation.
    session_id = body.session_id or str(uuid.uuid4())

    request_id = new_request_id()
    cb = ObservabilityCallback(request_id)
    t_start = time.perf_counter()

    logger.info("request_start", extra={"request_id": request_id,
                                         "session_id": session_id,
                                         "question": body.question[:80]})

    # Pass only the new question + the current HumanMessage.
    # The add_messages reducer in AgentState appends this to the saved history,
    # and MemorySaver loads that history automatically via thread_id.
    initial_state = AgentState(
        question=body.question,
        messages=[HumanMessage(content=body.question)],
        route="",
        answer="",
        sources=[],
        agent_used="",
        guardrail_results={},
        source_document=body.source_document or "",
    )

    try:
        result = await graph.ainvoke(
            initial_state,
            config={
                "configurable": {"thread_id": session_id},  # MemorySaver key
                "callbacks": [cb],
            },
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000
        record_request(request_id, "error", "error", False, total_ms, cb)
        logger.error(
            "request_failed",
            extra={"request_id": request_id, "total_ms": round(total_ms, 1),
                   "error": str(exc)},
        )
        logger.debug("traceback:\n%s", traceback.format_exc())
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        raise HTTPException(status_code=500, detail=detail) from exc

    total_ms  = (time.perf_counter() - t_start) * 1000
    agent_used = result.get("agent_used", "unknown")

    record_request(request_id, agent_used, agent_used, True, total_ms, cb)

    logger.info(
        "request_done",
        extra={
            "request_id": request_id,
            "agent_used": agent_used,
            "total_ms":   round(total_ms, 1),
            "stage_ms":   cb.stage_ms,
            "tokens":     cb.tokens,
        },
    )

    return AgentResponse(
        agent_used=agent_used,
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result.get("sources", [])],
        guardrails=result.get("guardrail_results", {}),
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Phase 4.5 — Document management endpoints
# ---------------------------------------------------------------------------

@app.get("/documents")
def documents_endpoint() -> dict:
    """
    List distinct source_document values stored in ChromaDB.

    Returns {"documents": ["resume.pdf", "report.pdf", ...]} sorted alphabetically.
    Returns an empty list if no PDFs have been ingested yet.
    Uses a raw chromadb client (no embedding model load) so it is fast.
    """
    return {"documents": list_documents()}


# ---------------------------------------------------------------------------
# Phase 4.5 — PDF upload endpoint
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    filename: str
    chunks_stored: int


@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a PDF upload, save it to data/, and ingest it into ChromaDB.

    The file is stored at data/<original_filename> so it can be re-ingested
    or inspected later. Existing files with the same name are overwritten.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest_path = os.path.join("data", file.filename)
    os.makedirs("data", exist_ok=True)

    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}") from exc

    try:
        chunks = pdf_ingest(dest_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    # "filename" is a reserved LogRecord attribute — use non-reserved names.
    logger.info("pdf_uploaded", extra={"pdf_name": file.filename, "chunk_count": chunks})
    return UploadResponse(filename=file.filename, chunks_stored=chunks)


# ---------------------------------------------------------------------------
# Phase 4 — Observability endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics")
def metrics_endpoint() -> dict:
    """
    Return aggregate statistics over the last 1 000 /agent requests.

    Fields:
      total_requests  — count of all recorded requests
      error_rate      — fraction that raised an exception (0.0–1.0)
      total_tokens    — cumulative tokens across all Groq LLM calls
      by_route        — per-agent: count, avg_ms, p95_ms
      recent          — last 10 requests (newest first)
    """
    return aggregate()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """
    Serve the metrics dashboard — a self-contained HTML page.

    The page calls GET /metrics on load and every 30 seconds, then renders:
      - Four summary cards (requests, error rate, avg latency, tokens)
      - Route breakdown table (count / avg ms / p95 ms per agent)
      - Recent requests table with per-stage latency pills
    """
    return HTMLResponse(content=DASHBOARD_HTML)
