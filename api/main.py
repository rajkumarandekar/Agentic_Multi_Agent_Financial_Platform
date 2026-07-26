"""
FastAPI app — structured logging + LangSmith tracing.

Start the server:
    uvicorn api.main:app --reload

Observability is LangSmith-only: every graph run is traced automatically
(see setup_langsmith(), below) and tagged/named per-request so individual
conversations are easy to find in the LangSmith UI. The custom in-process
metrics dashboard this project used to have (/dashboard, /metrics, /logs,
/system) was removed in favor of this — LangSmith already gives per-node
timing, token counts, and full input/output traces with no extra code to
maintain here.
"""

import json
import logging
import os
import re
import time
import traceback
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

# --- configure JSON logging before anything else emits log lines ---
from observability.logger import configure_logging, new_request_id
configure_logging()

logger = logging.getLogger(__name__)

load_dotenv()

# Validate required key at startup so the error is obvious immediately.
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

# --- LangSmith must be set up BEFORE importing the graph so that LangChain
#     picks up LANGCHAIN_TRACING_V2 at import time. ---
from observability.langsmith_setup import setup_langsmith   # noqa: E402
setup_langsmith()

# Import agents and graph after env is ready.
from agents.rag_agent import ask                            # noqa: E402
from orchestration.graph import (                            # noqa: E402
    graph, has_checkpoint, get_pending_interrupt, get_pending_interrupt_node,
    init_sqlite_checkpointer, close_sqlite_checkpointer,
)
from orchestration.state import AgentState                  # noqa: E402
from ingestion.pdf_ingest import ingest as pdf_ingest, list_documents, reset_collection  # noqa: E402
from data.chat_store import (                                       # noqa: E402
    create_chat           as _cs_create,
    list_chats            as _cs_list,
    get_chat              as _cs_get,
    rename_chat           as _cs_rename,
    delete_chat           as _cs_delete,
    save_message          as _cs_save_msg,
    auto_title            as _cs_auto_title,
    update_source_document as _cs_update_doc,
)

app = FastAPI(
    title="Agentic AI Platform",
    description="Phase 4.5 — Multi-turn memory, React frontend, PDF upload",
    version="0.5.0",
)

# Allow the React dev server (Vite) to call the API. A REGEX, not a fixed
# list -- Vite auto-increments past 5173 to 5174/5175/... whenever a prior
# dev server is still holding the port (routine here, given how many were
# started over the course of this session), so a fixed port number goes
# stale the moment that happens. "localhost" and "127.0.0.1" both covered --
# a browser's Origin header reflects whichever one is literally in the
# address bar, and CORS origin matching is exact-string, not DNS-aware.
# sendMessageStream (api.js) calls the backend's absolute origin directly in
# dev (bypasses the Vite proxy, which was found to buffer SSE responses), so
# this is a REAL cross-origin request needing a real preflight, not one
# Vite's proxy quietly absorbs -- getting this right actually matters now,
# where before every other endpoint went through the same-origin proxy.
# In production this should be tightened to the actual frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    """
    Swap the graph's in-memory MemorySaver for a database-backed AsyncSqliteSaver
    (data/checkpoints.db) so supervisor/conversation state survives a restart.
    Must run inside FastAPI's event loop — aiosqlite's connection thread is
    started on await, which isn't available at plain module-import time.
    """
    await init_sqlite_checkpointer()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await close_sqlite_checkpointer()


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
    agent_override: str | None = Field(None, description="Force agent: rag|sql|tool; omit for auto-routing")


class AgentResponse(BaseModel):
    agent_used: str
    answer: str
    sources: list[SourceItem]
    guardrails: dict
    session_id: str
    agents_used: list = []
    mode: str = "plan"
    # Single, deterministically-chosen follow-up question (no LLM call) --
    # None when there's no genuine on-topic next step for this answer.
    followup: str | None = None


@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(body: AgentRequest) -> AgentResponse:
    """
    Route the question through the LangGraph supervisor to the appropriate agent.

    Flow: input_guard → supervisor → agent(s) → output_guard
    Every run is traced in LangSmith (see setup_langsmith()) — graph_config's
    tags/metadata/run_name below make each request easy to find and filter
    there by session, rather than needing any custom in-process metrics code.

    Guardrails:
      - Input:  injection-pattern regex + LLM safety classifier
      - Output: PII masking + toxicity classifier
    Routing:
      - rag      → ChromaDB retrieval + Groq
      - sql      → SQLite company data query + Groq
      - finance  → deterministic tool dispatch + Groq
      - research → web search + Groq
    """
    # Use provided session_id or generate a new one for this conversation.
    session_id = body.session_id or str(uuid.uuid4())

    request_id = new_request_id()
    t_start = time.perf_counter()

    logger.info("request_start", extra={"request_id": request_id,
                                         "session_id": session_id,
                                         "question": body.question[:80]})

    graph_config = {
        "configurable": {"thread_id": session_id},
        # LangSmith UI: run_name shows up as the trace's title (instead of a
        # generic "LangGraph"); tags/metadata make a session's requests
        # filterable/searchable there instead of needing a custom dashboard.
        "run_name": body.question[:80],
        "tags": [f"session:{session_id}"],
        "metadata": {"session_id": session_id, "request_id": request_id},
    }

    # ── Human-in-the-loop: resume a paused clarify() interrupt ────────────────
    # If the supervisor routed to "clarify" on a PRIOR turn, orchestration/
    # clarify.py's interrupt() paused the graph mid-turn and the last response
    # already surfaced the clarifying question to the user (see the pause
    # check below). This turn's body.question is the user's answer to that
    # question, not a new question -- resume the paused graph with it instead
    # of starting a fresh turn. graph.ainvoke() never raises/returns an
    # "interrupted" marker in this LangGraph version; aget_state() is the only
    # way to detect a pause (see get_pending_interrupt's docstring).
    resuming = await get_pending_interrupt(session_id) is not None
    if resuming:
        # A paused clarify() interrupt already has all the state it needs
        # (the original question, scratchpad, etc.) checkpointed from before
        # the pause -- nothing to rebuild, just supply the resume value.
        graph_input = Command(resume=body.question)
    else:
        # Build initial state.  Fields listed here OVERWRITE the MemorySaver checkpoint.
        # source_document is deliberately omitted when the client sends None so that
        # MemorySaver keeps the previous value — the document stays "selected" across
        # turns without the frontend re-sending it every request.
        #
        # scratchpad and agents_used use the `add` reducer, so setting them to []
        # here resets them for each new request (old values were from the prior turn).
        # Treat missing, empty-string, and "None" all as no document selected.
        # Always write source_document so it overwrites the MemorySaver checkpoint —
        # without this, deselecting a PDF in the UI has no effect on routing.
        _source_doc = body.source_document if (
            body.source_document and str(body.source_document).strip() not in ("", "None")
        ) else None

        # Seed `messages` from the SQLite chat history ONLY when the checkpointer has
        # never seen this thread_id — e.g. an old chat reopened after checkpoints.db
        # was cleared, or a chat created before this session's server process started.
        # A thread with an existing checkpoint already carries its full history via
        # the `add_messages` reducer; re-seeding it every turn would re-append the
        # same messages under new ids indefinitely and blow up the context.
        turn_messages: list = [HumanMessage(content=body.question)]
        if not await has_checkpoint(session_id):
            chat = _cs_get(session_id)
            if chat and chat.get("messages"):
                prior: list = []
                for m in chat["messages"][-6:]:  # matches the n=6 window used everywhere else
                    # Truncate like _build_history/_contextual_question do — a long
                    # answer full of markdown tables shouldn't dominate the prompt.
                    if m["role"] == "user":
                        prior.append(HumanMessage(content=m["text"][:300]))
                    elif m["role"] == "assistant":
                        prior.append(AIMessage(content=m["text"][:300]))
                turn_messages = prior + turn_messages
                logger.info(
                    "history_seeded",
                    extra={"request_id": request_id, "session_id": session_id,
                           "prior_messages": len(prior)},
                )

        graph_input = {
            "question":          body.question,
            "messages":          turn_messages,
            "scratchpad":        [],        # always reset — clears stale agent results
            "route":             "",
            "answer":            "",
            "sources":           [],
            "agent_used":        "",
            "agents_used":       [],
            "iteration_count":   0,         # always reset — prevents supervisor seeing old count
            "source_document":   _source_doc,
            "clarified_source":  None,
            "order_confirmed":   False,     # always reset — only needs to survive within one turn's loop
            "guardrail_results": {},
            "knowledge_result":  {},
            "finance_result":    "",
            "plan":              {},
            # "pending_followup" deliberately NOT set here -- omitting it lets
            # the checkpointed value from the PRIOR turn's response_node
            # survive into this turn's supervisor routing decision (see
            # orchestration/state.py). response_node overwrites it again at
            # the end of this turn regardless, so it always reflects "the
            # follow-up suggested by the most recent answer."
        }

    try:
        result = await graph.ainvoke(graph_input, config=graph_config)
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.error(
            "request_failed",
            extra={"request_id": request_id, "total_ms": round(total_ms, 1),
                   "error": str(exc)},
        )
        logger.debug("traceback:\n%s", traceback.format_exc())
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        raise HTTPException(status_code=500, detail=detail) from exc

    total_ms = (time.perf_counter() - t_start) * 1000

    # ── Human-in-the-loop: did THIS invoke pause on a NEW interrupt? ──────────
    # result["answer"] is stale here if so (still "" from graph_input, or the
    # previous turn's answer when resuming) -- the paused node stopped before
    # ever reaching response_node, so surface its prompt directly instead of
    # falling through to the normal result-shaped response below. Two
    # different nodes can pause here (see orchestration/graph.py's module
    # docstring): "clarify" (which agent should answer this) and
    # "confirm_purchase" (should this invoice actually be placed) --
    # get_pending_interrupt_node() says which, so the response is labeled
    # correctly instead of always saying "clarify".
    pending_prompt = await get_pending_interrupt(session_id)
    if pending_prompt is not None:
        pending_node = await get_pending_interrupt_node(session_id) or "clarify"
        logger.info("request_paused_for_hitl",
                    extra={"request_id": request_id, "session_id": session_id,
                           "pending_node": pending_node})
        return AgentResponse(
            agent_used=pending_node,
            answer=pending_prompt,
            sources=[],
            guardrails={},
            session_id=session_id,
            agents_used=[],
            mode=pending_node,
            followup=None,
        )

    agent_used = result.get("agent_used", "unknown")

    logger.info(
        "request_done",
        extra={
            "request_id": request_id,
            "agent_used": agent_used,
            "total_ms":   round(total_ms, 1),
        },
    )

    return AgentResponse(
        agent_used=agent_used,
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result.get("sources", [])],
        guardrails=result.get("guardrail_results", {}),
        session_id=session_id,
        agents_used=result.get("agents_used", []),
        # chat = greeting/chitchat (empty scratchpad); plan = agent(s) ran
        mode="chat" if not result.get("scratchpad") else "plan",
        # Read from state, not recomputed here — response_node already
        # computed this (see orchestration/graph.py) and wrote it into
        # pending_followup, which is the same value next turn's supervisor
        # routing reads. Recomputing here would risk the two ever drifting.
        followup=result.get("pending_followup"),
    )


# Word-level chunk, keeping trailing whitespace attached so words don't
# visually collide when the frontend appends chunks back-to-back.
_WORD_RE = re.compile(r'\S+\s*')


@app.post("/agent/stream")
async def agent_stream_endpoint(body: AgentRequest) -> StreamingResponse:
    """
    Same pipeline as /agent — same guardrails, HITL, routing, memory — with
    ZERO duplicated logic: this calls agent_endpoint() directly (a plain
    async function call, not a second HTTP round-trip) and streams its
    already-fully-guarded AgentResponse back as Server-Sent Events, split
    into one event per word.

    Critically, output_guard has ALREADY run and approved the complete
    answer before this function ever sees it — streaming happens strictly
    AFTER guarding, never instead of it. If output_guard would have blocked
    or masked something, that already happened inside agent_endpoint();
    this endpoint only ever chunks up text that was already safe to return
    as a single response.

    No artificial per-chunk delay here on purpose: verified live that the
    browser's fetch/ReadableStream reader can deliver an entire multi-KB SSE
    response in a SINGLE `read()` regardless of how granularly the server
    flushes (confirmed directly -- a 130-chunk response server-flushed
    ~12ms apart still arrived as one browser-side read). Network-level
    pacing is therefore not reliably observable client-side at all, so the
    typing-effect pacing is done entirely in the frontend instead (see
    api.js's sendMessageStream -- it queues incoming words and reveals them
    on its own timer, decoupled from how the browser batches the underlying
    reads). This endpoint's only job is to hand over the words in order.

    Event stream shape (three event types, in order):
      event: meta   — everything from AgentResponse except `answer` (badges,
                      sources, followup, session_id, etc.) — sent first so
                      the UI can render the message shell immediately.
      event: chunk  — {"text": "<word> "} — one per word of the answer.
      event: done   — stream complete, no further events follow.
    """
    response: AgentResponse = await agent_endpoint(body)

    async def _event_stream():
        meta = response.model_dump()
        answer_text = meta.pop("answer")
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        for word in _WORD_RE.findall(answer_text):
            yield f"event: chunk\ndata: {json.dumps({'text': word})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Phase 4.5 — Document management endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset_endpoint() -> dict:
    """
    Clear all ingested documents from ChromaDB.
    Called by the frontend on every page load so each browser session
    starts with an empty document list.
    """
    reset_collection()
    return {"status": "ok"}


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
        # Run sync ChromaDB ingestion in a thread so it doesn't block the event loop.
        # Large PDFs can take 30-60 s; 120 s timeout prevents infinite hangs.
        import asyncio as _aio
        loop   = _aio.get_event_loop()
        chunks = await _aio.wait_for(
            loop.run_in_executor(None, pdf_ingest, dest_path),
            timeout=120.0,
        )
    except _aio.TimeoutError:
        raise HTTPException(status_code=504, detail="PDF ingestion timed out — file may be too large.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    # "filename" is a reserved LogRecord attribute — use non-reserved names.
    logger.info("pdf_uploaded", extra={"pdf_name": file.filename, "chunk_count": chunks})
    return UploadResponse(filename=file.filename, chunks_stored=chunks)


# ---------------------------------------------------------------------------
# Chat history endpoints
# ---------------------------------------------------------------------------

class ChatUpdateRequest(BaseModel):
    title: str | None = None          # present → rename the chat
    source_document: str | None = None  # present → update the active document


class SaveMessageRequest(BaseModel):
    role: str
    text: str
    agent_used: str = ""
    agents_used: list = []
    sources: list = []


class AutoTitleRequest(BaseModel):
    question: str


@app.post("/chats")
def create_chat_endpoint() -> dict:
    """Create a new chat session. Returns {id, title, created_at, updated_at}."""
    return _cs_create()


@app.get("/chats")
def list_chats_endpoint() -> dict:
    """List all chats, most recently updated first."""
    return {"chats": _cs_list()}


@app.get("/chats/{chat_id}")
def get_chat_endpoint(chat_id: str) -> dict:
    """Return a chat with its messages and metadata."""
    chat = _cs_get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.patch("/chats/{chat_id}")
def update_chat_endpoint(chat_id: str, body: ChatUpdateRequest) -> dict:
    """Rename a chat and/or update its active source_document."""
    if not _cs_get(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    if body.title is not None:
        _cs_rename(chat_id, body.title)
    if body.source_document is not None:
        _cs_update_doc(chat_id, body.source_document)
    result = _cs_get(chat_id)
    return result or {}


@app.delete("/chats/{chat_id}")
def delete_chat_endpoint(chat_id: str) -> dict:
    """Delete a chat and all its messages."""
    ok = _cs_delete(chat_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "deleted"}


@app.post("/chats/{chat_id}/messages")
def save_message_endpoint(chat_id: str, body: SaveMessageRequest) -> dict:
    """Persist a message (user or assistant) to a chat."""
    return _cs_save_msg(
        chat_id     = chat_id,
        role        = body.role,
        text        = body.text,
        agent_used  = body.agent_used,
        agents_used = body.agents_used,
        sources     = body.sources,
    )


@app.post("/chats/{chat_id}/title")
def auto_title_endpoint(chat_id: str, body: AutoTitleRequest) -> dict:
    """Set chat title from the first user question."""
    title = _cs_auto_title(chat_id, body.question)
    return {"title": title}
