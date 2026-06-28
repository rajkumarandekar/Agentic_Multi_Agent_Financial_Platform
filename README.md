---
title: Agentic AI Platform
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Agentic AI Platform

A production-style multi-agent AI platform built with LangGraph, FastAPI, and React.

## Features

- **RAG agent** — upload PDFs, chunk & embed them into ChromaDB, ask questions with source citations
- **SQL agent** — natural-language queries over a SQLite shipments database
- **Tool agent** — native file-listing and calculator tools via MCP
- **LangGraph supervisor** — routes each question to the right agent automatically
- **Guardrails** — input injection detection + LLM safety classifier; output PII masking + toxicity check
- **Observability** — per-request structured logging, token counts, per-stage latency, live metrics dashboard at `/dashboard`
- **Multi-turn memory** — conversation history persisted per session via LangGraph's MemorySaver

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq API (Llama / Gemma) |
| Orchestration | LangGraph + LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| SQL store | SQLite |
| Backend | FastAPI + uvicorn |
| Frontend | React + Vite |
| Observability | Structured JSON logging + in-process metrics |

## Required secrets (HF Spaces → Settings → Secrets)

| Secret | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — **required** |
| `LANGCHAIN_API_KEY` | LangSmith tracing key — optional (tracing disabled if absent) |
| `LANGCHAIN_PROJECT` | LangSmith project name — optional |

## Running locally

```bash
# Backend
cp .env.example .env   # fill in GROQ_API_KEY
pip install -r requirements.txt
uvicorn api.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` (Vite dev server with API proxy).

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/agent` | Main multi-agent chat endpoint |
| `POST` | `/upload` | Upload a PDF for RAG indexing |
| `GET` | `/documents` | List ingested document filenames |
| `GET` | `/metrics` | Aggregate request metrics (JSON) |
| `GET` | `/dashboard` | Live metrics dashboard (HTML) |
| `GET` | `/health` | Liveness check |
