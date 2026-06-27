# Project: Enterprise Agentic AI Platform

## Goal
A multi-agent AI platform built to match a specific job description (agentic AI 
engineer role). Showcase project for interviews. Must be explainable line-by-line — 
favor clarity over cleverness.

## Tech stack (do not add beyond this without asking)
- Python 3.14, FastAPI backend
- LangGraph for orchestration, LangChain core
- Groq API (Gemma/Llama) as the LLM
- HuggingFace embeddings
- ChromaDB (vector store), SQLite (SQL agent data)
- Streamlit frontend (simple)
- Docker, GitHub Actions CI
- Observability: LangSmith + OpenTelemetry -> Prometheus -> Grafana

## Scope — build ONLY these, in this order
1. RAG agent: PDF ingest -> chunk -> embed -> ChromaDB -> retrieve -> answer
2. LangGraph supervisor routing to: RAG agent, SQL agent, one MCP tool agent
3. Guardrails (PII + toxicity check) + eval harness (retrieval + answer scoring)
4. Observability: tracing + metrics dashboard
5. Productionize: Docker, GitHub Actions, README + runbook

## Explicitly OUT of scope (do not build)
Auth beyond a basic API key, payment/email/SMTP services, multiple vector DBs, 
backups, disaster recovery, more than one MCP tool, the full middleware stack.

## Working rules
- Build ONE phase at a time. Stop after each phase and wait for me to test.
- After writing code, briefly explain the key decisions so I understand it.
- Keep functions small and readable. Add docstrings.
- Use environment variables for all keys (.env, never hardcoded).