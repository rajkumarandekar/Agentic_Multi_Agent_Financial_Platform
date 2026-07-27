---
title: Agentic AI Platform
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# TechMart India — Multi-Agent Financial AI Platform

A 7-agent LangGraph system that answers pricing, credit, risk, and forecasting
questions over a synthetic Indian e-commerce/financial dataset — built as a
showcase project, with an explicit focus on every claim in it being
independently verifiable in the code rather than asserted in a README.

## Problem statement

Business staff who aren't SQL-literate need fast answers from operational and
financial data without waiting on an analyst — while anything consequential
(placing an order, approving a loan) stays under explicit human approval,
never an autonomous AI decision. That second half is deliberate: this system
is built around a **maker-checker** (four-eyes) pattern borrowed from banking
governance — an agent computes and proposes (the "maker"), a human explicitly
approves or rejects before it's final (the "checker"). See
[Human-in-the-loop](#human-in-the-loop-maker-checker) below.

**Scope boundary, stated honestly:** this covers revenue-side financial
operations — pricing, credit/loans, churn/fraud risk, forecasting, collections
— not full accounting/P&L, tax filing, or ledger reconciliation. It's an
operational decision-support layer, not a finance department replacement.

## Architecture

A looping LangGraph supervisor, not a linear pipeline: every worker node
loops back to the supervisor, which re-reads the accumulated scratchpad and
decides the next step (or emits `done`) on every iteration. A regex +
embedding fast-path (`orchestration/semantic_router.py`) resolves
unambiguous first-turn questions without an LLM call at all; the supervisor's
own LLM router only runs when that fast path has no confident opinion.

| Agent | Owns | Notable design point |
|---|---|---|
| **SQL** | Raw data lookups — customers, products, transactions | NL→SQL via Groq, deterministic id-normalization pre-flight |
| **RAG** | Uploaded PDF Q&A | ChromaDB + MMR retrieval, all-MiniLM-L6-v2 embeddings |
| **Finance** | Pricing, GST, bulk quotes, loyalty discounts, margins, CLV | CLV includes real NPV discounting, not just a spreadsheet sum |
| **Credit** | Credit eligibility, EMI, loan proposals, collections priority | `request_loan` calls **into the Risk agent** for a churn signal before recommending — a genuine cross-agent dependency, not a parallel tool |
| **Risk** | Churn/retention prediction, fraud/anomaly detection | RandomForest + LeaveOneOut CV, honestly reported (see [Model honesty](#model-honesty)) |
| **Forecast** | Revenue forecasting at any horizon (day/week/month/year) | One ARIMA model, scaled to the requested granularity — see below |
| **Research** | External market/industry benchmarks | CrewAI (Agent/Task/Crew) synthesis over Tavily search results |

Deterministic Python does all arithmetic (GST, EMI, margins, NPV) — the LLM's
job is routing and tool-argument extraction, never the math itself.

### Human-in-the-loop (maker-checker)

Two consequential actions are gated behind an explicit approve/reject, using
LangGraph's `interrupt()`/`Command(resume=...)`:

- **`confirm_purchase`** — Finance computes a full invoice; nothing is
  considered "ordered" until the user replies `approve`.
- **`confirm_loan`** — Credit computes a full loan proposal (EMI, credit
  check, Risk-agent churn signal); nothing is considered "approved" until
  the user replies `approve`. An ambiguous reply is treated as *not*
  approved, never guessed either way.

### Forecast: one model, honest scaling

Flexible day/week/month/year forecasting is **derived from a single trained
monthly ARIMA model** by scaling, not a second model fit at finer
granularity. An earlier attempt at training directly on daily/weekly revenue
rollups produced a 111–263% test MAPE — the synthetic transaction data is
generated per calendar month, so day/week boundaries don't carry its real
signal. Rather than ship a model that bad, day/week/year estimates are scaled
from the monthly model with **wider uncertainty bands at finer granularity**
— a genuine forecasting property, surfaced explicitly rather than hidden.

### Model honesty

The churn classifier's reported LeaveOneOut accuracy is **62%** (3-class:
High/Medium/Low risk), against a ~33–40% random baseline for 3 classes — not
a headline number, but a real one. An earlier version scored 98.8% purely
from label leakage (the label was thresholded on `days_inactive`/
`return_rate`, which were also fed back in as model features); removing that
leakage dropped the honest number to 41% — barely above chance, because the
synthetic data had no real behavioral signal to learn from. Fixing that
required changing how the data itself is generated (customers now have a
persistent engagement weight correlating historical purchase behavior with
future inactivity), not just re-tuning the model. See
`models/train_models.py` and `data/generate_company_data.py` for the full
account.

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq API (Llama 3.1) |
| Orchestration | LangGraph + LangChain |
| Research agent orchestration | CrewAI (custom `BaseLLM` calling Groq's OpenAI-compatible endpoint directly — see `agents/research_agent.py`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (MMR retrieval) |
| SQL store | SQLite |
| ML | scikit-learn (RandomForest, LeaveOneOut CV), pmdarima/statsmodels (ARIMA) |
| Backend | FastAPI + uvicorn |
| Frontend | React + Vite |
| Observability | LangSmith tracing (optional; degrades gracefully without a key) |

## Required secrets

| Secret | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — **required** |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace token for the embedding model — **required** |
| `LANGSMITH_API_KEY` | LangSmith tracing — optional, tracing disabled if absent |
| `TAVILY_API_KEY` | Research agent's web search — optional, degrades to an explicit "unavailable" message if absent |

## Running locally

```bash
# Backend
cp .env.example .env   # fill in GROQ_API_KEY at minimum
pip install -r requirements.txt
python data/setup_db.py         # build data/company.db from data/company_data.xlsx
python models/train_models.py   # train ARIMA + churn + demand models
uvicorn api.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` (Vite dev server with API proxy).

## Testing

```bash
pytest tests/ -q
```

294 tests: deterministic fast-dispatch coverage per agent, HITL
approve/reject/ambiguous-reply flows (real `interrupt()`/`Command(resume=...)`,
not mocked), routing accuracy, and guardrail behavior. No test asserts an
LLM's exact wording — only deterministic tool output and routing decisions,
since Groq's outputs aren't reproducible across calls.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/agent` | Main multi-agent chat endpoint |
| `POST` | `/agent/stream` | SSE-streamed version of the same endpoint |
| `POST` | `/upload` | Upload a PDF for RAG indexing |
| `GET` | `/documents` | List ingested document filenames |
| `POST` | `/chats`, `GET /chats`, `GET /chats/{id}` | Chat session persistence |
| `POST` | `/reset` | Clear a session's conversation state |
| `GET` | `/health` | Liveness check |

## Scaling to production

What's here vs. what a real production deployment would add:

- **Data**: synthetic SQLite dataset (140 products, 250 customers, ~6,000
  transactions) → a real deployment reads from the actual transactional
  database (Postgres/MySQL), with the SQL agent's schema description
  generated from the live schema, not hand-written.
- **HITL approval**: an in-chat approve/reject reply → a real deployment
  routes the pending proposal to a proper approval queue/dashboard with
  role-based access (who is allowed to approve a loan is not "whoever is
  chatting"), and an audit trail of every approve/reject decision.
- **Model retraining**: manual `python models/train_models.py` → a scheduled
  retraining pipeline with drift monitoring, since churn/forecast accuracy
  will genuinely degrade as real customer behavior shifts.
- **Observability**: LangSmith tracing of individual LLM calls → a real
  deployment adds business-level metrics (routing accuracy over time, HITL
  approve/reject rates, model drift) on top of per-call traces.
- **Auth**: none beyond a basic API key → real user authentication +
  per-role permissions (a Credit agent that can approve loans needs
  stricter access control than one that only checks eligibility).
- **Multi-tenancy**: single-tenant demo → the DB layer, checkpointer, and
  vector store would all need tenant-scoping for a real multi-customer
  deployment.

Explicitly out of scope by design (not "not done yet"): payment/email/SMTP
integration, multiple vector DBs, disaster recovery, and full accounting/P&L
— see [Scope boundary](#problem-statement) above.
