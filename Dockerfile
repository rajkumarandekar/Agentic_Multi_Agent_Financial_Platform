# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Agentic AI Platform — Hugging Face Spaces (Docker, port 7860)
# Single container: React frontend built at image build time, served by
# FastAPI/uvicorn at runtime.  Node is kept in the final image because the
# MCP tool agent spawns a Node subprocess at runtime.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# ---- system deps: Node 20 (build + runtime for MCP) + build essentials ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- build the React frontend -------------------------------------------------
# Copy manifests first so npm ci layer is cached independently of source changes.
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm ci --prefix ./frontend

COPY frontend/ ./frontend/
RUN npm run build --prefix ./frontend
# frontend/dist/ is now ready; node_modules/ is left in place for MCP runtime.

# ---- install Python dependencies ----------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- copy the rest of the application ----------------------------------------
COPY agents/        ./agents/
COPY api/           ./api/
COPY config/        ./config/
COPY data/          ./data/
COPY ingestion/     ./ingestion/
COPY models/        ./models/
COPY observability/ ./observability/
COPY orchestration/ ./orchestration/

# data/company.db and data/company_data.xlsx are gitignored (generated, not
# committed) -- so a fresh clone/build has neither. Regenerate them here so
# the SQL/finance/credit/risk/forecast agents have real data at runtime.
# models/*.pkl (churn classifier, sales forecaster) ARE committed, so those
# are not regenerated -- train_models.py takes minutes and isn't rerun on
# every build.
RUN python data/generate_company_data.py && python data/setup_db.py

# ---- runtime -----------------------------------------------------------------
# HF Spaces injects secrets as environment variables; .env is not present.
# LANGCHAIN_TRACING_V2 defaults to false so the app works without a LangSmith key.
ENV LANGCHAIN_TRACING_V2=false \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
