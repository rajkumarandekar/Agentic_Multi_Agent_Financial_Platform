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
COPY ingestion/     ./ingestion/
COPY observability/ ./observability/
COPY orchestration/ ./orchestration/
COPY data/          ./data/

# Ensure the data directory exists so /upload can write PDFs at runtime.
# (The COPY above already creates it, but this guards against an empty data/.)
RUN mkdir -p /app/data

# ---- runtime -----------------------------------------------------------------
# HF Spaces injects secrets as environment variables; .env is not present.
# LANGCHAIN_TRACING_V2 defaults to false so the app works without a LangSmith key.
ENV LANGCHAIN_TRACING_V2=false \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
