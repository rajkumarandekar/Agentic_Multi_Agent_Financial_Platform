# data/

This directory holds PDFs that the RAG agent indexes, plus the SQL agent's database.

## What's included in the repo

- `shipments.db` — SQLite database used by the SQL agent (tracked in git so the deployed app works immediately)
- `.gitkeep` — keeps the directory tracked when no PDFs are present

## What is NOT included (and why)

PDF files (`data/*.pdf`) are excluded from git via `.gitignore` because they may contain personal information. Upload your own PDFs through the UI or the `/upload` endpoint — they are ingested into ChromaDB and stored here temporarily.

## Adding your own PDFs

**Via the web UI:** Use the Upload panel in the app — drag and drop a PDF and it is chunked, embedded, and stored in ChromaDB automatically.

**Via the API:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"
```

**Via the CLI (ingest directly):**
```bash
python -m ingestion.pdf_ingest path/to/your/document.pdf
```
