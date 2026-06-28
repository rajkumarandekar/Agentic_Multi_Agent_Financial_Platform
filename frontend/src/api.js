/**
 * Thin fetch wrappers for the FastAPI backend.
 * Vite proxy (vite.config.js) forwards these paths to http://localhost:8000,
 * so no hardcoded base URL is needed here.
 */

/**
 * Send a question to the /agent endpoint.
 * @param {string} question
 * @param {string|null} sessionId       - pass null on the very first turn
 * @param {string|null} sourceDocument  - filename to restrict RAG retrieval; null = all docs
 * @returns {Promise<{answer, agent_used, sources, guardrails, session_id}>}
 */
export async function sendMessage(question, sessionId, sourceDocument = null) {
  const body = { question, session_id: sessionId }
  if (sourceDocument) body.source_document = sourceDocument
  const res = await fetch('/agent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

/**
 * Fetch the list of ingested document filenames from GET /documents.
 * @returns {Promise<string[]>}
 */
export async function getDocuments() {
  const res = await fetch('/documents')
  if (!res.ok) return []
  const data = await res.json()
  return data.documents || []
}

/**
 * Upload a PDF file to /upload.
 * @param {File} file
 * @returns {Promise<{filename, chunks_stored}>}
 */
export async function uploadPdf(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/upload', { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Upload failed')
  }
  return res.json()
}
