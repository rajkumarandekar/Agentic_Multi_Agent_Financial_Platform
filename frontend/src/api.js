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
export async function sendMessage(question, sessionId, sourceDocument = null, agentOverride = null) {
  const body = { question, session_id: sessionId }
  if (sourceDocument) body.source_document = sourceDocument
  if (agentOverride && agentOverride !== 'auto') body.agent_override = agentOverride
  let res
  try {
    res = await fetch('/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      // Without this, a stalled request (e.g. Groq rate-limit retries piling
      // up server-side) hangs the fetch indefinitely — handleSend's own
      // try/catch/finally already clears the loading spinner on any thrown
      // error, but only if the fetch promise actually settles.
      signal: AbortSignal.timeout(60000),
    })
  } catch (err) {
    throw new Error(err.name === 'TimeoutError' ? 'Request timed out. Please try again.' : err.message)
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

/**
 * Send a question to /agent/stream and receive the answer as it types in.
 *
 * Same guardrails/routing/memory pipeline as sendMessage() -- this is purely
 * a different transport (Server-Sent Events) for a typing-effect UI.
 * output_guard has ALREADY approved the complete answer before any of it is
 * sent (see api/main.py's agent_stream_endpoint docstring) -- streaming here
 * never bypasses or races the guard, it only paces out delivery of text that
 * was already fully computed and safe to return.
 *
 * @param {string} question
 * @param {string|null} sessionId
 * @param {string|null} sourceDocument
 * @param {string|null} agentOverride
 * @param {(meta: object) => void} onMeta   - fires once, before any chunks (agent_used, sources, followup, session_id, etc. -- everything from AgentResponse except `answer`)
 * @param {(chunkText: string) => void} onChunk - fires once per word, in order
 * @returns {Promise<void>} resolves once the stream completes
 */
export async function sendMessageStream(question, sessionId, sourceDocument = null, agentOverride = null, onMeta, onChunk) {
  const body = { question, session_id: sessionId }
  if (sourceDocument) body.source_document = sourceDocument
  if (agentOverride && agentOverride !== 'auto') body.agent_override = agentOverride

  // Dev only: Vite's dev-server proxy buffers this specific streamed response
  // (verified live -- a long answer's ~130 SSE chunks, each server-flushed
  // ~12ms apart, all arrived within the same ~10ms window through the proxy,
  // while hitting the backend directly delivered them with real spacing
  // intact). Bypass the proxy for just this call in dev. In production this
  // never applies -- frontend and backend are served from the same origin
  // (see CLAUDE.md's single-service deployment), so the relative path is
  // used and there's no separate dev proxy in the way at all.
  const url = import.meta.env.DEV ? 'http://localhost:8000/agent/stream' : '/agent/stream'

  let res
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      // Covers the whole stream, not just the initial response -- generous
      // since chunk delivery is paced in milliseconds, not seconds, even for
      // long answers (see api/main.py's _STREAM_CHUNK_DELAY_S).
      signal: AbortSignal.timeout(60000),
    })
  } catch (err) {
    throw new Error(err.name === 'TimeoutError' ? 'Request timed out. Please try again.' : err.message)
  }
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }

  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE events are separated by a blank line; each event is
    // "event: <name>\ndata: <json>\n\n".
    let sepIndex
    while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sepIndex)
      buffer = buffer.slice(sepIndex + 2)

      const eventMatch = rawEvent.match(/^event:\s*(\w+)/m)
      const dataMatch  = rawEvent.match(/^data:\s*(.*)$/m)
      if (!eventMatch || !dataMatch) continue

      const eventName = eventMatch[1]
      const data = JSON.parse(dataMatch[1])

      if (eventName === 'meta') onMeta?.(data)
      else if (eventName === 'chunk') onChunk?.(data.text)
      // 'done' needs no handling -- reader.read()'s `done` ends the loop
      // naturally right after the server closes the stream.
    }
  }
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
 * Clear all ingested documents from ChromaDB.
 * Called once on page load to start each session fresh.
 * @returns {Promise<void>}
 */
export async function resetDocuments() {
  await fetch('/reset', { method: 'POST' }).catch(() => {})
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

// ── Chat history API ───────────────────────────────────────────────────────

/** Create a new chat session. Returns {id, title, created_at, updated_at}. */
export async function createChat() {
  const res = await fetch('/chats', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
  if (!res.ok) throw new Error('Failed to create chat')
  return res.json()
}

/** List all chats, newest first. Returns [{id, title, updated_at}, ...]. */
export async function listChats() {
  const res = await fetch('/chats')
  if (!res.ok) return []
  const data = await res.json()
  return data.chats || []
}

/** Get a chat with its messages and metadata. Returns the full chat object. */
export async function getChat(chatId) {
  const res = await fetch(`/chats/${chatId}`)
  if (!res.ok) return null
  return res.json()
}

/** Rename a chat. */
export async function renameChat(chatId, title) {
  await fetch(`/chats/${chatId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
}

/**
 * Persist the active source_document for a chat to SQLite.
 * Called when a user uploads a PDF or selects one from the document panel.
 * @param {string} chatId
 * @param {string} sourceDocument  — filename or "" to clear
 */
export async function updateChatDoc(chatId, sourceDocument) {
  await fetch(`/chats/${chatId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_document: sourceDocument }),
  }).catch(() => {})
}

/** Delete a chat. */
export async function deleteChat(chatId) {
  await fetch(`/chats/${chatId}`, { method: 'DELETE' })
}

/**
 * Save a message to a chat.
 * @param {string} chatId
 * @param {{role, text, agent_used?, agents_used?, sources?}} msg
 */
export async function saveMessage(chatId, msg) {
  await fetch(`/chats/${chatId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      role:        msg.role,
      text:        msg.text,
      agent_used:  msg.agentUsed  || msg.agent_used  || '',
      agents_used: msg.agentsUsed || msg.agents_used || [],
      sources:     msg.sources    || [],
    }),
  })
}

/** Auto-title a chat from its first question. */
export async function autoTitleChat(chatId, question) {
  const res = await fetch(`/chats/${chatId}/title`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) return null
  return res.json()
}
