import { useState, useEffect, useRef } from 'react'
import ChatWindow from './components/ChatWindow.jsx'
import ChatInput from './components/ChatInput.jsx'
import ChatSidebar from './components/ChatSidebar.jsx'
import DocumentList from './components/DocumentList.jsx'
import {
  sendMessageStream, getDocuments, resetDocuments,
  createChat, listChats, getChat,
  renameChat, deleteChat, saveMessage, autoTitleChat,
  updateChatDoc,
} from './api.js'

function UploadOverlay({ filename }) {
  return (
    <div style={{
      position: 'fixed', top: 0, left: 0,
      width: '100vw', height: '100vh',
      backgroundColor: 'rgba(0,0,0,0.75)',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      zIndex: 9999, color: '#fff',
    }}>
      <div style={{
        width: '60px', height: '60px',
        border: '6px solid rgba(255,255,255,0.25)',
        borderTop: '6px solid #fff',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite',
        marginBottom: '24px',
      }} />
      <h2 style={{ fontSize: '22px', fontWeight: 600, margin: '0 0 10px' }}>
        Uploading PDF…
      </h2>
      <p style={{ fontSize: '14px', color: 'rgba(255,255,255,0.65)', margin: '0 0 14px' }}>
        Please wait while we process your document
      </p>
      {filename && (
        <p style={{
          fontSize: '13px', color: 'rgba(255,255,255,0.45)',
          background: 'rgba(255,255,255,0.1)',
          padding: '6px 18px', borderRadius: '20px',
          maxWidth: '320px', overflow: 'hidden',
          textOverflow: 'ellipsis', whiteSpace: 'nowrap', margin: 0,
        }}>
          📄 {filename}
        </p>
      )}
      <style>{`@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}`}</style>
    </div>
  )
}

export default function App() {
  const [messages, setMessages]           = useState([])
  const [sessionId, setSessionId]         = useState(() => crypto.randomUUID())
  const [currentChatId, setCurrentChatId] = useState(null)
  const [chats, setChats]                 = useState([])
  const [loading, setLoading]             = useState(false)
  // Live "which agent is running right now" label, driven by the
  // /agent/stream endpoint's "status" SSE events (see api/main.py's
  // agent_stream_endpoint) -- shown in ChatWindow's loading bubble instead
  // of a blind spinner. Reset to null whenever loading starts/ends so a
  // stale label from the previous turn never lingers into the next one.
  const [loadingStatus, setLoadingStatus] = useState(null)
  const [uploadMsg, setUploadMsg]         = useState(null)
  const [documents, setDocuments]         = useState([])
  const [selectedDoc, setSelectedDoc]     = useState(null)
  const [prefill, setPrefill]             = useState(null)
  const [uploadingFile, setUploadingFile] = useState(null)   // null = hidden

  // true = next user message is the first in this chat (triggers auto-title)
  const firstMessageRef = useRef(true)

  // On page load: wipe ChromaDB for a clean session, then load chat sidebar
  useEffect(() => {
    resetDocuments()
    loadChats()
  }, [])

  async function loadChats() {
    const list = await listChats()
    setChats(list)
  }

  async function fetchDocuments() {
    const docs = await getDocuments()
    setDocuments(docs)
  }

  // ── Chat management ──────────────────────────────────────────────────────

  async function handleNewChat() {
    const chat = await createChat()
    setCurrentChatId(chat.id)
    setSessionId(chat.id)
    setMessages([])
    setSelectedDoc(null)
    setDocuments([])
    firstMessageRef.current = true
    setChats(prev => [chat, ...prev.filter(c => c.id !== chat.id)])
  }

  async function handleSwitchChat(chatId) {
    if (chatId === currentChatId) return
    const chat = await getChat(chatId)
    if (!chat) return

    // Restore conversation identity
    setCurrentChatId(chat.id)
    setSessionId(chat.id)           // MemorySaver thread_id = chat.id

    // Restore all messages exactly as stored
    const restored = (chat.messages || []).map(m => ({
      role:       m.role,
      text:       m.text,
      agentUsed:  m.agent_used,
      agentsUsed: m.agents_used || [],
      sources:    m.sources || [],
    }))
    setMessages(restored)

    // Restore document filter — show the document name in the panel even if
    // ChromaDB was wiped (page refresh); the user knows which doc was active.
    const doc = chat.metadata?.source_document || ''
    setSelectedDoc(doc || null)
    setDocuments(doc ? [doc] : [])   // populate panel with the restored name

    firstMessageRef.current = false  // existing chat — no auto-title needed
  }

  async function handleRenameChat(chatId, title) {
    await renameChat(chatId, title)
    setChats(prev => prev.map(c => c.id === chatId ? { ...c, title } : c))
  }

  async function handleDeleteChat(chatId) {
    await deleteChat(chatId)
    setChats(prev => prev.filter(c => c.id !== chatId))
    if (currentChatId === chatId) {
      setCurrentChatId(null)
      setSessionId(crypto.randomUUID())
      setMessages([])
      setSelectedDoc(null)
      setDocuments([])
      firstMessageRef.current = true
    }
  }

  // ── Document selection — also persists to SQLite ─────────────────────────

  function handleSelectDoc(doc) {
    setSelectedDoc(doc)
    if (currentChatId) {
      updateChatDoc(currentChatId, doc || '')
    }
  }

  // ── Upload (from "+" in ChatInput) ───────────────────────────────────────

  function handleUploadSuccess(filename, chunks) {
    setUploadMsg(`"${filename}" ingested — ${chunks} chunks.`)
    setTimeout(() => setUploadMsg(null), 6000)
    fetchDocuments()        // refresh from ChromaDB (adds to existing list)
    setSelectedDoc(filename)
    // Persist selected document to this chat's SQLite metadata
    if (currentChatId) {
      updateChatDoc(currentChatId, filename)
    }
  }

  // ── Sending messages ─────────────────────────────────────────────────────

  async function handleSend(question) {
    // Show user message immediately — never freeze the UI
    const userMsg = { role: 'user', text: question }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    setLoadingStatus(null)

    // Ensure we have an active chat.
    // chatIsPersisted tracks whether this chatId has a real SQLite row.
    let chatId        = currentChatId
    let chatIsPersisted = !!chatId   // already had a chat before this call

    if (!chatId) {
      try {
        const chat = await createChat()
        chatId           = chat.id
        chatIsPersisted  = true
        setCurrentChatId(chat.id)
        setSessionId(chat.id)
        setChats(prev => [chat, ...prev])
        firstMessageRef.current = true
      } catch {
        // Backend unreachable — continue with ephemeral local session
        chatId          = sessionId
        chatIsPersisted = false
      }
    }

    // BUG FIX: always persist when we have a real DB-backed chat.
    // The old `chatId !== sessionId` check was wrong — after the first message
    // setSessionId(chatId) fires, making them equal and skipping all saves.
    if (chatIsPersisted) {
      saveMessage(chatId, userMsg)
    }

    // Auto-title from the first question in this chat
    if (firstMessageRef.current && chatIsPersisted) {
      firstMessageRef.current = false
      autoTitleChat(chatId, question).then(res => {
        if (res?.title) {
          setChats(prev => prev.map(c => c.id === chatId ? { ...c, title: res.title } : c))
        }
      })
    }

    // If a document is selected and we have a chat, persist the doc association
    // (handles the case where doc was selected before sending the first message)
    if (chatIsPersisted && selectedDoc) {
      updateChatDoc(chatId, selectedDoc)
    }

    // Filled in once the 'meta' SSE event arrives (agent_used, sources,
    // followup, etc.) -- the message bubble is appended then, with empty
    // text, then filled in progressively.
    let assistantMeta = null
    let fullText = ''

    // Typing-effect pacing happens ENTIRELY here, not over the network:
    // verified live that the browser's fetch/ReadableStream reader can
    // deliver an entire multi-KB SSE response in a single `read()`
    // regardless of how granularly the server flushes (a 130-word answer
    // server-flushed ~12ms apart still arrived as one browser-side read) --
    // so network timing can't be relied on to pace anything visually.
    // Instead, incoming words are queued as they arrive and revealed one at
    // a time on our own timer, decoupled from however the browser batches
    // the underlying reads.
    const queue = []
    let networkDone = false
    let revealResolve
    const revealDone = new Promise(resolve => { revealResolve = resolve })
    const REVEAL_INTERVAL_MS = 16

    function startRevealer() {
      const timer = setInterval(() => {
        if (queue.length > 0) {
          fullText += queue.shift()
          setMessages(prev => {
            const next = [...prev]
            next[next.length - 1] = { ...next[next.length - 1], text: fullText }
            return next
          })
        } else if (networkDone) {
          clearInterval(timer)
          revealResolve()
        }
      }, REVEAL_INTERVAL_MS)
    }

    try {
      await sendMessageStream(
        question, chatId, selectedDoc, null,
        (meta) => {
          assistantMeta = meta
          setMessages(prev => [...prev, {
            role:       'assistant',
            text:       '',
            agentUsed:  meta.agent_used,
            agentsUsed: meta.agents_used || [],
            sources:    meta.sources || [],
            followup:   meta.followup || null,
          }])
          // The backend already fully computed + guarded the answer by the
          // time 'meta' arrives -- swap the "..." loading bubble for the
          // real one now, which then fills in via the reveal queue.
          setLoading(false)
          setLoadingStatus(null)
          startRevealer()
        },
        (chunkText) => { queue.push(chunkText) },
        (status) => { setLoadingStatus(status) },
      )
      networkDone = true
      await revealDone   // let the queue finish revealing before persisting

      if (chatIsPersisted && assistantMeta) {
        saveMessage(chatId, {
          role: 'assistant', text: fullText,
          agentUsed: assistantMeta.agent_used,
          agentsUsed: assistantMeta.agents_used || [],
          sources: assistantMeta.sources || [],
        })
        loadChats()   // refresh sidebar order (updated_at changed)
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant', text: `Error: ${err.message}`,
        agentUsed: 'error', sources: [],
      }])
    } finally {
      setLoading(false)
      setLoadingStatus(null)
    }
  }

  function handleCardClick(question) {
    setPrefill(question)
  }

  // Clicking a suggested follow-up IS the "yes, I want that" confirmation —
  // sends immediately rather than just prefilling the input.
  function handleFollowUpSelect(question) {
    handleSend(question)
  }

  const turns = messages.filter(m => m.role === 'user').length

  return (
    <div className="app">
      {uploadingFile !== null && <UploadOverlay filename={uploadingFile} />}
      <header className="header">
        <div className="header-left">
          <span className="logo">Agentic AI Platform</span>
          {turns > 0 && (
            <span className="memory-badge">{turns} turn{turns !== 1 ? 's' : ''}</span>
          )}
        </div>
      </header>

      <main className="main">
        {/* Left: chat history sidebar */}
        <ChatSidebar
          chats={chats}
          currentChatId={currentChatId}
          onNew={handleNewChat}
          onSwitch={handleSwitchChat}
          onRename={handleRenameChat}
          onDelete={handleDeleteChat}
        />

        {/* Middle: document filter panel */}
        <aside className="sidebar">
          <p className="sidebar-label">Documents</p>
          {uploadMsg && <p className="upload-msg">{uploadMsg}</p>}
          {selectedDoc && (
            <p className="filter-active" title={selectedDoc}>Filtering: {selectedDoc}</p>
          )}
          <DocumentList
            documents={documents}
            selected={selectedDoc}
            onSelect={handleSelectDoc}
          />
          {documents.length === 0 && (
            <p className="doc-list-empty">Use the + button in the chat bar to upload a PDF.</p>
          )}
        </aside>

        {/* Right: chat area */}
        <div className="chat-col">
          <ChatWindow
            messages={messages}
            loading={loading}
            loadingStatus={loadingStatus}
            onCardClick={handleCardClick}
            onFollowUpSelect={handleFollowUpSelect}
          />
          <ChatInput
            onSend={handleSend}
            disabled={loading}
            prefill={prefill}
            onPrefillConsumed={() => setPrefill(null)}
            onUpload={handleUploadSuccess}
            onUploadStart={name => setUploadingFile(name)}
            onUploadEnd={() => setUploadingFile(null)}
          />
        </div>
      </main>
    </div>
  )
}
