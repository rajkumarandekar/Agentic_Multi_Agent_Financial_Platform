import { useState, useEffect } from 'react'
import ChatWindow from './components/ChatWindow.jsx'
import ChatInput from './components/ChatInput.jsx'
import UploadZone from './components/UploadZone.jsx'
import DocumentList from './components/DocumentList.jsx'
import { sendMessage, getDocuments } from './api.js'

/**
 * Read or create a session ID stored in sessionStorage.
 * sessionStorage persists for the lifetime of the browser tab — cleared when
 * the tab is closed, so each new tab starts a fresh conversation.
 */
function getOrCreateSessionId() {
  let id = sessionStorage.getItem('session_id')
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem('session_id', id)
  }
  return id
}

export default function App() {
  // messages shape: { role: 'user'|'assistant', text, agentUsed?, sources? }
  const [messages, setMessages]   = useState([])
  // Session ID is stable for the tab lifetime — initialised once via function ref.
  const [sessionId]               = useState(getOrCreateSessionId)
  const [loading, setLoading]     = useState(false)
  const [uploadMsg, setUploadMsg] = useState(null)
  const [documents, setDocuments] = useState([])       // filenames from GET /documents
  const [selectedDoc, setSelectedDoc] = useState(null) // null = search all docs

  async function fetchDocuments() {
    const docs = await getDocuments()
    setDocuments(docs)
  }

  // Load document list on first render.
  useEffect(() => { fetchDocuments() }, [])

  async function handleSend(question) {
    setMessages(prev => [...prev, { role: 'user', text: question }])
    setLoading(true)
    try {
      const data = await sendMessage(question, sessionId, selectedDoc)
      setMessages(prev => [...prev, {
        role:      'assistant',
        text:      data.answer,
        agentUsed: data.agent_used,
        sources:   data.sources || [],
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role:      'assistant',
        text:      `Error: ${err.message}`,
        agentUsed: 'error',
        sources:   [],
      }])
    } finally {
      setLoading(false)
    }
  }

  function handleUploadSuccess(filename, chunks) {
    setUploadMsg(`Ingested "${filename}" — ${chunks} chunks added to ChromaDB.`)
    setTimeout(() => setUploadMsg(null), 6000)
    fetchDocuments()  // refresh the document list immediately after upload
  }

  // Count completed user turns (each user message = one turn remembered).
  const turns = messages.filter(m => m.role === 'user').length

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <span className="logo">Agentic AI Platform</span>
          <span className="session-badge" title={`Full session ID: ${sessionId}`}>
            session {sessionId.slice(0, 8)}
          </span>
          {turns > 0 && (
            <span className="memory-badge" title="Turns remembered in this session">
              {turns} turn{turns !== 1 ? 's' : ''} in memory
            </span>
          )}
        </div>
        <a className="dashboard-btn" href="/dashboard" target="_blank" rel="noreferrer">
          Dashboard
        </a>
      </header>

      <main className="main">
        <aside className="sidebar">
          <p className="sidebar-label">Upload PDF</p>
          <UploadZone onSuccess={handleUploadSuccess} />
          {uploadMsg && <p className="upload-msg">{uploadMsg}</p>}

          <p className="sidebar-label" style={{ marginTop: '20px' }}>Documents</p>
          {selectedDoc && (
            <p className="filter-active">
              Filtering: {selectedDoc}
            </p>
          )}
          <DocumentList
            documents={documents}
            selected={selectedDoc}
            onSelect={setSelectedDoc}
          />
        </aside>

        <div className="chat-col">
          <ChatWindow messages={messages} loading={loading} />
          <ChatInput onSend={handleSend} disabled={loading} />
        </div>
      </main>
    </div>
  )
}
