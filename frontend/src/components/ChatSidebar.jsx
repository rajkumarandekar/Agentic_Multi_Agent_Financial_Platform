import { useState, useRef, useEffect } from 'react'

/**
 * ChatGPT-style sidebar listing saved conversations.
 * Props:
 *   chats         — [{id, title, updated_at}, ...]
 *   currentChatId — active chat id or null
 *   onNew         — () => void
 *   onSwitch      — (chatId) => void
 *   onRename      — (chatId, newTitle) => void
 *   onDelete      — (chatId) => void
 */
export default function ChatSidebar({ chats, currentChatId, onNew, onSwitch, onRename, onDelete }) {
  const [search, setSearch]       = useState('')
  const [renamingId, setRenamingId] = useState(null)
  const [renameVal, setRenameVal] = useState('')
  const [hoverId, setHoverId]     = useState(null)
  const [deleteConfirm, setDeleteConfirm] = useState(null)
  const renameRef = useRef(null)

  useEffect(() => {
    if (renamingId && renameRef.current) renameRef.current.focus()
  }, [renamingId])

  function startRename(chat, e) {
    e.stopPropagation()
    setRenamingId(chat.id)
    setRenameVal(chat.title)
    setHoverId(null)
  }

  function commitRename(chatId) {
    if (renameVal.trim()) onRename(chatId, renameVal.trim())
    setRenamingId(null)
  }

  function handleRenameKey(e, chatId) {
    if (e.key === 'Enter')  { e.preventDefault(); commitRename(chatId) }
    if (e.key === 'Escape') setRenamingId(null)
  }

  function handleDeleteClick(chatId, e) {
    e.stopPropagation()
    if (deleteConfirm === chatId) {
      onDelete(chatId)
      setDeleteConfirm(null)
    } else {
      setDeleteConfirm(chatId)
      // Auto-cancel confirm after 3 s
      setTimeout(() => setDeleteConfirm(id => id === chatId ? null : id), 3000)
    }
  }

  function fmtDate(ts) {
    const d    = new Date(ts * 1000)
    const now  = new Date()
    const diff = Math.floor((now - d) / 86400000)
    if (diff === 0) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    if (diff === 1) return 'Yesterday'
    if (diff < 7)  return d.toLocaleDateString([], { weekday: 'short' })
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
  }

  // Group chats into Today / Yesterday / Earlier
  function groupChats(list) {
    const now = Date.now()
    const groups = { Today: [], Yesterday: [], Earlier: [] }
    for (const c of list) {
      const diff = Math.floor((now - c.updated_at * 1000) / 86400000)
      if (diff === 0) groups.Today.push(c)
      else if (diff === 1) groups.Yesterday.push(c)
      else groups.Earlier.push(c)
    }
    return groups
  }

  const filtered = chats.filter(c =>
    c.title.toLowerCase().includes(search.toLowerCase())
  )
  const groups = search ? null : groupChats(filtered)

  function renderItem(chat) {
    const isActive  = chat.id === currentChatId
    const isHovered = hoverId === chat.id
    const isRename  = renamingId === chat.id
    const isDel     = deleteConfirm === chat.id

    return (
      <div
        key={chat.id}
        className={`cs-item${isActive ? ' cs-item--active' : ''}`}
        onClick={() => !isRename && onSwitch(chat.id)}
        onMouseEnter={() => setHoverId(chat.id)}
        onMouseLeave={() => { setHoverId(null); if (deleteConfirm === chat.id) setDeleteConfirm(null) }}
      >
        {isRename ? (
          <input
            ref={renameRef}
            className="cs-rename-input"
            value={renameVal}
            onChange={e => setRenameVal(e.target.value)}
            onBlur={() => commitRename(chat.id)}
            onKeyDown={e => handleRenameKey(e, chat.id)}
            onClick={e => e.stopPropagation()}
          />
        ) : (
          <>
            {/* Chat icon */}
            <span className="cs-chat-icon">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </span>

            <div className="cs-item-body">
              <span className="cs-item-title">{chat.title}</span>
            </div>

            {(isHovered || isActive) && (
              <div className="cs-item-actions" onClick={e => e.stopPropagation()}>
                <button
                  className="cs-action-btn"
                  title="Rename"
                  onClick={e => startRename(chat, e)}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                  </svg>
                </button>
                <button
                  className={`cs-action-btn${isDel ? ' cs-action-btn--del-confirm' : ' cs-action-btn--del'}`}
                  title={isDel ? 'Click again to confirm' : 'Delete'}
                  onClick={e => handleDeleteClick(chat.id, e)}
                >
                  {isDel ? '?' : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                    </svg>
                  )}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    )
  }

  return (
    <div className="chat-sidebar">
      {/* Header */}
      <div className="cs-header">
        <div className="cs-logo">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <span>History</span>
        </div>
        <button className="cs-new-btn" onClick={onNew} title="New chat">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
        </button>
      </div>

      {/* Search */}
      <div className="cs-search-wrap">
        <div className="cs-search-inner">
          <svg className="cs-search-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <input
            className="cs-search"
            placeholder="Search chats…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          {search && (
            <button className="cs-search-clear" onClick={() => setSearch('')}>✕</button>
          )}
        </div>
      </div>

      {/* Chat list */}
      <div className="cs-list">
        {chats.length === 0 && (
          <div className="cs-empty">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#d1d5db" strokeWidth="1.5" style={{margin:'0 auto 8px',display:'block'}}>
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <p>No chats yet.</p>
            <p>Click <strong>+</strong> or start typing.</p>
          </div>
        )}

        {search ? (
          // Search results — flat list
          filtered.length === 0 ? (
            <p className="cs-no-results">No matches for "{search}"</p>
          ) : (
            filtered.map(renderItem)
          )
        ) : (
          // Grouped by date
          Object.entries(groups).map(([label, items]) =>
            items.length > 0 && (
              <div key={label} className="cs-group">
                <p className="cs-group-label">{label}</p>
                {items.map(renderItem)}
              </div>
            )
          )
        )}
      </div>
    </div>
  )
}
