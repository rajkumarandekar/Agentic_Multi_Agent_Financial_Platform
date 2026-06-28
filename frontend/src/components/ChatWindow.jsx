import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'

/**
 * Scrollable message list. Auto-scrolls to the bottom whenever the messages
 * array or loading state changes.
 */
export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <div className="chat-window">
      {messages.length === 0 && !loading && (
        <p className="empty-hint">
          Ask a question about your documents, shipment data, or local files.
        </p>
      )}

      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}

      {loading && (
        <div className="bubble assistant loading-bubble">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
      )}

      {/* Invisible anchor that we scroll into view */}
      <div ref={bottomRef} />
    </div>
  )
}
