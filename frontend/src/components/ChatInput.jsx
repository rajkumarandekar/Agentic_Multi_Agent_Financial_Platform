import { useState } from 'react'

/**
 * Controlled textarea + Send button.
 * Enter submits; Shift+Enter inserts a newline.
 */
export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    setValue('')
    onSend(q)
  }

  return (
    <div className="chat-input-row">
      <textarea
        className="chat-textarea"
        placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={2}
      />
      <button
        className="send-btn"
        onClick={submit}
        disabled={disabled || !value.trim()}
      >
        Send
      </button>
    </div>
  )
}
