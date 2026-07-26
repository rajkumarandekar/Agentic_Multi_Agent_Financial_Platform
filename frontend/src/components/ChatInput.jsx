import { useState, useEffect, useRef } from 'react'
import { uploadPdf } from '../api.js'

/**
 * Textarea + Send button + inline PDF upload ("+") button.
 * Enter submits; Shift+Enter inserts a newline.
 * `prefill` prop pre-populates textarea (from prompt card clicks).
 * `onUpload(filename, chunks)` called after a successful PDF upload.
 */
export default function ChatInput({ onSend, disabled, prefill, onPrefillConsumed, onUpload, onUploadStart, onUploadEnd }) {
  const [value, setValue] = useState('')
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef(null)

  useEffect(() => {
    if (prefill) { setValue(prefill); onPrefillConsumed?.() }
  }, [prefill])

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    setValue('')
    onSend(q)
  }

  async function handleFileChange(e) {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''
    setUploading(true)
    onUploadStart?.(file.name)
    try {
      const result = await uploadPdf(file)
      onUpload?.(result.filename, result.chunks_stored)
    } catch (err) {
      alert(`Upload failed: ${err.message}`)
    } finally {
      setUploading(false)
      onUploadEnd?.()
    }
  }

  return (
    <div className="chat-input-row">
      <input
        ref={fileRef}
        type="file"
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      <button
        className="upload-inline-btn"
        title="Upload PDF"
        disabled={disabled || uploading}
        onClick={() => fileRef.current?.click()}
      >
        {uploading ? '…' : '+'}
      </button>
      <textarea
        className="chat-textarea"
        placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={2}
      />
      <button className="send-btn" onClick={submit} disabled={disabled || !value.trim()}>
        Send
      </button>
    </div>
  )
}
