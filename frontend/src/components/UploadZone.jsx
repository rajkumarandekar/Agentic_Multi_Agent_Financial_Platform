import { useState, useRef } from 'react'
import { uploadPdf } from '../api.js'

/**
 * Drag-and-drop PDF uploader.
 *
 * Dragging a file over the box highlights the border. Clicking opens a file
 * picker. Multiple PDFs can be dropped at once; they are uploaded sequentially.
 */
export default function UploadZone({ onSuccess }) {
  const [dragging,  setDragging]  = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error,     setError]     = useState(null)
  const inputRef = useRef(null)

  async function handleFiles(files) {
    const pdfs = [...files].filter(f => f.name.toLowerCase().endsWith('.pdf'))
    if (!pdfs.length) {
      setError('Only PDF files are accepted.')
      return
    }
    setError(null)
    setUploading(true)
    try {
      for (const file of pdfs) {
        const data = await uploadPdf(file)
        onSuccess(data.filename, data.chunks_stored)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  const zoneClass = [
    'upload-zone',
    dragging  ? 'drag-over'  : '',
    uploading ? 'uploading' : '',
  ].filter(Boolean).join(' ')

  return (
    <div
      className={zoneClass}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !uploading && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
      aria-label="Upload PDF"
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        multiple
        style={{ display: 'none' }}
        onChange={e => handleFiles(e.target.files)}
      />

      {uploading
        ? <p>Uploading…</p>
        : (
          <p>
            Drop PDF here
            <br />
            <span className="upload-sub">or click to browse</span>
          </p>
        )
      }

      {error && <p className="upload-error">{error}</p>}
    </div>
  )
}
