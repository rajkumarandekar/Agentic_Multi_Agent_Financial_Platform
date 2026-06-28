import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// One colour per agent so the badge is instantly readable.
const AGENT_COLORS = {
  rag:     '#6366f1',
  sql:     '#10b981',
  tool:    '#f59e0b',
  blocked: '#ef4444',
  error:   '#ef4444',
}

/**
 * Renders a single conversation turn.
 *
 * User messages appear right-aligned; assistant messages left-aligned with an
 * agent badge. RAG answers include a collapsible sources panel.
 */
export default function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false)
  const { role, text, agentUsed, sources } = message

  if (role === 'user') {
    return <div className="bubble user">{text}</div>
  }

  const color = AGENT_COLORS[agentUsed] || '#6b7280'
  const hasSources = sources && sources.length > 0

  return (
    <div className="bubble assistant">
      <div className="bubble-header">
        <span className="agent-badge" style={{ background: color }}>
          {agentUsed || 'assistant'}
        </span>
      </div>

      <div className="bubble-text markdown-body">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </div>

      {hasSources && (
        <div className="sources">
          <button
            className="sources-toggle"
            onClick={() => setShowSources(s => !s)}
          >
            {showSources ? 'Hide' : 'Show'} {sources.length} source{sources.length !== 1 ? 's' : ''}
          </button>

          {showSources && (
            <ul className="sources-list">
              {sources.map((s, i) => (
                <li key={i} className="source-item">
                  <span className="source-page">p. {s.page}</span>
                  <span className="source-snippet">{s.snippet}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
