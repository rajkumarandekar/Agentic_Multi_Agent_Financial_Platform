import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'

// Real examples matching the platform's actual 7 agents (SQL, Finance,
// Credit, Risk, Forecast, RAG, Research) over the real TechMart India
// dataset -- these used to be personal-finance-app examples ("my spending",
// "my resume", "my savings rate") and referenced a "tool" agent that no
// longer exists, left over from before the TechMart multi-agent pivot.
const CARDS = [
  {
    q:       'Show me all customers in Chennai',
    sub:     'Raw data lookups over TechMart\'s database',
    agent:   'sql',
    agentLabel: 'SQL',
    gradClass: 'card--green',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
      </svg>
    ),
  },
  {
    q:       "What's the profit margin on PRD001?",
    sub:     'Pricing, GST, bulk quotes, margins',
    agent:   'finance',
    agentLabel: 'Finance',
    gradClass: 'card--teal',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
      </svg>
    ),
  },
  {
    q:       'Can CUST001 apply for a ₹50,000 loan?',
    sub:     'Credit eligibility, EMI, loan proposals',
    agent:   'credit',
    agentLabel: 'Credit',
    gradClass: 'card--blue',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/>
      </svg>
    ),
  },
  {
    q:       'Is CUST001 at risk of churning?',
    sub:     'Churn prediction, fraud/anomaly checks',
    agent:   'risk',
    agentLabel: 'Risk',
    gradClass: 'card--red',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    ),
  },
  {
    q:       'Forecast revenue for the next 3 months',
    sub:     'Flexible day/week/month/year forecasts',
    agent:   'forecast',
    agentLabel: 'Forecast',
    gradClass: 'card--purple',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
  },
  {
    q:       'Summarise this document',
    sub:     'Upload a PDF, then ask about it',
    agent:   'rag',
    agentLabel: 'Documents',
    gradClass: 'card--orange',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
      </svg>
    ),
  },
]

const BankIcon = () => (
  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
    <polyline points="9 22 9 12 15 12 15 22"/>
  </svg>
)

export default function ChatWindow({ messages, loading, loadingStatus, onCardClick, onFollowUpSelect }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const isEmpty = messages.length === 0 && !loading

  return (
    <div className="chat-window">
      {isEmpty && (
        <div className="chat-hero">
          <div className="hero-icon"><BankIcon /></div>
          <h1 className="hero-title">Financial Intelligence Assistant</h1>
          <p className="hero-sub">Powered by Multi-Agent AI</p>

          <div className="prompt-cards">
            {CARDS.map((card, i) => (
              <button
                key={i}
                className={`prompt-card ${card.gradClass}`}
                onClick={() => onCardClick?.(card.q)}
              >
                <div className="card-top">
                  <span className="card-icon"><card.Icon /></span>
                  <span className={`card-agent-label label--${card.agent}`}>{card.agentLabel}</span>
                </div>
                <p className="card-q">{card.q}</p>
                <p className="card-sub">{card.sub}</p>
              </button>
            ))}
          </div>
        </div>
      )}

      {messages.map((msg, i) => (
        <MessageBubble
          key={i}
          message={msg}
          onFollowUpSelect={i === messages.length - 1 ? onFollowUpSelect : undefined}
        />
      ))}

      {loading && (
        <div className="bubble assistant loading-bubble">
          <span className="dot" /><span className="dot" /><span className="dot" />
          {/* Live "which agent is running" label, driven by /agent/stream's
              "status" SSE events (see api/main.py, App.jsx) -- falls back to
              a generic "Thinking..." for the brief window before the first
              status event arrives. */}
          <span className="loading-status-label">{loadingStatus?.label || 'Thinking…'}</span>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
