import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'

const CARDS = [
  {
    q:       'What did I spend on groceries?',
    sub:     'View your grocery transactions',
    agent:   'sql',
    agentLabel: 'Database',
    gradClass: 'card--green',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
      </svg>
    ),
  },
  {
    q:       'Show my top merchants this month',
    sub:     'See where you spend most',
    agent:   'sql',
    agentLabel: 'Database',
    gradClass: 'card--green',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
      </svg>
    ),
  },
  {
    q:       'Summarise this document',
    sub:     'Extract key insights from uploads',
    agent:   'rag',
    agentLabel: 'Documents',
    gradClass: 'card--blue',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
      </svg>
    ),
  },
  {
    q:       'What are the skills in my resume?',
    sub:     'Analyse uploaded documents',
    agent:   'rag',
    agentLabel: 'Documents',
    gradClass: 'card--blue',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
      </svg>
    ),
  },
  {
    q:       'Calculate my monthly savings rate',
    sub:     '₹80,000 income minus expenses',
    agent:   'tool',
    agentLabel: 'Tools',
    gradClass: 'card--orange',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
      </svg>
    ),
  },
  {
    q:       'What is CAGR on ₹50,000 over 3 years?',
    sub:     'Compound growth calculation',
    agent:   'tool',
    agentLabel: 'Tools',
    gradClass: 'card--orange',
    Icon: () => (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
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

export default function ChatWindow({ messages, loading, onCardClick, onFollowUpSelect }) {
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
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
