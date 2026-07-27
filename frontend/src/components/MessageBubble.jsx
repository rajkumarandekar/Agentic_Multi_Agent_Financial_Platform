import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import FinancialChart, { parseChartData, stripChartBlock, stripMarkdownTables } from './FinancialChart.jsx'
import AnswerCard from './AnswerCard.jsx'

// ── SQL table rendering ───────────────────────────────────────────────────

function parseSqlTable(text) {
  const lines = text.trim().split('\n').map(l => l.trim()).filter(Boolean)
  if (lines.length < 3) return null
  const tabular = lines.filter(l => l.split(',').length >= 4)
  if (tabular.length < 3) return null
  const headers = lines[0].split(',').map(h => h.trim())
  const rows    = lines.slice(1).map(l => l.split(',').map(c => c.trim()))
  return { headers, rows }
}

function isAmountCol(header) {
  return /amount|balance|total|price|value/i.test(header)
}

function SqlTable({ headers, rows }) {
  return (
    <div className="sql-table-wrap">
      <table className="sql-table">
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i} className={isAmountCol(h) ? 'sql-col-amount' : ''}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className={ri % 2 === 0 ? 'sql-row-even' : 'sql-row-odd'}>
              {row.map((cell, ci) => {
                const amtCol = isAmountCol(headers[ci] || '')
                return (
                  <td key={ci} className={amtCol ? 'sql-col-amount' : ''}>
                    {cell}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const BADGE_CONFIG = {
  rag:       { bg: '#6366f1', label: 'RAG' },
  sql:       { bg: '#10b981', label: 'SQL' },
  tool:      { bg: '#f59e0b', label: 'TOOL' },
  finance:   { bg: '#0d9488', label: 'FINANCE' },
  credit:    { bg: '#2563eb', label: '💳 CREDIT' },
  risk:      { bg: '#dc2626', label: '⚠️ RISK' },
  forecast:  { bg: '#7c3aed', label: '📈 FORECAST' },
  research:  { bg: '#0891b2', label: '🔎 RESEARCH' },
  chat:      { bg: '#6b7280', label: 'CHAT' },
  clarify:   { bg: '#8b5cf6', label: '❓ CLARIFY' },
  confirm_purchase: { bg: '#0891b2', label: '🛒 CONFIRM ORDER' },
  confirm_loan:     { bg: '#2563eb', label: '💳 CONFIRM LOAN' },
  blocked:   { bg: '#ef4444', label: 'BLOCKED' },
  error:     { bg: '#ef4444', label: 'ERROR' },
  synthesis: { bg: '#f59e0b', label: '✦ SYNTHESIS', gold: true },
}

function AgentBadge({ name, synthesis }) {
  const cfg = BADGE_CONFIG[name] || { bg: '#6b7280', label: name.toUpperCase() }
  return (
    <span
      className={`agent-badge${cfg.gold ? ' agent-badge--synthesis' : ''}`}
      style={{ background: cfg.gold ? undefined : cfg.bg }}
    >
      {cfg.label}
    </span>
  )
}

function BadgeRow({ agentsUsed, agentUsed }) {
  // Use agents_used array if available (multi-agent); fall back to single agentUsed
  const agents = agentsUsed && agentsUsed.length > 0 ? agentsUsed : (agentUsed ? [agentUsed] : [])
  if (!agents.length) return null
  return (
    <div className="bubble-badges">
      {agents.map((name, i) => (
        <AgentBadge key={i} name={name} synthesis={name === 'synthesis'} />
      ))}
    </div>
  )
}

function FollowUpPrompt({ question, onSelect }) {
  return (
    <button className="followup-prompt" onClick={() => onSelect(question)}>
      <span className="followup-icon">💡</span>
      <span className="followup-text">{question}</span>
      <span className="followup-cta">Ask this →</span>
    </button>
  )
}

export default function MessageBubble({ message, onFollowUpSelect }) {
  const [showSources, setShowSources] = useState(false)
  const { role, text, agentUsed, agentsUsed, sources, followup } = message

  if (role === 'user') {
    return <div className="bubble user">{text}</div>
  }

  const hasSources = sources && sources.length > 0

  // Detect CHART_DATA block — rendered for tool, finance, credit, risk,
  // forecast, synthesis, and the confirm_purchase/confirm_loan HITL pauses
  // (their prompt is finance's/credit's own already-computed invoice/
  // proposal text, embedded CHART_DATA block and all — see orchestration/
  // confirm.py, confirm_loan.py). Real bug this fixes: without
  // confirm_purchase here, the raw "<CHART_DATA>{...}</CHART_DATA>" JSON
  // showed up as literal text in the pause message instead of rendering as
  // a card, AND the AnswerCard's TL;DR (only shown when there's no chart to
  // defer to) picked up a meaningless fragment from the instructional
  // "Reply **approve**..." line. credit/risk/forecast (Phase 2/3 agents)
  // had the exact same gap until this fix -- their proposal/analysis cards
  // were showing as raw unparsed JSON for the same reason.
  const _CHART_AGENTS = ['tool', 'finance', 'credit', 'risk', 'forecast']
  const isToolMsg  = _CHART_AGENTS.includes(agentUsed) || agentUsed === 'synthesis'
    || agentUsed === 'confirm_purchase' || agentUsed === 'confirm_loan'
    || (agentsUsed || []).some(a => _CHART_AGENTS.includes(a))
  const chartData  = isToolMsg ? parseChartData(text || '') : null
  // The chart card already renders every number in <CHART_DATA> as a table/
  // metrics row — the markdown table that follows it in the same string
  // repeats those exact values, so drop just the table lines and keep any
  // surrounding prose (titles, footnotes) intact.
  const plainText  = chartData
    ? stripMarkdownTables(stripChartBlock(text || ''))
    : (text || '')

  // Detect SQL tabular output
  const isSqlMsg   = agentUsed === 'sql' || (agentsUsed || []).includes('sql')
  const sqlTable   = isSqlMsg ? parseSqlTable(plainText) : null

  // Show the TL;DR answer card only when there's real agent-produced detail
  // to summarise. Skip it for: FinancialChart answers (already have their own
  // summary/metrics card), plain chat/greeting replies (no agent ran -- the
  // whole point of a TL;DR is to shortcut a longer detailed answer, and a
  // 1-2 sentence greeting has nothing to shortcut), and short answers where
  // the card would just repeat the whole message verbatim.
  // agent_used is only ever "chat" when NO real agent (sql/rag/finance/
  // research) ran this turn (see orchestration/graph.py's response_node) --
  // agentsUsed itself isn't a reliable check here since it always includes
  // a trailing "response" entry even for a plain greeting.
  const isChatOnly    = agentUsed === 'chat'
  const showAnswerCard = !chartData && !isChatOnly && plainText && plainText.length > 60

  return (
    <div className="bubble assistant">
      <div className="bubble-header">
        <BadgeRow agentsUsed={agentsUsed} agentUsed={agentUsed} />
      </div>

      {chartData && <FinancialChart data={chartData} />}

      {showAnswerCard && <AnswerCard text={plainText} />}

      {sqlTable ? (
        <SqlTable headers={sqlTable.headers} rows={sqlTable.rows} />
      ) : plainText && (
        <div className="bubble-text markdown-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{plainText}</ReactMarkdown>
        </div>
      )}

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

      {followup && onFollowUpSelect && (
        <FollowUpPrompt question={followup} onSelect={onFollowUpSelect} />
      )}
    </div>
  )
}
