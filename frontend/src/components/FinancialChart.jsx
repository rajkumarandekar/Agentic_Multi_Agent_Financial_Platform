/**
 * Renders structured financial data from tool agent responses.
 * Pure CSS + SVG — no chart library required.
 */

// ── Utilities ─────────────────────────────────────────────────────────────

const TREND_COLOR = { rising: '#f97316', falling: '#3b82f6', stable: '#22c55e' }
const TREND_ICON  = { rising: '↑', falling: '↓', stable: '→' }

const STATUS_COLOR = {
  good: '#16a34a',
  ok:   '#ca8a04',
  low:  '#dc2626',
  high: '#ea580c',
}
const STATUS_ICON = { good: '✅', ok: '🟡', low: '⬇️', high: '⬆️' }

function scoreColor(score) {
  return score >= 80 ? '#16a34a' : score >= 65 ? '#ca8a04' : score >= 50 ? '#ea580c' : '#dc2626'
}

function inrFmt(n) {
  if (typeof n !== 'number') return n
  return '₹' + n.toLocaleString('en-IN')
}

// ── Sub-components ────────────────────────────────────────────────────────

function HBar({ pct, color }) {
  return (
    <div style={{ flex: 1, height: 8, background: '#e5e7eb', borderRadius: 4, overflow: 'hidden' }}>
      <div style={{ width: `${Math.min(pct, 100)}%`, height: '100%', background: color, borderRadius: 4, transition: 'width .4s' }} />
    </div>
  )
}

function ScoreSVG({ score }) {
  const color = scoreColor(score)
  const r = 44, circ = 2 * Math.PI * r
  return (
    <svg width="110" height="110" viewBox="0 0 110 110" style={{ flexShrink: 0 }}>
      <circle cx="55" cy="55" r={r} fill="none" stroke="#e5e7eb" strokeWidth="12" />
      <circle cx="55" cy="55" r={r} fill="none" stroke={color} strokeWidth="12"
        strokeDasharray={`${circ * score / 100} ${circ}`}
        strokeLinecap="round" transform="rotate(-90 55 55)" />
      <text x="55" y="51" textAnchor="middle" fontSize="22" fontWeight="700" fill={color}>{score}</text>
      <text x="55" y="67" textAnchor="middle" fontSize="11" fill="#9ca3af">/100</text>
    </svg>
  )
}

// metrics can be object {label: value} or array [{label, value}]
function MetricsRow({ metrics }) {
  if (!metrics) return null
  const entries = Array.isArray(metrics)
    ? metrics.map(m => [m.label, m.value])
    : Object.entries(metrics).map(([k, v]) => [k, typeof v === 'number' ? inrFmt(v) : v])
  if (!entries.length) return null
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', margin: '12px 0' }}>
      {entries.map(([label, val], i) => (
        <div key={i} style={{
          flex: '1 1 100px', minWidth: 90, background: '#f8fafc',
          border: '1px solid #e2e8f0', borderRadius: 10, padding: '10px 12px',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
        }}>
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#1e293b', textAlign: 'center' }}>{val}</span>
          <span style={{ fontSize: '0.65rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '.05em', textAlign: 'center' }}>{label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Health Score ──────────────────────────────────────────────────────────

function HealthScore({ data }) {
  const color = scoreColor(data.score)
  // metrics is an object with numeric values
  const metricsDisplay = data.metrics ? [
    { label: 'Total Spend',   value: inrFmt(data.metrics.total_spend) },
    { label: 'Savings Rate',  value: data.metrics.savings_rate },
    { label: 'Balance',       value: inrFmt(data.metrics.balance) },
  ] : []

  return (
    <div className="fc-card">
      <div className="fc-header">💰 Financial Health Report</div>

      {/* Score + Grade */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 20, marginBottom: 14 }}>
        <ScoreSVG score={data.score} />
        <div>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, color }}>Grade {data.grade}</div>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#374151' }}>{data.customer}</div>
          <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>Last 60 days</div>
        </div>
      </div>

      <MetricsRow metrics={metricsDisplay} />

      {/* Score breakdown bars */}
      {data.breakdown?.length > 0 && (
        <div className="fc-section">
          <div className="fc-section-title">📊 Score Breakdown</div>
          <table className="fc-table">
            <thead><tr><th>Metric</th><th>Value</th><th>Score</th><th></th></tr></thead>
            <tbody>
              {data.breakdown.map((row, i) => {
                const sc = STATUS_COLOR[row.status] || '#6b7280'
                return (
                  <tr key={i}>
                    <td>{row.label}</td>
                    <td>{row.value}</td>
                    <td style={{ minWidth: 120 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <HBar pct={(row.points / row.max) * 100} color={sc} />
                        <span style={{ fontSize: '0.72rem', color: '#6b7280', whiteSpace: 'nowrap' }}>
                          {row.points}/{row.max}
                        </span>
                      </div>
                    </td>
                    <td style={{ color: sc }}>{STATUS_ICON[row.status] || ''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {data.recommendations?.length > 0 && (
        <div className="fc-section">
          <div className="fc-section-title">💡 Recommendations</div>
          {data.recommendations.map((r, i) => (
            <div key={i} className="fc-rec-item">
              <span className="fc-rec-num">{i + 1}</span>
              <span>{r}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Spend Forecast ────────────────────────────────────────────────────────

function Forecast({ data }) {
  const rows = data.chart_data || []
  const max  = Math.max(...rows.map(r => r.forecast || 0), 1)

  return (
    <div className="fc-card">
      <div className="fc-header">📅 Spend Forecast — Next 30 Days</div>

      <MetricsRow metrics={[
        { label: 'Total Forecast',    value: inrFmt(data.total_forecast) },
        { label: 'Projected Savings', value: inrFmt(data.projected_savings) },
        { label: 'Income',            value: inrFmt(data.income) },
      ]} />

      {rows.length > 0 && (
        <div className="fc-section">
          <div className="fc-section-title">Forecast by Category</div>
          {rows.map((row, i) => {
            const color = TREND_COLOR[row.trend] || '#6b7280'
            const chg   = row.change
            return (
              <div key={i} className="fc-hbar-row">
                <span className="fc-hbar-label">{row.name}</span>
                <HBar pct={(row.forecast / max) * 100} color={color} />
                <span className="fc-hbar-value">{inrFmt(row.forecast)}</span>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color, whiteSpace: 'nowrap' }}>
                  {TREND_ICON[row.trend]} {row.trend}
                  {chg != null && chg !== 0 && ` (${chg > 0 ? '+' : ''}${chg}%)`}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Investment Comparison ─────────────────────────────────────────────────

const CMP_COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6']

function Comparison({ data }) {
  const rows = data.chart_data || []
  const max  = Math.max(...rows.map(r => r.value || 0), 1)

  return (
    <div className="fc-card">
      <div className="fc-header">📊 Investment Comparison</div>

      <MetricsRow metrics={[
        { label: 'Principal',   value: data.metrics?.Principal },
        { label: 'Period',      value: data.metrics?.Period },
        { label: 'Best Return', value: data.metrics?.['Best Return'] },
      ]} />

      {rows.length > 0 && (
        <div className="fc-section">
          <div className="fc-section-title">Maturity Value</div>
          {rows.map((row, i) => {
            const color    = CMP_COLORS[i % CMP_COLORS.length]
            const isWinner = row.name === data.winner
            return (
              <div key={i} className="fc-hbar-row" style={isWinner ? { fontWeight: 700 } : {}}>
                <span className="fc-hbar-label">{row.name}{isWinner ? ' 🏆' : ''}</span>
                <HBar pct={(row.value / max) * 100} color={color} />
                <span className="fc-hbar-value">{inrFmt(row.value)}</span>
                <span style={{ fontSize: '0.72rem', color: '#6b7280', whiteSpace: 'nowrap' }}>
                  {row.rate}% | real {row.real_return > 0 ? '+' : ''}{row.real_return}%
                </span>
              </div>
            )
          })}
        </div>
      )}
      {data.winner && (
        <div className="fc-winner-badge">🏆 Best: {data.winner}</div>
      )}
    </div>
  )
}

// ── Goal Planner ──────────────────────────────────────────────────────────

function GoalPlanner({ data }) {
  const rows    = data.breakdown || []
  const surplus = data.surplus || 0
  const needed  = rows[0]?.monthly || surplus
  const pct     = surplus > 0 ? Math.min((surplus / needed) * 100, 100) : 0

  return (
    <div className="fc-card">
      <div className="fc-header">🎯 {data.title}</div>

      <MetricsRow metrics={data.metrics
        ? Object.entries(data.metrics).map(([k, v]) => ({ label: k, value: v }))
        : []} />

      {surplus > 0 && rows.length > 0 && (
        <div className="fc-section">
          <div className="fc-section-title">Surplus vs Required Monthly</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: '0.82rem' }}>
            <span style={{ width: 90, fontWeight: 600 }}>Your surplus</span>
            <HBar pct={pct} color="#22c55e" />
            <span style={{ fontWeight: 700, color: '#16a34a', whiteSpace: 'nowrap' }}>
              {inrFmt(surplus)}/mo
            </span>
          </div>
        </div>
      )}

      {rows.length > 0 && (
        <div className="fc-section">
          <div className="fc-section-title">Options</div>
          <table className="fc-table">
            <thead><tr><th>Option</th><th>Monthly</th><th>Feasible</th></tr></thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} style={row.label === data.recommendation ? { fontWeight: 700 } : {}}>
                  <td>{row.label}{row.label === data.recommendation ? ' ⭐' : ''}</td>
                  <td>{inrFmt(row.monthly)}</td>
                  <td style={{ color: row.feasible ? '#16a34a' : '#dc2626' }}>
                    {row.feasible ? '✅' : '❌'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data.months_to_goal != null && (
        <div className="fc-winner-badge">At current rate: goal in {data.months_to_goal} months</div>
      )}
    </div>
  )
}

// ── Anomaly Detector ──────────────────────────────────────────────────────

const ANOMALY_COLORS = {
  'Category Spike':    '#f97316',
  'Large Transaction': '#ef4444',
  'High-Spend Day':    '#8b5cf6',
  'New Merchant':      '#3b82f6',
}

function AnomalyReport({ data }) {
  const items       = data.anomalies || []
  const hasAnomalies = items.length > 0

  return (
    <div className="fc-card">
      <div className="fc-header">{hasAnomalies ? '🚨' : '✅'} Spending Anomaly Report</div>

      <MetricsRow metrics={data.metrics
        ? Object.entries(data.metrics).map(([k, v]) => ({ label: k, value: v }))
        : []} />

      {!hasAnomalies && (
        <div className="fc-clean-bill">✅ No unusual spending — all patterns normal.</div>
      )}

      {hasAnomalies && (
        <div className="fc-section">
          <div className="fc-section-title">{items.length} anomalies found</div>
          <table className="fc-table">
            <thead><tr><th>Type</th><th>Amount</th><th>Detail</th></tr></thead>
            <tbody>
              {items.map((a, i) => (
                <tr key={i}>
                  <td>
                    <span className="fc-anomaly-badge"
                      style={{ background: ANOMALY_COLORS[a.type] || '#6b7280' }}>
                      {a.type}
                    </span>
                  </td>
                  <td>{inrFmt(a.amount)}</td>
                  <td className="fc-anomaly-detail">{a.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ── Calculation ───────────────────────────────────────────────────────────

function Calculation({ data }) {
  return (
    <div className="fc-card" style={{ maxWidth: 500 }}>
      <div className="fc-header">🔧 {data.title}</div>

      {data.result_label && (
        <div style={{ textAlign: 'center', padding: '16px 0 8px' }}>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: '#6366f1' }}>
            {data.result_value}
          </div>
          <div style={{ fontSize: '0.78rem', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '.06em', marginTop: 4 }}>
            {data.result_label}
          </div>
        </div>
      )}

      {data.metrics?.length > 0 && (
        <table className="fc-table" style={{ marginTop: 8 }}>
          <tbody>
            {data.metrics.map((m, i) => (
              <tr key={i}>
                <td style={{ color: '#6b7280', fontWeight: 500 }}>{m.label}</td>
                <td style={{ fontWeight: 700, textAlign: 'right' }}>{m.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

// ── Public API ────────────────────────────────────────────────────────────

export function parseChartData(text) {
  const match = text.match(/<CHART_DATA>\s*([\s\S]*?)\s*<\/CHART_DATA>/)
  if (!match) return null
  try { return JSON.parse(match[1]) } catch { return null }
}

export function stripChartBlock(text) {
  return text.replace(/<CHART_DATA>[\s\S]*?<\/CHART_DATA>\n*/g, '').trim()
}

/**
 * Remove markdown table rows from text — used after a <CHART_DATA> block is
 * parsed and rendered as a FinancialChart card. Finance tool responses embed
 * BOTH a chart/metrics card AND a plain-markdown table describing the exact
 * same numbers (so the answer is still readable if the frontend can't parse
 * the chart block). Rendering both left every finance answer showing the
 * same data twice — once as a card, once as a markdown table right below it.
 * Any non-table prose (titles, footnotes) is kept.
 */
export function stripMarkdownTables(text) {
  return text
    .split('\n')
    .filter(line => !/^\s*\|/.test(line))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export default function FinancialChart({ data }) {
  if (!data) return null
  switch (data.type) {
    case 'health_score': return <HealthScore   data={data} />
    case 'forecast':     return <Forecast      data={data} />
    case 'comparison':   return <Comparison    data={data} />
    case 'goal':         return <GoalPlanner   data={data} />
    case 'anomaly':      return <AnomalyReport data={data} />
    case 'calculation':  return <Calculation   data={data} />
    default:             return null
  }
}
