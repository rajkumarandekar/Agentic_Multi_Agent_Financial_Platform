/**
 * AnswerCard — a compact "TL;DR" summary + bullet highlights shown above the
 * full detailed answer (table/chart/markdown), which still renders below it
 * unchanged.
 *
 * Deliberately derived purely from the markdown the backend already returns
 * (no new API fields): the SQL/finance/combine formatting prompts in
 * agents/response_agent.py already produce a bold summary line and, for
 * multi-item answers, bullet points -- this just lifts those two pieces to
 * the top so the user gets the gist before reading the full detail. Some
 * answers only have a summary (short chat replies, single-fact lookups) --
 * the card renders bullets only when there are enough of them to be useful.
 */

const MIN_BULLETS_TO_SHOW = 2
const MAX_BULLETS_TO_SHOW = 6

// Matches "- text", "* text", or "1. text" list lines.
const BULLET_LINE_RE = /^\s*(?:[-*]|\d+\.)\s+(.+)$/

function stripInlineMarkdown(s) {
  return s.replace(/\*\*(.+?)\*\*/g, '$1').replace(/`(.+?)`/g, '$1').trim()
}

// A line that is ENTIRELY one bold span, start to end -- e.g.
// "**2 customers are from Chennai.**" -- not just a line that happens to
// contain "**" somewhere in it. Real bug this guards against: a confirm-
// purchase prompt's instructional line "Reply **approve** to place the
// order, or **reject** to cancel it." contains bold spans but ISN'T itself
// a summary -- a looser "does this line contain **...**" test picked out
// the fragment "approve" as the entire TL;DR, which is meaningless.
const FULL_BOLD_LINE_RE = /^\*\*(.+)\*\*$/

// A bare section label ("Summary:", "Details:") introduces content that
// follows (usually a bullet list) rather than being a summary itself --
// must not be picked even though it's a genuine full-bold line.
const BARE_LABEL_RE = /^[\w\s]{1,20}:$/

/**
 * Pull a one-line summary out of already-generated markdown:
 * prefer the last FULLY-bold ("**...**") line (the formatting prompts are
 * instructed to add exactly one of these), else fall back to the first
 * non-empty, non-list, non-heading sentence.
 */
export function extractSummary(text) {
  if (!text) return null
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean)

  // Prefer the LAST fully-bold line, not the first: response_agent.py's
  // formatting prompts add a bold *title* at the top (e.g.
  // "**Customers in Chennai:**") but the actual one-line takeaway
  // ("**2 customers are from Chennai.**") is the bold line placed after
  // the table/detail, per its own formatting rule ("STEP 4: after the
  // table, add ONE bold summary line").
  const boldLines = lines.filter(l => FULL_BOLD_LINE_RE.test(l) && !BULLET_LINE_RE.test(l))
  const boldLine  = [...boldLines].reverse().find(l => !BARE_LABEL_RE.test(l.replace(/\*\*/g, '')))
  if (boldLine) {
    const m = boldLine.match(FULL_BOLD_LINE_RE)
    return stripInlineMarkdown(m ? m[1] : boldLine)
  }

  const proseLine = lines.find(l => !BULLET_LINE_RE.test(l) && !/^#{1,6}\s/.test(l) && !/^\|/.test(l))
  if (!proseLine) return null
  const sentence = proseLine.split(/(?<=[.!?])\s/)[0]
  const clipped  = sentence.length > 180 ? sentence.slice(0, 177) + '…' : sentence
  return stripInlineMarkdown(clipped)
}

/** Pull markdown list items ("- x", "* x", "1. x") out of the answer text. */
export function extractBullets(text) {
  if (!text) return []
  return text
    .split('\n')
    .map(l => l.match(BULLET_LINE_RE))
    .filter(Boolean)
    .map(m => stripInlineMarkdown(m[1]))
    .filter(Boolean)
    .slice(0, MAX_BULLETS_TO_SHOW)
}

export default function AnswerCard({ text }) {
  const summary = extractSummary(text)
  const bullets = extractBullets(text)
  const showBullets = bullets.length >= MIN_BULLETS_TO_SHOW

  if (!summary) return null

  return (
    <div className="answer-card">
      <div className="answer-card-summary">
        <span className="answer-card-badge">TL;DR</span>
        {summary}
      </div>
      {showBullets && (
        <ul className="answer-card-bullets">
          {bullets.map((b, i) => <li key={i}>{b}</li>)}
        </ul>
      )}
    </div>
  )
}
