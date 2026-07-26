"""
Console trace helper -- prints which agent/stage is about to make an LLM
call, and the exact prompt it's sending (system prompt, conversation
history if any, and the query/user content). Purely a debugging aid, not
used for any business logic.

Toggle off with DEBUG_TRACE_PROMPTS=false in .env; on by default.

Windows console cp1252 can't encode ₹ and some other characters that show
up in real prompts -- every print here falls back to an ascii-safe encode
rather than crashing the request (this bit us earlier in this project with
a stray debug print; see agents/finance_agent.py history).
"""
import os

_ENABLED = os.getenv("DEBUG_TRACE_PROMPTS", "true").strip().lower() not in ("false", "0", "no")

_BAR = "=" * 78
_SEP = "-" * 78


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode())


def trace_llm_call(
    stage: str,
    *,
    query: str = "",
    history: str = "",
    system: str = "",
    extra: str = "",
    system_preview_chars: int = 600,
) -> None:
    """Print one clearly-delimited block for a single LLM call.

    stage:   which agent/component, e.g. "SUPERVISOR routing",
             "FINANCE planner (candidates: [...])".
    query:   the actual question/content being asked this call.
    history: conversation history included in the prompt, if any -- printed
             explicitly as "(none)" when empty so it's obvious whether this
             call had memory of prior turns or not.
    system:  the system prompt -- previewed (not always full) to keep the
             console readable; full text still visible via system_preview_chars.
    extra:   anything else worth showing (e.g. candidate tool list, feedback
             from a rejected previous attempt).
    """
    if not _ENABLED:
        return

    lines = [_BAR, f"[LLM CALL] {stage}", _BAR]

    lines.append("HISTORY:")
    lines.append(history if history else "(none -- this call has no conversation memory)")
    lines.append(_SEP)

    lines.append("QUERY:")
    lines.append(query if query else "(none)")

    if system:
        lines.append(_SEP)
        lines.append("SYSTEM PROMPT:")
        preview = system.strip()
        if len(preview) > system_preview_chars:
            preview = preview[:system_preview_chars] + f"... [{len(system)} chars total]"
        lines.append(preview)

    if extra:
        lines.append(_SEP)
        lines.append(extra)

    lines.append(_BAR)
    _safe_print("\n".join(lines))


def trace_agent_start(agent: str, question: str) -> None:
    """One-line banner marking which agent node the graph just entered --
    printed from graph.py's node functions so it's obvious which agent is
    processing a given turn before its own internal LLM calls fire."""
    if not _ENABLED:
        return
    _safe_print(f">>> AGENT: {agent}  |  question: {question[:120]!r}")
