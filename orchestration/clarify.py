"""
Clarify node: Human-in-the-Loop via LangGraph interrupt().

When the supervisor routes to "clarify":
  1. An LLM generates ONE specific question to resolve the ambiguity.
  2. interrupt() pauses the graph and surfaces the question to the caller.
  3. The FastAPI endpoint sends it to the frontend, waits for the user's reply.
  4. The endpoint resumes the graph with Command(resume=<user_answer>).
  5. Execution continues from exactly where it paused; clarified_source is set.
  6. The loop-back edge returns to the supervisor, which now routes correctly.

The MemorySaver checkpointer persists the full graph state across the pause,
so the supervisor still sees all accumulated scratchpad data after resuming.
"""
import os
import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.types import interrupt

from orchestration.state import AgentState

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_CLARIFY_SYSTEM = """\
You are a helpful financial assistant. The user's question is ambiguous:
it could be answered from an uploaded PDF document OR from their transaction
database, but using the wrong source would give an incorrect result.

Generate exactly ONE short, friendly clarifying question (under 20 words)
that resolves which source to use. Be specific about the two options.
"""


def clarify_node(state: AgentState) -> dict:
    """
    Generate a clarifying question then pause the graph until the user answers.

    The return value of interrupt() is the user's answer supplied by the
    FastAPI endpoint via Command(resume=<user_answer>).  Execution resumes
    from the line immediately after interrupt() — the rest of this function
    runs only after the user has replied.
    """
    question = state.get("question", "")
    doc      = state.get("source_document") or "uploaded document"

    # Generate a targeted clarifying question
    try:
        # max_tokens caps the RESERVED completion budget Groq counts toward
        # its TPM rate limit -- see agents/sql_agent.py's identical comment.
        llm = ChatGroq(model=_MODEL, temperature=0.3, max_tokens=256)
        resp = llm.invoke([
            SystemMessage(content=_CLARIFY_SYSTEM),
            HumanMessage(
                content=(
                    f"User question: {question}\n"
                    f"PDF available: {doc}\n"
                    f"DB available: financial transactions (CUST001, CUST002)"
                )
            ),
        ])
        clarifying_question = resp.content.strip()
    except Exception as exc:
        logger.warning("clarify LLM failed: %s", exc)
        clarifying_question = (
            f"Should I answer using the uploaded '{doc}' or your transaction database?"
        )

    print(f"[clarify] pausing — question: {clarifying_question!r}")

    # Pause the graph; resume when Command(resume=answer) is sent to the endpoint
    user_answer: str = interrupt(clarifying_question)

    print(f"[clarify] resumed — user_answer: {user_answer!r}")
    return {"clarified_source": str(user_answer)}
