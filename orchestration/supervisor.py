"""
Supervisor node: the single LangGraph node that decides which agent handles the question.

It calls the Groq LLM with a tight classification prompt and writes the chosen
route ("rag" | "sql" | "tool") into AgentState. Nothing else — routing only.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from orchestration.state import AgentState

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Tight prompt: three mutually exclusive categories, one-word response.
# Keeping it explicit reduces hallucinated routing labels.
_CLASSIFY_SYSTEM = """\
You are a routing assistant for a multi-agent AI system.
Given a user question, classify it into exactly one of these categories:

  rag   - questions about document content, PDFs, "what does it say",
          "according to", passages, summaries of uploaded files
  sql   - questions about shipment data, counts, totals, averages,
          filtering records, "how many", "list all shipments"
  tool  - questions about the local filesystem (listing files, reading
          a file's contents, "what files are in", "show me the file")
          OR math / calculations ("calculate", "what is X plus Y",
          "square root", "sqrt", arithmetic expressions, unit conversions)

Reply with exactly one lowercase word: rag, sql, or tool. Nothing else.
"""

_VALID_ROUTES = {"rag", "sql", "tool"}


def supervisor_node(state: AgentState) -> dict:
    """
    Classify the user's question and set state["route"].

    For multi-turn sessions, the last few messages from the conversation are
    included as context so follow-up questions like "what about the pending
    ones?" can be routed correctly based on the prior exchange.

    Returns only {"route": <str>} — LangGraph merges this into the full state.
    """
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    # Build a short context block from the previous turn(s).
    # state["messages"] ends with the current HumanMessage; everything before
    # it is prior conversation history. Cap at last 4 messages (2 exchanges).
    history = state.get("messages", [])
    prior = history[:-1][-4:] if len(history) > 1 else []

    if prior:
        lines = []
        for m in prior:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content[:200]}")
        context_block = "\n".join(lines)
        classify_text = (
            f"[Prior conversation]\n{context_block}\n\n"
            f"[New question to classify]\n{state['question']}"
        )
    else:
        classify_text = state["question"]

    response = llm.invoke([
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=classify_text),
    ])
    route = response.content.strip().lower()

    # Guard against unexpected output — fall back to rag (safest default).
    if route not in _VALID_ROUTES:
        route = "rag"

    print(f"[supervisor] '{state['question'][:60]}' → route={route}")
    return {"route": route}
