"""
Shared state that flows through every node in the LangGraph.

LangGraph passes this dict into each node function. Nodes return a partial
dict with only the keys they update; LangGraph merges it back into the state.

The `messages` field uses the `add_messages` reducer so node updates APPEND
new messages rather than replacing the list. Combined with the MemorySaver
checkpointer in graph.py this gives multi-turn memory across requests on
the same thread_id.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Single source of truth for one request through the supervisor graph."""

    question:          str                                     # user's question — set once at graph entry
    messages:          Annotated[list[BaseMessage], add_messages]  # accumulates across turns
    route:             str                                     # set by supervisor: "rag"|"sql"|"tool"|"blocked"
    answer:            str                                     # set by the chosen agent node
    sources:           list[dict]                              # RAG chunk metadata; empty for sql / tool
    agent_used:        str                                     # echoed in the /agent API response
    guardrail_results: dict                                    # {"input": {...}, "output": {...}}
    source_document:   str                                     # optional RAG filter; "" = search all docs
