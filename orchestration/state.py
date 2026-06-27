"""
Shared state that flows through every node in the LangGraph.

LangGraph passes this dict into each node function. Nodes return a partial
dict with only the keys they update; LangGraph merges it back into the state.
"""

from typing import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Single source of truth for one request through the supervisor graph."""

    question:   str               # user's question — set once at graph entry
    messages:   list[BaseMessage] # conversation history (HumanMessage → AIMessage)
    route:      str               # set by supervisor: "rag" | "sql" | "tool"
    answer:     str               # set by the chosen agent node
    sources:    list[dict]        # RAG chunk metadata; empty for sql / tool
    agent_used: str               # echoed in the /agent API response
