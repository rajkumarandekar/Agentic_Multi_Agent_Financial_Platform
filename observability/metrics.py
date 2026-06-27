"""
In-process metrics store and LangChain callback for per-stage timing.

Design decisions:
- collections.deque(maxlen=1000) is thread-safe for single appends/reads,
  which is all we do. No lock needed for single-process uvicorn.
- ObservabilityCallback hooks into LangGraph's chain events to get per-node
  wall-clock time without modifying graph.py. LangGraph fires on_chain_start
  / on_chain_end for each node using the node's registered name as serialized["name"].
- Token counts come from on_llm_end → LLMResult.llm_output["token_usage"],
  which Groq (OpenAI-compatible) populates automatically.
"""

import logging
import time
from collections import deque

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_STORE: deque = deque(maxlen=1000)

# Maps kwargs["metadata"]["langgraph_node"] → stage bucket name.
# LangGraph sets this key reliably on every real node callback.
# Sub-chain events (__start__, _write, ChatPromptTemplate, …) either have
# no metadata key or a value not in this map, so they are silently ignored.
_NODE_TO_STAGE: dict[str, str] = {
    "input_guard":  "input_guard",
    "supervisor":   "supervisor",
    "rag":          "agent",
    "sql":          "agent",
    "tool":         "agent",
    "output_guard": "output_guard",
}


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class ObservabilityCallback(BaseCallbackHandler):
    """
    Captures per-node wall-clock timings and cumulative token usage for one request.

    Pass one instance per request to graph.ainvoke(config={"callbacks": [cb]}).
    After the call, read cb.stage_ms and cb.tokens for the collected data.
    """

    def __init__(self, request_id: str) -> None:
        super().__init__()
        self.request_id = request_id
        # run_id (str) → (stage_name, perf_counter start)
        self._starts: dict[str, tuple[str, float]] = {}
        # Stage name → accumulated milliseconds
        self.stage_ms: dict[str, float] = {}
        self.tokens: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

    def on_chain_start(
        self, serialized: dict | None, inputs: dict, *, run_id, **kwargs
    ) -> None:
        """
        Record start time when a real LangGraph node begins.

        Uses kwargs["metadata"]["langgraph_node"] — the only field LangGraph
        sets reliably for every node. serialized is None on most callbacks
        (LangGraph does not populate it for internal chain events), so we never
        call .get() on it without a guard.
        """
        node = kwargs.get("metadata", {}).get("langgraph_node")
        stage = _NODE_TO_STAGE.get(node or "")
        if stage:
            self._starts[str(run_id)] = (stage, time.perf_counter())

    def on_chain_end(self, outputs: dict, *, run_id, **kwargs) -> None:
        """Compute elapsed time for the node and accumulate into stage_ms."""
        entry = self._starts.pop(str(run_id), None)
        if entry:
            stage, start = entry
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Accumulate: the "agent" stage may include sub-chains (SQL ReAct loops).
            self.stage_ms[stage] = self.stage_ms.get(stage, 0.0) + elapsed_ms

    def on_chain_error(self, error: Exception, *, run_id, **kwargs) -> None:
        """Clean up on error so we don't leak start-time entries."""
        self._starts.pop(str(run_id), None)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Accumulate token counts from the Groq (OpenAI-compatible) usage object."""
        usage: dict = {}
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
        self.tokens["prompt"]     += usage.get("prompt_tokens", 0)
        self.tokens["completion"] += usage.get("completion_tokens", 0)
        self.tokens["total"]      += usage.get("total_tokens", 0)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def record_request(
    request_id: str,
    route: str,
    agent_used: str,
    success: bool,
    total_ms: float,
    cb: ObservabilityCallback,
) -> None:
    """Append one completed request record to the in-memory store."""
    _STORE.append({
        "request_id": request_id,
        "timestamp":  time.time(),
        "route":      route,
        "agent_used": agent_used,
        "success":    success,
        "total_ms":   round(total_ms, 1),
        "stage_ms":   {k: round(v, 1) for k, v in cb.stage_ms.items()},
        "tokens":     dict(cb.tokens),
    })


# ---------------------------------------------------------------------------
# Read / aggregate
# ---------------------------------------------------------------------------

def _p95(values: list[float]) -> float:
    """Return the 95th-percentile value from a list (no external library)."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx]


def aggregate() -> dict:
    """
    Compute aggregate statistics over all stored request records.

    Returns a dict suitable for JSON serialisation as the /metrics response.
    """
    records = list(_STORE)
    if not records:
        return {
            "total_requests": 0,
            "error_rate":     0.0,
            "total_tokens":   0,
            "by_route":       {},
            "recent":         [],
        }

    total    = len(records)
    failures = sum(1 for r in records if not r["success"])
    total_tokens = sum(r["tokens"].get("total", 0) for r in records)

    # Group latencies by the agent that actually handled the request.
    buckets: dict[str, list[float]] = {}
    for r in records:
        key = r.get("agent_used") or r.get("route") or "unknown"
        buckets.setdefault(key, []).append(r["total_ms"])

    by_route: dict[str, dict] = {}
    for route, lats in buckets.items():
        by_route[route] = {
            "count":  len(lats),
            "avg_ms": round(sum(lats) / len(lats), 1),
            "p95_ms": round(_p95(lats), 1),
        }

    # Most-recent 10 records, newest first.
    recent = [
        {
            "request_id": r["request_id"],
            "route":      r.get("agent_used") or r.get("route") or "?",
            "total_ms":   r["total_ms"],
            "success":    r["success"],
            "tokens":     r["tokens"].get("total", 0),
            "stage_ms":   r["stage_ms"],
        }
        for r in reversed(list(records)[-10:])
    ]

    return {
        "total_requests": total,
        "error_rate":     round(failures / total, 4),
        "total_tokens":   total_tokens,
        "by_route":       by_route,
        "recent":         recent,
    }
