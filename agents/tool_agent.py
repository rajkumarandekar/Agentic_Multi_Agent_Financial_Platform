"""
Native tool agent — Windows-compatible, no MCP subprocess required.

Why not MCP here:
  Windows ProactorEventLoop cannot spawn stdio subprocesses inside a running
  asyncio loop (uvicorn's loop), so the MCP filesystem server raises
  NotImplementedError at runtime. The native tools below replicate the same
  capabilities using pathlib and Python's math module — no child process needed.

To switch to the MCP version (Linux / Docker), change the import in
orchestration/graph.py from:
    from agents.tool_agent import run as tool_run
to:
    from agents.tool_agent_mcp import run as tool_run
"""

import math
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-8b-8192")
DATA_DIR     = Path(os.getenv("MCP_DATA_DIR", "data")).resolve()

# ---------------------------------------------------------------------------
# Tool definitions — decorated functions become LangChain BaseTool objects.
# The docstring IS the tool description the LLM reads when deciding to call it.
# ---------------------------------------------------------------------------

_MATH_NS = {k: v for k, v in vars(math).items() if not k.startswith("_")}


@tool
def list_files(subdirectory: str = "") -> str:
    """
    List the files and folders inside the data/ directory.
    Pass a subdirectory name (relative to data/) to look inside it,
    or leave empty to list the top-level data/ folder.
    Returns names, types (file/dir), and sizes in bytes.
    """
    target = DATA_DIR / subdirectory if subdirectory else DATA_DIR
    if not target.exists():
        return f"Directory not found: {target}"
    if not target.is_dir():
        return f"Not a directory: {target}"

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return "Directory is empty."

    lines = []
    for entry in entries:
        kind = "file" if entry.is_file() else "dir"
        size = f"{entry.stat().st_size} B" if entry.is_file() else ""
        lines.append(f"  [{kind}] {entry.name}  {size}".rstrip())
    return "\n".join(lines)


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports arithmetic operators (+, -, *, /, **, %) and all functions
    from Python's math module (sqrt, log, sin, cos, floor, ceil, etc.).
    Example: 'sqrt(144) + 2 ** 8'
    """
    try:
        # Restricted eval: no builtins, only math-module names in scope.
        result = eval(expression, {"__builtins__": {}}, _MATH_NS)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


# ---------------------------------------------------------------------------
# Public interface — same signature as tool_agent_mcp.run so graph.py is unchanged
# ---------------------------------------------------------------------------

async def run(question: str) -> str:
    """
    Answer a tool-use question with native LangChain tools (no subprocess).

    The ReAct agent decides which tool(s) to call, observes the result, and
    produces a final natural-language answer.

    Args:
        question: e.g. "What files are in the data directory?" or "What is sqrt(256)?"

    Returns:
        The agent's final answer as a plain string.
    """
    llm   = ChatGroq(model=GROQ_MODEL, temperature=0)
    agent = create_react_agent(llm, [list_files, calculate])
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )
    return result["messages"][-1].content
