"""
MCP tool agent — Linux / Docker path.

Uses the @modelcontextprotocol/server-filesystem Node.js server over stdio.
This works correctly on Linux (uvicorn uses SelectorEventLoop) but raises
NotImplementedError on Windows because ProactorEventLoop cannot create pipes
for subprocesses inside a running asyncio loop.

For local Windows development, use tool_agent.py (native LangChain tools).
Swap the import in orchestration/graph.py to this file when deploying to Docker.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-8b-8192")
MCP_DATA_DIR = os.getenv("MCP_DATA_DIR", "data")


def _npx_command() -> str:
    """
    Return the npx executable name that works on this OS.

    On Windows, npm scripts are .cmd wrappers — 'npx' alone is not found
    by CreateProcess without shell=True, but 'npx.cmd' is.
    shutil.which checks both and returns the first match on PATH.
    """
    candidates = ["npx.cmd", "npx"] if sys.platform == "win32" else ["npx"]
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return candidates[0]  # last-resort fallback; will fail with a clear OS error


async def run(question: str) -> str:
    """
    Answer a filesystem question using the MCP filesystem server.

    Flow:
      1. Start @modelcontextprotocol/server-filesystem as a subprocess.
      2. Open a ClientSession over its stdio pipes.
      3. Load the server's exposed tools (list_directory, read_file, …).
      4. Wrap them as LangChain tools and hand to a ReAct agent.
      5. Return the agent's final message content.

    Args:
        question: A question about local files, e.g. "What files are in data/?"

    Returns:
        A natural-language answer from the agent.
    """
    data_dir = str(Path(MCP_DATA_DIR).resolve())

    server_params = StdioServerParameters(
        command=_npx_command(),
        args=["-y", "@modelcontextprotocol/server-filesystem", data_dir],
    )

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, tools)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]}
            )

    return result["messages"][-1].content
