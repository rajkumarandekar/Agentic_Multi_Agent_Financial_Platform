"""
Research agent: web search for external benchmarks and industry data.

Used when the question requires data not in the local DB or uploaded PDFs —
e.g. "What is the average savings rate in India?" or "Compare my spending
to industry benchmarks."

Web search backend (in priority order):
  1. Tavily Search — high-quality, structured results (requires TAVILY_API_KEY)
  2. Graceful fallback message — tells the user to set the API key

The agent is built with create_react_agent (ReAct loop), same pattern as the
SQL and finance agents.  Temperature=0 for factual retrieval.
"""
import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_SYSTEM = """\
You are a financial research assistant. Use web_search to find current
industry benchmarks, market rates, regulatory figures, and external
financial data that is NOT available in the local database or uploaded PDFs.

Guidelines:
- Search for specific, quantitative data when possible.
- Always note the source/date of the data you found.
- Be concise — 3–5 sentences max per finding.
- If search returns no useful result, say so explicitly rather than guessing.
"""


@tool
def web_search(query: str) -> str:
    """Search the web for current financial benchmarks and industry data."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return (
            "Web search is unavailable: TAVILY_API_KEY is not set. "
            "Add it to .env to enable research queries. "
            f"(Query was: {query})"
        )
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        searcher = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)
        results  = searcher.invoke(query)
        if isinstance(results, list) and results:
            return "\n\n".join(
                f"[{r.get('url', 'source')}]\n{r.get('content', '')}"
                for r in results[:3]
            )
        return "No results found for that query."
    except Exception as exc:
        logger.warning("Tavily search error: %s", exc)
        return f"Search failed: {exc}. Try rephrasing the query."


async def run(question: str) -> str:
    """
    Run the research ReAct agent and return a concise summary.

    Falls back to a friendly error message on any exception so the supervisor
    can still proceed to the response node.
    """
    try:
        llm   = ChatGroq(model=_MODEL, temperature=0)
        agent = create_react_agent(
            llm,
            [web_search],
            state_modifier=SystemMessage(content=_SYSTEM),
            checkpointer=None,
        )
        result   = await agent.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 10},
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else "No research results found."
    except Exception as exc:
        logger.error("research_agent error: %s", exc, exc_info=True)
        return "External research unavailable. Answer based on internal data only."
