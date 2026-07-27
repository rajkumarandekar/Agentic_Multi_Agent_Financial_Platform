"""
Research agent: web search for external benchmarks and industry data.

Used when the question requires data not in the local DB or uploaded PDFs —
e.g. "What is the average savings rate in India?" or "Compare my spending
to industry benchmarks."

Phase 3 rewrite (see project chat history): orchestration moved from
langgraph's create_react_agent to CrewAI (Agent/Task/Crew), matching the
resume's CrewAI claim genuinely rather than leaving it unsupported by the
code. web_search itself is UNCHANGED — still a plain Python function backed
by Tavily (requires TAVILY_API_KEY), with the same graceful "unavailable"
fallback message when the key isn't set.

Why search runs in plain Python instead of as a CrewAI-native tool: CrewAI's
built-in tool-calling loop is implemented against litellm's provider
abstraction, and litellm doesn't have a prebuilt wheel for this project's
Python version — installing it here requires a Rust/Cargo toolchain, which
isn't available in this environment (see requirements.txt comment). Rather
than force that install or half-implement CrewAI's internal native-tool-call
protocol against a hand-rolled LLM class (fragile, effectively re-building
part of CrewAI's own executor), the search step runs first in Python exactly
as before, and its results are handed to the Crew as context for the actual
reasoning/synthesis step — CrewAI genuinely does the agentic
role/goal/backstory-driven synthesis, just not the tool dispatch.

_GroqLLM is a small custom crewai.llms.base_llm.BaseLLM implementation that
calls Groq's OpenAI-compatible endpoint directly via the `openai` SDK
(already a transitive dependency of crewai) — this is the officially
supported way to plug in a non-litellm backend (see BaseLLM's own
docstring: "Users can extend this class to create custom LLM
implementations that don't rely on litellm's authentication mechanism").
"""
import logging
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


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


class _GroqLLM:
    """
    Minimal crewai.llms.base_llm.BaseLLM implementation that talks to Groq's
    OpenAI-compatible endpoint directly, bypassing litellm entirely (see
    module docstring for why). Only implements plain text calls — no
    tools/available_functions handling, since this agent's Task never asks
    the LLM to make native tool calls (search runs beforehand in Python).
    """
    def __new__(cls, model: str = _MODEL, temperature: float = 0.0):
        from crewai.llms.base_llm import BaseLLM

        class _Impl(BaseLLM):
            def __init__(self):
                super().__init__(model=model, temperature=temperature)
                self._client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=os.getenv("GROQ_API_KEY"),
                )

            def call(
                self, messages, tools=None, callbacks=None,
                available_functions=None, from_task=None, from_agent=None,
                response_model=None,
            ) -> str:
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                # Groq's endpoint rejects extra per-message fields CrewAI
                # attaches (e.g. cache_breakpoint) -- keep only role/content.
                clean = [{"role": m["role"], "content": m["content"]} for m in messages]
                resp = self._client.chat.completions.create(
                    model=self.model, messages=clean,
                    temperature=self.temperature or 0.0,
                )
                return resp.choices[0].message.content

        return _Impl()


_RESEARCHER_BACKSTORY = (
    "You are a financial research analyst for TechMart India, an Indian "
    "e-commerce and financial services platform. You specialise in turning "
    "raw web search results into concise, sourced findings for business "
    "stakeholders -- never guessing when the search results don't cover "
    "something."
)


async def run(question: str) -> str:
    """
    Answer a research question: run web_search in Python first, then hand
    the raw results to a CrewAI Agent/Task/Crew for sourced synthesis.

    Falls back to a friendly error message on any exception so the
    supervisor can still proceed to the response node.
    """
    try:
        search_results = web_search.invoke({"query": question})

        from crewai import Agent, Crew, Process, Task

        agent = Agent(
            role="Financial Research Analyst",
            goal="Answer the user's question using ONLY the provided search results, citing sources.",
            backstory=_RESEARCHER_BACKSTORY,
            llm=_GroqLLM(),
            verbose=False,
        )
        task = Task(
            description=(
                f"User question:\n{question}\n\n"
                f"Raw web search results:\n{search_results}\n\n"
                "Write a concise answer (3-5 sentences) using ONLY the "
                "information above. Note the source/date where the search "
                "results provide one. If the results don't actually answer "
                "the question, say so explicitly rather than guessing."
            ),
            expected_output="A concise, sourced answer or an explicit statement that the search results don't cover the question.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = await crew.kickoff_async()
        return str(result)
    except Exception as exc:
        logger.error("research_agent error: %s", exc, exc_info=True)
        return "External research unavailable. Answer based on internal data only."
