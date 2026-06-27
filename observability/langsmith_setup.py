"""
Optional LangSmith tracing setup.

Call setup_langsmith() once at application startup, before importing the graph.
LangChain reads LANGCHAIN_TRACING_V2 at import time, so the env var must be
set before any langchain_* module is imported.

If LANGSMITH_API_KEY is absent or setup fails for any reason, the function
logs a warning and returns False. The application continues normally — tracing
is purely additive and never required for the app to run.
"""

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
_DEFAULT_PROJECT  = "agentic-ai-platform"


def setup_langsmith() -> bool:
    """
    Enable LangSmith tracing when LANGSMITH_API_KEY is present in the environment.

    Sets the three env vars LangChain checks automatically:
      LANGCHAIN_TRACING_V2   — activates the tracing callback
      LANGCHAIN_ENDPOINT     — LangSmith API base URL
      LANGCHAIN_PROJECT      — project name shown in the LangSmith UI

    Returns True if tracing was enabled, False otherwise.
    """
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        logger.info("LangSmith tracing disabled (LANGSMITH_API_KEY not set)")
        return False

    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv(
            "LANGCHAIN_ENDPOINT", _DEFAULT_ENDPOINT
        )
        os.environ["LANGCHAIN_PROJECT"]    = os.getenv(
            "LANGCHAIN_PROJECT", _DEFAULT_PROJECT
        )
        logger.info(
            "LangSmith tracing enabled",
            extra={"project": os.environ["LANGCHAIN_PROJECT"]},
        )
        return True
    except Exception as exc:
        logger.warning("LangSmith setup failed — continuing without tracing: %s", exc)
        return False
