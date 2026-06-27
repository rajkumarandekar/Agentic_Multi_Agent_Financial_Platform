"""
Output guardrail: runs after the agent produces an answer, before /agent returns.

Two sequential checks:
  1. PII masking — regex replaces sensitive patterns in-place. Never blocks;
     always masks and reports what was found.
  2. Toxicity classifier — Groq call checks for harmful content in the
     (already-masked) answer. Sets passed=False if toxic, but still returns
     the answer so the caller can decide whether to surface or suppress it.
"""

import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# (pattern, replacement_token) pairs — ordered from most- to least-specific
# so overlapping patterns (e.g. card vs phone) are resolved predictably.
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),                                   "[CARD]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                                          "[SSN]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),          "[EMAIL]"),
    (re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),           "[PHONE]"),
]

_TOXICITY_SYSTEM = """\
You are a content moderation classifier.
Given a text passage, decide if it contains toxic, hateful, harmful, or
inappropriate content that should not be shown to end users.

Reply with exactly one of:
  NO: <one-line reason>
  YES: <one-line reason>
"""


def _check_pii(text: str) -> tuple[str, dict]:
    """
    Scan text for PII and replace each match with a token.

    Returns (masked_text, check_result). Always passes — masking is the action.
    """
    masked = text
    total = 0
    breakdown: list[str] = []
    for pattern, token in _PII_PATTERNS:
        masked, n = pattern.subn(token, masked)
        if n:
            breakdown.append(f"{n}×{token}")
        total += n

    detail = ", ".join(breakdown) if breakdown else "no PII found"
    return masked, {
        "name": "pii",
        "passed": True,
        "detail": detail,
        "masked_count": total,
    }


def _check_toxicity(text: str) -> dict:
    """Ask the LLM whether the answer contains harmful content."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    # Truncate to keep prompt cost low; 1 000 chars covers typical answers.
    response = llm.invoke([
        SystemMessage(content=_TOXICITY_SYSTEM),
        HumanMessage(content=text[:1000]),
    ])
    raw = response.content.strip()
    passed = raw.upper().startswith("NO")
    return {
        "name": "toxicity",
        "passed": passed,
        "detail": raw[:150],
    }


def check_output(answer: str) -> dict:
    """
    Run PII masking then toxicity check on an agent answer.

    Returns a dict with:
      passed  — False if toxicity check failed
      answer  — text with PII replaced by tokens
      checks  — list of per-check result dicts
    """
    masked_answer, pii_check = _check_pii(answer)
    toxicity_check = _check_toxicity(masked_answer)

    return {
        "passed": pii_check["passed"] and toxicity_check["passed"],
        "answer": masked_answer,
        "checks": [pii_check, toxicity_check],
    }
