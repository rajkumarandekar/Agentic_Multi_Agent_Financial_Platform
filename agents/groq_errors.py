"""
Shared helper so a Groq rate limit reads as "try again shortly", not as a
misleading "no data found"/"unable to complete" message.

Every agent below wraps its create_react_agent().invoke() call in a broad
except Exception -- necessary since a malformed LLM response, a connection
drop, etc. all need SOME friendly fallback. But without distinguishing a
genuine RateLimitError from those, a rate-limited request looked exactly
like "this customer/product doesn't exist", which is actively misleading
(confirmed live -- see project chat history).
"""

from groq import RateLimitError as GroqRateLimitError

RATE_LIMIT_MESSAGE = (
    "Groq's request limit has been reached for now. Please try again in a minute."
)


def is_rate_limited(exc: Exception) -> bool:
    return isinstance(exc, GroqRateLimitError)
