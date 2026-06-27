"""
Structured logging: replaces plain logging.basicConfig with a JSON formatter
so every log line is machine-readable and carries a consistent set of fields.

Usage:
    from observability.logger import configure_logging, new_request_id
    configure_logging()              # call once at app startup
    rid = new_request_id()           # 8-char hex per request
    logger.info("msg", extra={"request_id": rid, "stage": "agent_done", "ms": 412})
"""

import json
import logging
import secrets


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    # Fields that belong to LogRecord internals — excluded from the JSON output.
    _SKIP = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        doc: dict = {
            "ts":     self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        # Attach any extra fields the caller passed via extra={...}
        for key, val in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                doc[key] = val
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, default=str)


def configure_logging(level: str = "INFO") -> None:
    """
    Install the JSON formatter on the root logger.

    Call this once at application startup before any other logging calls.
    All loggers in the process inherit the handler automatically.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def new_request_id() -> str:
    """Return an 8-character hex string suitable as a request identifier."""
    return secrets.token_hex(4)
