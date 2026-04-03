"""Structured logging utilities with secret redaction."""

from __future__ import annotations

import logging
import re
import sys
from typing import Any


_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9]{20,})"),
    re.compile(r"(AIza[A-Za-z0-9_-]{35})"),
    re.compile(r"(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"),
]

_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key", "authorization", "token",
    "secret", "password", "credential", "x-api-key",
})


def _redact_value(value: str) -> str:
    """Redact known secret patterns in a string."""
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub(lambda m: m.group(0)[:6] + "***REDACTED***", value)
    return value


def redact_dict(data: dict[str, Any], *, redact: bool = True) -> dict[str, Any]:
    """Return a shallow copy with sensitive values redacted."""
    if not redact:
        return data
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in _SENSITIVE_KEYS:
            result[key] = "***REDACTED***"
        elif isinstance(value, str):
            result[key] = _redact_value(value)
        elif isinstance(value, dict):
            result[key] = redact_dict(value, redact=redact)
        else:
            result[key] = value
    return result


class StructuredFormatter(logging.Formatter):
    """Formatter that adds correlation context to log records."""

    def format(self, record: logging.LogRecord) -> str:
        # Attach extra fields if present
        extras = []
        for attr in ("request_id", "provider_name", "correlation_id"):
            val = getattr(record, attr, None)
            if val:
                extras.append(f"{attr}={val}")
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        base = super().format(record)
        return f"{base}{extra_str}"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a configured logger for Bijoux modules."""
    logger = logging.getLogger(f"bijoux.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
