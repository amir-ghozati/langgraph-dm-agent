"""Structured logging.

This replaces the source system's error handling wholesale. 68 of its 149 nodes
(34 Google Sheets error-appends plus the 34 timestamp nodes that exist only to
feed them) are error logging, because n8n has no exception handling. Here it is
a logger.
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.config import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    settings = settings or get_settings()

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def configure_console() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to cp1252, which cannot encode an emoji — and the
    composed reply is *required* to contain one to three of them, so the
    primary development channel crashes on its own output. `errors="replace"`
    rather than strict: a terminal that cannot render a glyph should show a
    box, not kill the conversation.
    """
    import contextlib
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):  # detached stream
                reconfigure(encoding="utf-8", errors="replace")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
