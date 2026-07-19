import logging

import structlog

from app.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structured logs once at application startup."""

    log_level = getattr(logging, settings.observability.log_level)
    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.app.environment == "local"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
