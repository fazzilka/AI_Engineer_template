from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import structlog
from fastapi import FastAPI

from app.bootstrap.container import ApplicationContainer, build_container
from app.config import Settings
from app.observability.logging import configure_logging
from app.ports.llm import ManagedChatModel


def build_lifespan(
    *,
    settings: Settings,
    container: ApplicationContainer | None = None,
    model: ManagedChatModel | None = None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings)
        logger = structlog.get_logger(__name__)
        resolved = container or build_container(settings, model=model)
        application.state.container = resolved
        await resolved.start()
        logger.info(
            "application_started",
            environment=settings.app.environment,
            model_backend=settings.model.backend.value,
            qdrant_mode=settings.qdrant.mode.value,
        )
        try:
            yield
        finally:
            await resolved.aclose()
            logger.info("application_stopped")

    return lifespan
