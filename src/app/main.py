from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.adapters.llm import build_llm_client
from app.api.router import build_router
from app.application.chat import ChatService
from app.config import Settings, get_settings
from app.observability.logging import configure_logging
from app.observability.metrics import InstrumentedLLMClient
from app.observability.middleware import RequestObservabilityMiddleware
from app.ports.llm import LLMClient, LLMUnavailableError
from app.prompts import load_system_prompt


def create_app(
    *,
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings)
        logger = structlog.get_logger(__name__)
        owned_client = None
        client = llm_client
        if client is None:
            owned_client = build_llm_client(resolved_settings.llm)
            client = owned_client

        application.state.chat_service = ChatService(
            llm=InstrumentedLLMClient(
                client=client,
                provider=resolved_settings.llm.provider,
            ),
            system_prompt=load_system_prompt(),
        )
        logger.info(
            "application_started",
            environment=resolved_settings.app_env,
            llm_provider=resolved_settings.llm.provider,
        )
        try:
            yield
        finally:
            if owned_client is not None:
                await owned_client.aclose()
            logger.info("application_stopped")

    docs_url = "/docs" if resolved_settings.docs_enabled else None
    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(RequestObservabilityMiddleware)
    app.include_router(build_router(api_prefix=resolved_settings.api_prefix))

    @app.exception_handler(LLMUnavailableError)
    async def handle_llm_unavailable(
        _request: Request,
        exc: LLMUnavailableError,
    ) -> JSONResponse:
        structlog.get_logger(__name__).warning("llm_unavailable", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": "llm_unavailable",
                    "message": "The language model is temporarily unavailable",
                }
            },
        )

    return app


app = create_app()


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
