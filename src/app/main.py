import uvicorn
from fastapi import FastAPI

from app.api.body_limit import RequestBodyLimitMiddleware
from app.api.errors import install_error_handlers
from app.api.router import build_router
from app.bootstrap.container import ApplicationContainer
from app.bootstrap.lifespan import build_lifespan
from app.config import Settings, get_settings
from app.observability.middleware import RequestObservabilityMiddleware
from app.ports.llm import ManagedChatModel


def create_app(
    *,
    settings: Settings | None = None,
    llm_client: ManagedChatModel | None = None,
    container: ApplicationContainer | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    docs_url = "/docs" if resolved_settings.api.docs_enabled else None
    application = FastAPI(
        title=resolved_settings.app.name,
        version="0.2.0",
        docs_url=docs_url,
        redoc_url=None,
        lifespan=build_lifespan(
            settings=resolved_settings,
            container=container,
            model=llm_client,
        ),
    )
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=resolved_settings.api.max_request_body_bytes,
    )
    application.add_middleware(RequestObservabilityMiddleware)
    application.include_router(build_router(api_prefix=resolved_settings.api.prefix))
    install_error_handlers(application)
    return application


app = create_app()


def run() -> None:
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # noqa: S104 -- local service intentionally listens on all interfaces
        port=8000,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    run()
