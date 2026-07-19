from fastapi import APIRouter

from app.api.routes import chat, documents, rag, retrieval, system


def build_router(*, api_prefix: str) -> APIRouter:
    router = APIRouter()
    router.include_router(system.health_router)
    router.include_router(chat.router, prefix=api_prefix)
    router.include_router(documents.router, prefix=api_prefix)
    router.include_router(retrieval.router, prefix=api_prefix)
    router.include_router(rag.router, prefix=api_prefix)
    router.include_router(system.api_router, prefix=api_prefix)
    return router
