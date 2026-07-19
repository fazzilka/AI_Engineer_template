from fastapi import APIRouter

from app.api.routes import chat, system


def build_router(*, api_prefix: str) -> APIRouter:
    router = APIRouter()
    router.include_router(system.router)
    router.include_router(chat.router, prefix=api_prefix)
    return router
