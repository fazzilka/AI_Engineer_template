from typing import cast

from fastapi import Request

from app.application.chat import ChatService


def get_chat_service(request: Request) -> ChatService:
    return cast(ChatService, request.app.state.chat_service)
