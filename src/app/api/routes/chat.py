from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_container
from app.api.schemas import ChatRequest, ChatResponse
from app.bootstrap.container import ApplicationContainer

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def create_chat_completion(
    payload: ChatRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> ChatResponse:
    result = await container.chat.reply([message.to_domain() for message in payload.messages])
    return ChatResponse.from_result(result)
