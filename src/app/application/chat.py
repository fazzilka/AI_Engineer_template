from collections.abc import Sequence

from app.domain.chat import ChatMessage, GenerationResult, MessageRole
from app.ports.llm import ChatModel


class ChatService:
    def __init__(self, *, model: ChatModel, system_prompt: str) -> None:
        self._model = model
        self._system_prompt = system_prompt

    async def reply(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        if not messages:
            msg = "At least one chat message is required"
            raise ValueError(msg)
        if messages[-1].role is not MessageRole.USER:
            msg = "The final chat message must have the user role"
            raise ValueError(msg)

        return await self._model.generate(messages=messages, system_prompt=self._system_prompt)
