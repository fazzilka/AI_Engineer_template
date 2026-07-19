from collections.abc import Sequence
from typing import Protocol

from app.domain.chat import ChatMessage, GenerationResult
from app.domain.errors import ModelUnavailableError
from app.domain.generation import ModelStatus


class ChatModel(Protocol):
    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult: ...

    async def count_tokens(self, text: str) -> int | None: ...


class ModelLifecycle(Protocol):
    async def load(self) -> None: ...

    async def warmup(self) -> None: ...

    def status(self) -> ModelStatus: ...

    async def aclose(self) -> None: ...


class ManagedChatModel(ChatModel, ModelLifecycle, Protocol):
    pass


# Compatibility alias for projects built from earlier template revisions.
LLMClient = ChatModel
LLMUnavailableError = ModelUnavailableError
