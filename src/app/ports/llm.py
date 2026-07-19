from collections.abc import Sequence
from typing import Protocol

from app.domain.chat import ChatMessage, GenerationResult


class LLMError(RuntimeError):
    """Base error raised by an LLM adapter."""


class LLMUnavailableError(LLMError):
    """The configured LLM provider could not serve the request."""


class LLMClient(Protocol):
    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult: ...
