from collections.abc import Sequence

from app.domain.chat import ChatMessage, GenerationResult, TokenUsage


class FakeLLMClient:
    """Deterministic offline adapter for local development and tests."""

    def __init__(self, *, model: str = "fake-model") -> None:
        self._model = model

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        del system_prompt
        last_message = messages[-1]
        return GenerationResult(
            content=f"Fake response: {last_message.content}",
            model=self._model,
            finish_reason="stop",
            usage=TokenUsage(input_tokens=len(messages), output_tokens=2),
        )

    async def aclose(self) -> None:
        return None
