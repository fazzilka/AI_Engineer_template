from collections.abc import Sequence

from app.domain.chat import ChatMessage, GenerationResult, TokenUsage
from app.domain.errors import ModelUnavailableError
from app.domain.generation import LifecycleState, ModelStatus


class FakeChatModel:
    """Deterministic local adapter used by tests, evals, and the zero-download quick start."""

    def __init__(
        self,
        *,
        model: str = "fake-model",
        fail_on: str | None = None,
    ) -> None:
        self._model = model
        self._fail_on = fail_on
        self._state = LifecycleState.UNLOADED

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        await self.load()
        last_message = messages[-1]
        if self._fail_on and self._fail_on in last_message.content:
            raise ModelUnavailableError("Controlled fake model failure")
        context = _first_source_content(last_message.content)
        content = (
            f"Fake grounded answer: {context}"
            if context
            else f"Fake response: {last_message.content}"
        )
        input_tokens = sum(len(message.content.split()) for message in messages)
        input_tokens += len(system_prompt.split())
        output_tokens = len(content.split())
        return GenerationResult(
            content=content,
            model=self._model,
            finish_reason="stop",
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    async def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def load(self) -> None:
        self._state = LifecycleState.READY

    async def warmup(self) -> None:
        await self.load()

    def status(self) -> ModelStatus:
        return ModelStatus(
            backend="fake",
            model_alias=self._model,
            source="built-in",
            device="cpu",
            dtype="n/a",
            state=self._state,
            local_files_only=True,
            max_input_tokens=4_096,
            max_new_tokens=512,
        )

    async def aclose(self) -> None:
        self._state = LifecycleState.CLOSED


def _first_source_content(prompt: str) -> str:
    marker = "<source id="
    source_start = prompt.find(marker)
    if source_start < 0:
        return ""
    content_start = prompt.find(">", source_start)
    content_end = prompt.find("</source>", content_start)
    if content_start < 0 or content_end < 0:
        return ""
    return " ".join(prompt[content_start + 1 : content_end].split())[:500]


FakeLLMClient = FakeChatModel
