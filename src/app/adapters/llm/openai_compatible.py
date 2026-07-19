from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import LLMSettings
from app.domain.chat import ChatMessage, GenerationResult, MessageRole, TokenUsage
from app.ports.llm import LLMUnavailableError

TRANSIENT_PROVIDER_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class OpenAICompatibleLLMClient:
    """Adapter for OpenAI and servers exposing the Chat Completions API."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if settings.api_key is None:
            msg = "An API key is required by the OpenAI-compatible adapter"
            raise ValueError(msg)

        self._client = client or AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )
        self._model = settings.model
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens
        self._max_retries = settings.max_retries

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        payload = self._to_provider_messages(messages=messages, system_prompt=system_prompt)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_random_exponential(multiplier=0.5, max=8),
                retry=retry_if_exception_type(TRANSIENT_PROVIDER_ERRORS),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=payload,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
        except OpenAIError as exc:
            raise LLMUnavailableError("The LLM provider request failed") from exc

        if not response.choices:
            raise LLMUnavailableError("The LLM provider returned no choices")

        choice = response.choices[0]
        if choice.message.content is None:
            raise LLMUnavailableError("The LLM provider returned no text content")

        usage = response.usage
        return GenerationResult(
            content=choice.message.content,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )

    async def aclose(self) -> None:
        await self._client.close()

    @staticmethod
    def _to_provider_messages(
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> list[ChatCompletionMessageParam]:
        payload: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        for message in messages:
            if message.role is MessageRole.USER:
                payload.append({"role": "user", "content": message.content})
            else:
                payload.append({"role": "assistant", "content": message.content})
        return payload
