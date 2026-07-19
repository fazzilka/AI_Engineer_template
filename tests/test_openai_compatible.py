import httpx
import pytest
from openai import AsyncOpenAI

from app.adapters.llm.openai_compatible import OpenAICompatibleLLMClient
from app.config import LLMSettings
from app.domain.chat import ChatMessage, MessageRole
from app.ports.llm import LLMUnavailableError


def _settings(*, max_retries: int = 0) -> LLMSettings:
    return LLMSettings(
        provider="openai_compatible",
        api_key="test-key",
        base_url="https://llm.test/v1",
        model="test-model",
        max_retries=max_retries,
    )


def _sdk_client(handler: httpx.AsyncBaseTransport) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://llm.test/v1",
        http_client=httpx.AsyncClient(transport=handler),
    )


@pytest.mark.asyncio
async def test_adapter_maps_provider_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )

    adapter = OpenAICompatibleLLMClient(
        _settings(),
        client=_sdk_client(httpx.MockTransport(handler)),
    )

    result = await adapter.generate(
        messages=[
            ChatMessage(role=MessageRole.USER, content="ping"),
            ChatMessage(role=MessageRole.ASSISTANT, content="previous"),
            ChatMessage(role=MessageRole.USER, content="ping again"),
        ],
        system_prompt="system",
    )
    await adapter.aclose()

    assert result.content == "pong"
    assert result.model == "served-model"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_adapter_wraps_provider_errors() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(401, json={"error": {}}))
    adapter = OpenAICompatibleLLMClient(_settings(), client=_sdk_client(transport))

    with pytest.raises(LLMUnavailableError, match="provider request failed"):
        await adapter.generate(
            messages=[ChatMessage(role=MessageRole.USER, content="ping")],
            system_prompt="system",
        )

    await adapter.aclose()


def test_adapter_requires_api_key() -> None:
    settings = LLMSettings()
    settings.provider = "openai_compatible"

    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleLLMClient(settings)
