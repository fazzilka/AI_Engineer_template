from collections.abc import Sequence

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.domain.chat import ChatMessage, GenerationResult
from app.main import create_app
from app.ports.llm import LLMUnavailableError


class UnavailableLLMClient:
    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        del messages, system_prompt
        raise LLMUnavailableError("private provider detail")


@pytest.mark.asyncio
async def test_health_endpoints(client: AsyncClient) -> None:
    assert (await client.get("/health/live")).json() == {"status": "ok"}
    assert (await client.get("/health/ready")).json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_chat_endpoint_uses_offline_provider(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"x-request-id": "request-123"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"
    assert response.json() == {
        "content": "Fake response: hello",
        "model": "test-model",
        "finish_reason": "stop",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }


@pytest.mark.asyncio
async def test_chat_endpoint_validates_conversation(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "assistant", "content": "hello"}]},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_request_id_is_replaced(client: AsyncClient) -> None:
    response = await client.get("/health/live", headers={"x-request-id": "invalid id"})

    assert response.headers["x-request-id"] != "invalid id"
    assert len(response.headers["x-request-id"]) == 32


@pytest.mark.asyncio
async def test_metrics_endpoint_exports_application_metrics(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "metrics"}]},
    )
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "app_http_requests_total" in response.text
    assert "app_llm_requests_total" in response.text
    assert 'route="/api/v1/chat"' in response.text


@pytest.mark.asyncio
async def test_provider_error_returns_safe_service_unavailable(
    test_settings: Settings,
) -> None:
    app = create_app(settings=test_settings, llm_client=UnavailableLLMClient())

    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        response = await client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "llm_unavailable",
            "message": "The language model is temporarily unavailable",
        }
    }
    assert "private provider detail" not in response.text


@pytest.mark.asyncio
async def test_docs_can_be_disabled(client: AsyncClient) -> None:
    assert (await client.get("/docs")).status_code == 404
