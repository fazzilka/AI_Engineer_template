from collections.abc import Sequence

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.llm.fake import FakeChatModel
from app.config import ApiSettings, IngestionSettings, Settings
from app.domain.chat import ChatMessage, GenerationResult
from app.domain.errors import ModelUnavailableError
from app.main import create_app


class UnavailableChatModel(FakeChatModel):
    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        del messages, system_prompt
        raise ModelUnavailableError("private local path and internal detail")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_endpoints(client: AsyncClient) -> None:
    assert (await client.get("/health/live")).json() == {
        "status": "ok",
        "components": None,
    }
    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready_with_lazy_model"
    assert response.json()["components"]["vector_store"] == "ready"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_endpoint_uses_offline_model(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"x-request-id": "request-123"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"
    payload = response.json()
    assert payload["content"] == "Fake response: hello"
    assert payload["model"] == "test-model"
    assert payload["usage"]["total_tokens"] == (
        payload["usage"]["input_tokens"] + payload["usage"]["output_tokens"]
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_validation_has_stable_error_contract(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "assistant", "content": "hello"}]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["request_id"] != "unknown"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_request_id_is_replaced(client: AsyncClient) -> None:
    response = await client.get("/health/live", headers={"x-request-id": "invalid id"})

    assert response.headers["x-request-id"] != "invalid id"
    assert len(response.headers["x-request-id"]) == 32


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_retrieve_rag_and_delete(client: AsyncClient) -> None:
    upload = await client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "knowledge.md",
                b"# Local facts\nQdrant stores vectors locally and supports dense retrieval.",
                "text/markdown",
            )
        },
    )
    assert upload.status_code == 200
    document_id = upload.json()["document_id"]
    assert upload.json()["status"] == "indexed"

    repeated = await client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "knowledge.md",
                b"# Local facts\nQdrant stores vectors locally and supports dense retrieval.",
                "text/markdown",
            )
        },
    )
    assert repeated.json()["status"] == "unchanged"

    updated = await client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "knowledge.md",
                b"# Local facts\nQdrant stores updated vectors locally.",
                "text/markdown",
            )
        },
    )
    assert updated.json()["status"] == "updated"

    retrieval = await client.post(
        "/api/v1/retrieval/search",
        json={"query": "Where are vectors stored?", "document_ids": []},
    )
    assert retrieval.status_code == 422  # raw filter fields are deliberately rejected

    retrieval = await client.post(
        "/api/v1/retrieval/search",
        json={"query": "Where are vectors stored?", "filters": {"document_ids": [document_id]}},
    )
    assert retrieval.status_code == 200
    assert retrieval.json()["results"][0]["document_id"] == document_id

    rag = await client.post(
        "/api/v1/rag/query",
        json={"query": "Where are vectors stored?", "filters": {"document_ids": [document_id]}},
    )
    assert rag.status_code == 200
    assert "Qdrant stores updated vectors locally" in rag.json()["answer"]
    assert rag.json()["sources"][0]["document_id"] == document_id
    assert len(rag.json()["sources"][0]["snippet"]) <= 300

    deleted = await client.delete(f"/api/v1/documents/{document_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert (await client.delete(f"/api/v1/documents/{document_id}")).status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_rejects_unsupported_content(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("image.png", b"not an image", "image/png")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_document_type"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_url_ingestion_can_be_disabled(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents/url",
        json={"url": "https://example.com/article"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsafe_url"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_status_is_safe(client: AsyncClient) -> None:
    payload = (await client.get("/api/v1/system/model")).json()

    assert payload["backend"] == "fake"
    assert payload["embedding_backend"] == "fake"
    assert payload["qdrant_mode"] == "memory"
    assert "path" not in payload


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_exports_ai_metrics(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "metrics"}]},
    )
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "app_http_requests_total" in response.text
    assert "ai_generation_requests_total" in response.text
    assert "ai_embedding_requests_total" in response.text
    assert "ai_retrieval_requests_total" in response.text
    assert 'route="/api/v1/chat"' in response.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_error_returns_safe_service_unavailable(test_settings: Settings) -> None:
    app = create_app(settings=test_settings, llm_client=UnavailableChatModel())

    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as local_client,
    ):
        response = await local_client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    assert "private local path" not in response.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_docs_can_be_disabled(client: AsyncClient) -> None:
    assert (await client.get("/docs")).status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_request_body_limit_rejects_before_parsing(test_settings: Settings) -> None:
    settings = Settings(
        app=test_settings.app,
        api=ApiSettings(docs_enabled=False, max_request_body_bytes=100),
        model=test_settings.model,
        qdrant=test_settings.qdrant,
        web=test_settings.web,
        ingestion=IngestionSettings(max_file_bytes=10),
    )
    app = create_app(settings=settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as local_client,
    ):
        response = await local_client.post(
            "/api/v1/documents/upload",
            files={"file": ("large.txt", b"x" * 200, "text/plain")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "document_too_large"
    assert response.json()["error"]["request_id"] != "unknown"
