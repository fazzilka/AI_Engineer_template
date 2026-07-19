import pytest

from app.bootstrap.container import build_container
from app.config import AppSettings, QdrantMode, QdrantSettings, Settings
from app.domain.documents import FetchedDocument, IngestionStatus


class Fetcher:
    def __init__(self) -> None:
        self.content = b"<html><title>Local article</title><p>Qdrant keeps vectors.</p></html>"
        self.closed = False

    async def fetch(self, url: str) -> FetchedDocument:
        return FetchedDocument(
            content=self.content,
            content_type="text/html",
            final_url=url,
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_url_ingestion_uses_shared_pipeline_and_is_idempotent() -> None:
    fetcher = Fetcher()
    container = build_container(
        Settings(
            app=AppSettings(environment="test", allow_fake_backends=True),
            qdrant=QdrantSettings(mode=QdrantMode.MEMORY),
        ),
        web_fetcher=fetcher,
    )
    await container.start()
    try:
        first = await container.ingest_url.ingest("https://example.com/article")
        second = await container.ingest_url.ingest("https://example.com/article")
        fetcher.content = b"<html><p>Updated vectors stay local.</p></html>"
        third = await container.ingest_url.ingest("https://example.com/article")

        assert first.status is IngestionStatus.INDEXED
        assert second.status is IngestionStatus.UNCHANGED
        assert third.status is IngestionStatus.UPDATED
        assert await container.vector_store.count() == 1
    finally:
        await container.aclose()
    assert fetcher.closed
