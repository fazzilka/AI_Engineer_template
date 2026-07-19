from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import (
    ApiSettings,
    AppSettings,
    ModelSettings,
    QdrantMode,
    QdrantSettings,
    Settings,
    WebFetchSettings,
)
from app.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        app=AppSettings(environment="test", allow_fake_backends=True),
        api=ApiSettings(docs_enabled=False),
        model=ModelSettings(alias="test-model"),
        qdrant=QdrantSettings(mode=QdrantMode.MEMORY),
        web=WebFetchSettings(enabled=False),
    )


@pytest_asyncio.fixture
async def client(test_settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings=test_settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as test_client,
    ):
        yield test_client
