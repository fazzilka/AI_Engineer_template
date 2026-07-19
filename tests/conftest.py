from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import LLMSettings, Settings
from app.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        app_env="test",
        docs_enabled=False,
        llm=LLMSettings(provider="fake", model="test-model"),
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
