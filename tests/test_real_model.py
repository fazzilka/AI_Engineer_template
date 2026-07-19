import os
from pathlib import Path

import anyio
import pytest

from app.adapters.llm.huggingface import HuggingFaceChatModel
from app.config import ModelBackend, ModelSettings, ModelSource
from app.domain.chat import ChatMessage, MessageRole


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_pre_downloaded_local_model_smoke() -> None:
    configured_path = os.environ.get("TEST_MODEL_PATH")
    if not configured_path:
        pytest.skip("Set TEST_MODEL_PATH to a pre-downloaded tiny local model")
    path = Path(configured_path)
    if not await anyio.Path(path).is_dir():
        pytest.skip("TEST_MODEL_PATH is not an existing local directory")
    model = HuggingFaceChatModel(
        ModelSettings(
            backend=ModelBackend.HUGGINGFACE,
            source=ModelSource.FILESYSTEM,
            path=path,
            local_files_only=True,
            max_new_tokens=8,
            timeout_seconds=120,
            alias="smoke-model",
        )
    )
    try:
        await model.load()
        result = await model.generate(
            messages=(ChatMessage(role=MessageRole.USER, content="Reply with OK."),),
            system_prompt="Answer briefly.",
        )
        assert result.content.strip()
    finally:
        await model.aclose()
