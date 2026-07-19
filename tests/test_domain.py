import pytest

from app.domain.chat import ChatMessage, GenerationResult, MessageRole, TokenUsage
from app.domain.generation import LifecycleState, ModelStatus


@pytest.mark.unit
def test_chat_and_generation_invariants() -> None:
    with pytest.raises(ValueError, match="between 1"):
        ChatMessage(role=MessageRole.USER, content="")
    with pytest.raises(ValueError, match="negative"):
        TokenUsage(input_tokens=-1)
    with pytest.raises(ValueError, match="empty"):
        GenerationResult(content="", model="model")
    with pytest.raises(ValueError, match="alias"):
        GenerationResult(content="answer", model="")


@pytest.mark.unit
def test_token_total_and_model_status_properties() -> None:
    usage = TokenUsage(input_tokens=2, output_tokens=3)
    status = ModelStatus(
        backend="fake",
        model_alias="fake",
        source="built-in",
        device="cpu",
        dtype="n/a",
        state=LifecycleState.LOADING,
        local_files_only=True,
        max_input_tokens=100,
        max_new_tokens=10,
    )

    assert usage.total_tokens == 5
    assert status.loading is True
    assert status.loaded is False
