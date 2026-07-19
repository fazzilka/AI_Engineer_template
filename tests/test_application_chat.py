from collections.abc import Sequence

import pytest

from app.application.chat import ChatService
from app.domain.chat import ChatMessage, GenerationResult, MessageRole


class RecordingLLMClient:
    def __init__(self) -> None:
        self.messages: Sequence[ChatMessage] = ()
        self.system_prompt = ""

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        self.messages = messages
        self.system_prompt = system_prompt
        return GenerationResult(content="answer", model="test-model")


@pytest.mark.asyncio
async def test_chat_service_delegates_to_llm() -> None:
    llm = RecordingLLMClient()
    service = ChatService(llm=llm, system_prompt="be helpful")
    messages = [ChatMessage(role=MessageRole.USER, content="hello")]

    result = await service.reply(messages)

    assert result.content == "answer"
    assert llm.messages == messages
    assert llm.system_prompt == "be helpful"


@pytest.mark.asyncio
async def test_chat_service_rejects_empty_conversation() -> None:
    service = ChatService(llm=RecordingLLMClient(), system_prompt="prompt")

    with pytest.raises(ValueError, match="At least one"):
        await service.reply([])


@pytest.mark.asyncio
async def test_chat_service_requires_final_user_message() -> None:
    service = ChatService(llm=RecordingLLMClient(), system_prompt="prompt")
    messages = [ChatMessage(role=MessageRole.ASSISTANT, content="answer")]

    with pytest.raises(ValueError, match="final chat message"):
        await service.reply(messages)
