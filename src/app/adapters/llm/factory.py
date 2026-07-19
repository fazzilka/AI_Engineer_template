from typing import Protocol

from app.adapters.llm.fake import FakeLLMClient
from app.adapters.llm.openai_compatible import OpenAICompatibleLLMClient
from app.config import LLMSettings
from app.ports.llm import LLMClient


class ManagedLLMClient(LLMClient, Protocol):
    async def aclose(self) -> None: ...


def build_llm_client(settings: LLMSettings) -> ManagedLLMClient:
    if settings.provider == "fake":
        return FakeLLMClient(model=settings.model)
    return OpenAICompatibleLLMClient(settings)
