from app.adapters.llm.fake import FakeChatModel
from app.adapters.llm.huggingface import HuggingFaceChatModel
from app.config import ModelBackend, ModelSettings
from app.ports.llm import ManagedChatModel


def build_chat_model(settings: ModelSettings) -> ManagedChatModel:
    if settings.backend is ModelBackend.FAKE:
        return FakeChatModel(model=settings.alias)
    return HuggingFaceChatModel(settings)


build_llm_client = build_chat_model
