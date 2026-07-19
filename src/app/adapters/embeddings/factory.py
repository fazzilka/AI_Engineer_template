from app.adapters.embeddings.fake import FakeEmbeddingModel
from app.adapters.embeddings.huggingface import HuggingFaceEmbeddingModel
from app.config import EmbeddingSettings, ModelBackend
from app.ports.embeddings import EmbeddingModel


def build_embedding_model(settings: EmbeddingSettings) -> EmbeddingModel:
    if settings.backend is ModelBackend.FAKE:
        return FakeEmbeddingModel(
            dimension=settings.fake_dimension,
            normalize=settings.normalize,
        )
    return HuggingFaceEmbeddingModel(settings)
