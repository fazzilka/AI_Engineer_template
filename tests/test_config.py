from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import (
    ApiSettings,
    AppSettings,
    ChunkingSettings,
    EmbeddingSettings,
    IngestionSettings,
    ModelBackend,
    ModelSettings,
    ModelSource,
    QdrantMode,
    QdrantSettings,
    RagSettings,
    RetrievalMode,
    Settings,
    get_settings,
)


@pytest.mark.unit
def test_settings_load_nested_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__ENVIRONMENT", "staging")
    monkeypatch.setenv("MODEL__ALIAS", "custom-local-model")
    monkeypatch.setenv("QDRANT__MODE", "memory")

    settings = Settings()

    assert settings.app.environment == "staging"
    assert settings.model.alias == "custom-local-model"
    assert settings.qdrant.mode is QdrantMode.MEMORY


@pytest.mark.unit
def test_cache_model_requires_id() -> None:
    with pytest.raises(ValidationError, match="MODEL__ID"):
        ModelSettings(backend=ModelBackend.HUGGINGFACE, source=ModelSource.CACHE, id="")


@pytest.mark.unit
def test_eager_filesystem_model_requires_directory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="existing directory"):
        ModelSettings(
            backend=ModelBackend.HUGGINGFACE,
            source=ModelSource.FILESYSTEM,
            path=tmp_path / "missing",
            load_on_startup=True,
        )


@pytest.mark.unit
def test_huggingface_embeddings_require_local_directory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="EMBEDDINGS__PATH"):
        EmbeddingSettings(
            backend=ModelBackend.HUGGINGFACE,
            source=ModelSource.FILESYSTEM,
            path=tmp_path / "missing",
        )


@pytest.mark.unit
def test_chunk_overlap_must_be_smaller() -> None:
    with pytest.raises(ValidationError, match="CHUNKING__CHUNK_OVERLAP"):
        ChunkingSettings(chunk_size=100, chunk_overlap=100)


@pytest.mark.unit
def test_rag_budget_must_fit_model_context() -> None:
    with pytest.raises(ValidationError, match="RAG__MAX_CONTEXT_TOKENS"):
        Settings(
            model=ModelSettings(max_input_tokens=100),
            rag=RagSettings(max_context_tokens=101),
        )


@pytest.mark.unit
def test_production_rejects_accidental_fake_backends() -> None:
    with pytest.raises(ValidationError, match="Fake backends"):
        Settings(app=AppSettings(environment="production"))


@pytest.mark.unit
def test_offline_mode_forces_local_files_and_disables_web() -> None:
    settings = Settings(offline_mode=True)

    assert settings.model.local_files_only is True
    assert settings.embeddings.local_files_only is True
    assert settings.web.enabled is False


@pytest.mark.unit
def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()


@pytest.mark.unit
def test_local_qdrant_accepts_path(tmp_path: Path) -> None:
    settings = QdrantSettings(mode=QdrantMode.LOCAL, path=tmp_path / "vectors")

    assert settings.path.name == "vectors"


@pytest.mark.unit
def test_request_body_limit_must_exceed_upload_limit() -> None:
    with pytest.raises(ValidationError, match="MAX_REQUEST_BODY_BYTES"):
        Settings(
            api=ApiSettings(max_request_body_bytes=100),
            ingestion=IngestionSettings(max_file_bytes=100),
        )


@pytest.mark.unit
def test_hybrid_retrieval_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.find_spec", lambda _name: None)

    with pytest.raises(ValidationError, match="optional dependency"):
        QdrantSettings(retrieval_mode=RetrievalMode.HYBRID)
