from enum import StrEnum
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Literal, Self

from pydantic import AnyHttpUrl, BaseModel, Field, PositiveFloat, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.errors import ConfigurationError


class ModelBackend(StrEnum):
    FAKE = "fake"
    HUGGINGFACE = "huggingface"


class ModelSource(StrEnum):
    FILESYSTEM = "filesystem"
    CACHE = "cache"


class Device(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class QdrantMode(StrEnum):
    MEMORY = "memory"
    LOCAL = "local"
    SERVER = "server"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"


class AppSettings(BaseModel):
    name: str = "AI Engineer Template"
    environment: Literal["local", "test", "staging", "production"] = "local"
    allow_fake_backends: bool = False


class ApiSettings(BaseModel):
    prefix: str = "/api/v1"
    docs_enabled: bool = True
    max_request_body_bytes: PositiveInt = 22_020_096


class ModelSettings(BaseModel):
    backend: ModelBackend = ModelBackend.FAKE
    source: ModelSource = ModelSource.FILESYSTEM
    id: str = ""
    path: Path = Path("models/generator")
    revision: str = ""
    local_files_only: bool = True
    trust_remote_code: bool = False
    device: Device = Device.AUTO
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    task: Literal["text-generation", "text2text-generation"] = "text-generation"
    load_on_startup: bool = False
    max_input_tokens: PositiveInt = 4_096
    max_new_tokens: PositiveInt = 512
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    do_sample: bool = False
    repetition_penalty: float = Field(default=1.05, gt=0, le=5)
    concurrency: PositiveInt = 1
    timeout_seconds: PositiveFloat = 120
    alias: str = "fake-model"

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.backend is ModelBackend.HUGGINGFACE:
            if self.source is ModelSource.CACHE and not self.id:
                msg = "MODEL__ID is required when MODEL__SOURCE=cache"
                raise ValueError(msg)
            if (
                self.source is ModelSource.FILESYSTEM
                and self.load_on_startup
                and not self.path.is_dir()
            ):
                msg = "MODEL__PATH must be an existing directory for eager filesystem loading"
                raise ValueError(msg)
        return self


class EmbeddingSettings(BaseModel):
    backend: ModelBackend = ModelBackend.FAKE
    source: ModelSource = ModelSource.FILESYSTEM
    id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    path: Path = Path("models/embeddings")
    revision: str = ""
    local_files_only: bool = True
    device: Device = Device.AUTO
    normalize: bool = True
    batch_size: PositiveInt = 32
    query_prefix: str = ""
    document_prefix: str = ""
    fake_dimension: PositiveInt = 64
    alias: str = "local-embeddings"

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.backend is ModelBackend.HUGGINGFACE:
            if self.source is ModelSource.CACHE and not self.id:
                msg = "EMBEDDINGS__ID is required when EMBEDDINGS__SOURCE=cache"
                raise ValueError(msg)
            if self.source is ModelSource.FILESYSTEM and not self.path.is_dir():
                msg = "EMBEDDINGS__PATH must be an existing directory"
                raise ValueError(msg)
        return self


class QdrantSettings(BaseModel):
    mode: QdrantMode = QdrantMode.LOCAL
    path: Path = Path("data/qdrant")
    url: AnyHttpUrl = AnyHttpUrl("http://localhost:6333")
    collection: str = Field(default="documents", pattern=r"^[A-Za-z0-9_-]{1,128}$")
    distance: DistanceMetric = DistanceMetric.COSINE
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE
    top_k: PositiveInt = 5
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    prefer_grpc: bool = False
    request_timeout_seconds: PositiveFloat = 10
    sparse_model_id: str = "Qdrant/bm25"
    sparse_cache_dir: Path = Path("models/fastembed")

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode is QdrantMode.LOCAL and not str(self.path):
            msg = "QDRANT__PATH is required in local mode"
            raise ValueError(msg)
        if self.retrieval_mode is not RetrievalMode.DENSE and find_spec("fastembed") is None:
            msg = (
                "Sparse and hybrid retrieval require the optional dependency: "
                "uv sync --extra hybrid"
            )
            raise ValueError(msg)
        return self


class ChunkingSettings(BaseModel):
    chunk_size: PositiveInt = 1_000
    chunk_overlap: int = Field(default=150, ge=0)
    separators: tuple[str, ...] | None = None
    length_function: Literal["characters"] = "characters"

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            msg = "CHUNKING__CHUNK_OVERLAP must be smaller than CHUNKING__CHUNK_SIZE"
            raise ValueError(msg)
        return self


class IngestionSettings(BaseModel):
    max_file_bytes: PositiveInt = 20_971_520
    max_pdf_pages: PositiveInt = 500
    max_extracted_characters: PositiveInt = 5_000_000
    max_files_per_request: PositiveInt = 1


class WebFetchSettings(BaseModel):
    enabled: bool = True
    max_redirects: int = Field(default=3, ge=0, le=10)
    connect_timeout_seconds: PositiveFloat = 5
    read_timeout_seconds: PositiveFloat = 15
    max_response_bytes: PositiveInt = 5_242_880
    allow_private_hosts: bool = False
    user_agent: str = "AI-Engineer-Template/1.0"
    max_retries: int = Field(default=2, ge=0, le=5)


class RagSettings(BaseModel):
    top_k: PositiveInt = 5
    max_context_chunks: PositiveInt = 8
    max_context_characters: PositiveInt = 16_000
    max_context_tokens: PositiveInt = 3_000
    return_sources: bool = True
    min_relevant_chunks: PositiveInt = 1
    citation_snippet_characters: PositiveInt = 300


class ObservabilitySettings(BaseModel):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class Settings(BaseSettings):
    """Server-owned configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="ignore",
    )

    offline_mode: bool = False
    app: AppSettings = Field(default_factory=AppSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    web: WebFetchSettings = Field(default_factory=WebFetchSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode="after")
    def validate_application(self) -> Self:
        if (
            self.app.environment == "production"
            and not self.app.allow_fake_backends
            and (
                self.model.backend is ModelBackend.FAKE
                or self.embeddings.backend is ModelBackend.FAKE
            )
        ):
            msg = "Fake backends in production require APP__ALLOW_FAKE_BACKENDS=true"
            raise ValueError(msg)
        if self.rag.max_context_tokens > self.model.max_input_tokens:
            msg = "RAG__MAX_CONTEXT_TOKENS cannot exceed MODEL__MAX_INPUT_TOKENS"
            raise ValueError(msg)
        if self.api.max_request_body_bytes <= self.ingestion.max_file_bytes:
            msg = "API__MAX_REQUEST_BODY_BYTES must exceed INGESTION__MAX_FILE_BYTES"
            raise ValueError(msg)
        if self.offline_mode:
            self.model.local_files_only = True
            self.embeddings.local_files_only = True
            self.web.enabled = False
        return self


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValueError as exc:
        raise ConfigurationError from exc
