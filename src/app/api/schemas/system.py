from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    components: dict[str, str] | None = None


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    model_alias: str
    source: str
    device: str
    dtype: str
    loaded: bool
    loading: bool
    local_files_only: bool
    context_limit: int
    max_new_tokens: int
    embedding_backend: str
    embedding_model_alias: str
    embedding_dimension: int | None
    qdrant_mode: str
    retrieval_mode: str
