from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas.chat import TokenUsageResponse
from app.api.schemas.retrieval import RetrievalFilterPayload
from app.domain.documents import SourceType
from app.domain.retrieval import Citation, RagResult


class RagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=32_000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    filters: RetrievalFilterPayload = Field(default_factory=RetrievalFilterPayload)


class CitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    document_id: str
    chunk_id: str
    title: str
    source: str
    source_type: SourceType
    page_number: int | None
    score: float
    snippet: str

    @classmethod
    def from_domain(cls, citation: Citation) -> Self:
        return cls(**{field: getattr(citation, field) for field in cls.model_fields})


class RagResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    model: str
    usage: TokenUsageResponse
    sources: list[CitationResponse]

    @classmethod
    def from_domain(cls, result: RagResult) -> Self:
        return cls(
            answer=result.generation.content,
            model=result.generation.model,
            usage=TokenUsageResponse.from_domain(result.generation.usage),
            sources=[CitationResponse.from_domain(source) for source in result.sources],
        )
