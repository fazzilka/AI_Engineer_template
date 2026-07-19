from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from app.domain.documents import SourceType
from app.domain.retrieval import RetrievalFilter, RetrievedChunk


class RetrievalFilterPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: list[str] = Field(default_factory=list, max_length=100)
    source_types: list[SourceType] = Field(default_factory=list, max_length=10)

    def to_domain(self) -> RetrievalFilter:
        return RetrievalFilter(
            document_ids=tuple(self.document_ids),
            source_types=tuple(self.source_types),
        )


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=32_000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    filters: RetrievalFilterPayload = Field(default_factory=RetrievalFilterPayload)


class RetrievedChunkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_version: str
    chunk_id: str
    chunk_index: int
    title: str
    source: str
    source_type: SourceType
    page_number: int | None
    score: float
    snippet: str

    @classmethod
    def from_domain(cls, item: RetrievedChunk, *, snippet_characters: int = 500) -> Self:
        chunk = item.chunk
        return cls(
            document_id=chunk.document_id,
            document_version=chunk.document_version,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            title=chunk.title,
            source=chunk.source,
            source_type=chunk.source_type,
            page_number=chunk.page_number,
            score=item.score,
            snippet=chunk.text[:snippet_characters],
        )


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[RetrievedChunkResponse]
