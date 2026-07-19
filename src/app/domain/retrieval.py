from dataclasses import dataclass

from app.domain.chat import GenerationResult
from app.domain.documents import DocumentChunk, SourceType


@dataclass(frozen=True, slots=True)
class RetrievalFilter:
    document_ids: tuple[str, ...] = ()
    source_types: tuple[SourceType, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    document_id: str
    chunk_id: str
    title: str
    source: str
    source_type: SourceType
    page_number: int | None
    score: float
    snippet: str


@dataclass(frozen=True, slots=True)
class RagResult:
    generation: GenerationResult
    sources: tuple[Citation, ...]
