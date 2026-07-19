from collections.abc import Sequence
from typing import Protocol

from app.domain.documents import DocumentChunk
from app.domain.retrieval import RetrievalFilter, RetrievedChunk


class VectorStore(Protocol):
    async def initialize(self, *, dimension: int, embedding_fingerprint: str) -> None: ...

    async def health_check(self) -> bool: ...

    async def document_checksum(self, document_id: str) -> str | None: ...

    async def replace_document(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        embedding_fingerprint: str,
    ) -> None: ...

    async def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None,
        filters: RetrievalFilter,
    ) -> tuple[RetrievedChunk, ...]: ...

    async def delete_document(self, document_id: str) -> bool: ...

    async def count(self) -> int: ...

    async def aclose(self) -> None: ...
