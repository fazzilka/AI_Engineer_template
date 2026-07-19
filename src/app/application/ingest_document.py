from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import PurePath

from app.domain.document_rules import (
    content_checksum,
    document_identity,
    document_version,
    sanitize_filename,
)
from app.domain.documents import (
    DocumentDescriptor,
    DocumentSource,
    IngestionResult,
    IngestionStatus,
    SourceType,
)
from app.domain.errors import (
    DocumentTooLargeError,
    NoExtractableTextError,
    UnsupportedDocumentTypeError,
)
from app.ports.chunking import TextChunker
from app.ports.documents import DocumentParser
from app.ports.embeddings import EmbeddingModel
from app.ports.retrieval import VectorStore


class IngestUploadedDocumentService:
    def __init__(
        self,
        *,
        parsers: Sequence[DocumentParser],
        chunker: TextChunker,
        embeddings: EmbeddingModel,
        vector_store: VectorStore,
        max_file_bytes: int,
    ) -> None:
        self._parsers = parsers
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._max_file_bytes = max_file_bytes

    async def ingest(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> IngestionResult:
        if not content:
            raise NoExtractableTextError
        if len(content) > self._max_file_bytes:
            raise DocumentTooLargeError
        try:
            safe_filename = sanitize_filename(filename)
        except ValueError as exc:
            raise UnsupportedDocumentTypeError("The upload filename is invalid") from exc
        normalized_content_type = content_type.partition(";")[0].strip().lower()
        parser = next(
            (
                candidate
                for candidate in self._parsers
                if candidate.supports(
                    filename=safe_filename,
                    content_type=normalized_content_type,
                )
            ),
            None,
        )
        if parser is None:
            raise UnsupportedDocumentTypeError
        source_type = _source_type(safe_filename, normalized_content_type)
        checksum = content_checksum(content)
        source = DocumentSource(
            source_type=source_type,
            source=safe_filename,
            title=PurePath(safe_filename).stem[:500],
            content_type=normalized_content_type,
            filename=safe_filename,
        )
        descriptor = DocumentDescriptor(
            document_id=document_identity(source),
            document_version=document_version(checksum),
            document_checksum=checksum,
            source=source,
            ingested_at=datetime.now(UTC),
        )
        parsed = parser.parse(content=content, descriptor=descriptor)
        chunks = self._chunker.split(parsed)
        if not chunks:
            raise NoExtractableTextError
        previous_checksum = await self._vector_store.document_checksum(descriptor.document_id)
        if previous_checksum == checksum:
            return IngestionResult(
                document_id=descriptor.document_id,
                document_version=descriptor.document_version,
                status=IngestionStatus.UNCHANGED,
                chunk_count=len(chunks),
            )
        vectors = await self._embeddings.embed_documents([chunk.text for chunk in chunks])
        embedding_status = self._embeddings.status()
        if embedding_status.fingerprint is None:
            raise RuntimeError("Embedding fingerprint is unavailable after embedding")
        await self._vector_store.replace_document(
            chunks=chunks,
            vectors=vectors,
            embedding_fingerprint=embedding_status.fingerprint,
        )
        return IngestionResult(
            document_id=descriptor.document_id,
            document_version=descriptor.document_version,
            status=(
                IngestionStatus.UPDATED
                if previous_checksum is not None
                else IngestionStatus.INDEXED
            ),
            chunk_count=len(chunks),
        )


def _source_type(filename: str, content_type: str) -> SourceType:
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return SourceType.PDF
    if content_type in {"text/markdown", "text/x-markdown"} or filename.lower().endswith(".md"):
        return SourceType.MARKDOWN
    if content_type == "text/plain" or filename.lower().endswith(".txt"):
        return SourceType.TEXT
    raise UnsupportedDocumentTypeError
