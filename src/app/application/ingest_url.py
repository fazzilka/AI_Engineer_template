from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import urlsplit

from app.domain.document_rules import content_checksum, document_identity, document_version
from app.domain.documents import (
    DocumentDescriptor,
    DocumentSource,
    IngestionResult,
    IngestionStatus,
    SourceType,
)
from app.domain.errors import NoExtractableTextError, UnsupportedDocumentTypeError
from app.ports.chunking import TextChunker
from app.ports.documents import DocumentParser, WebDocumentFetcher
from app.ports.embeddings import EmbeddingModel
from app.ports.retrieval import VectorStore


class IngestUrlService:
    def __init__(
        self,
        *,
        fetcher: WebDocumentFetcher,
        parsers: Sequence[DocumentParser],
        chunker: TextChunker,
        embeddings: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self._fetcher = fetcher
        self._parsers = parsers
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def ingest(self, url: str) -> IngestionResult:
        fetched = await self._fetcher.fetch(url)
        parser = next(
            (
                candidate
                for candidate in self._parsers
                if candidate.supports(filename=None, content_type=fetched.content_type)
            ),
            None,
        )
        if parser is None:
            raise UnsupportedDocumentTypeError
        source_type = (
            SourceType.HTML
            if fetched.content_type in {"text/html", "application/xhtml+xml"}
            else SourceType.MARKDOWN
            if "markdown" in fetched.content_type
            else SourceType.TEXT
        )
        checksum = content_checksum(fetched.content)
        host = urlsplit(fetched.final_url).hostname or "web-document"
        source = DocumentSource(
            source_type=source_type,
            source=fetched.final_url,
            title=host,
            content_type=fetched.content_type,
        )
        descriptor = DocumentDescriptor(
            document_id=document_identity(source),
            document_version=document_version(checksum),
            document_checksum=checksum,
            source=source,
            ingested_at=datetime.now(UTC),
        )
        parsed = parser.parse(content=fetched.content, descriptor=descriptor)
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
        fingerprint = self._embeddings.status().fingerprint
        if fingerprint is None:
            raise RuntimeError("Embedding fingerprint is unavailable after embedding")
        await self._vector_store.replace_document(
            chunks=chunks,
            vectors=vectors,
            embedding_fingerprint=fingerprint,
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
