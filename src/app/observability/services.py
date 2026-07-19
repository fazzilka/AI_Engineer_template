from time import perf_counter

import structlog

from app.application.delete_document import DeleteDocumentService
from app.application.ingest_document import IngestUploadedDocumentService
from app.application.ingest_url import IngestUrlService
from app.domain.documents import IngestionResult
from app.observability.metrics import (
    DOCUMENTS_TOTAL,
    INGESTED_CHUNKS,
    INGESTION_DURATION,
    INGESTION_REQUESTS,
)
from app.ports.retrieval import VectorStore


class InstrumentedUploadIngestion:
    def __init__(
        self,
        service: IngestUploadedDocumentService,
        *,
        vector_store: VectorStore,
    ) -> None:
        self._service = service
        self._vector_store = vector_store
        self._logger = structlog.get_logger(__name__)

    async def ingest(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> IngestionResult:
        source_type = _upload_source_type(filename, content_type)
        started = perf_counter()
        self._logger.info("ingestion_started", source_type=source_type)
        try:
            result = await self._service.ingest(
                filename=filename,
                content_type=content_type,
                content=content,
            )
        except Exception:
            INGESTION_REQUESTS.labels(source_type, "error").inc()
            self._logger.exception("ingestion_failed", source_type=source_type)
            raise
        INGESTION_REQUESTS.labels(source_type, result.status.value).inc()
        INGESTION_DURATION.labels(source_type).observe(perf_counter() - started)
        INGESTED_CHUNKS.labels(source_type).inc(result.chunk_count)
        DOCUMENTS_TOTAL.set(await self._vector_store.count())
        self._logger.info(
            "ingestion_completed",
            source_type=source_type,
            document_id=result.document_id,
            chunk_count=result.chunk_count,
            outcome=result.status.value,
        )
        return result


class InstrumentedUrlIngestion:
    def __init__(self, service: IngestUrlService, *, vector_store: VectorStore) -> None:
        self._service = service
        self._vector_store = vector_store
        self._logger = structlog.get_logger(__name__)

    async def ingest(self, url: str) -> IngestionResult:
        started = perf_counter()
        self._logger.info("ingestion_started", source_type="url")
        try:
            result = await self._service.ingest(url)
        except Exception:
            INGESTION_REQUESTS.labels("url", "error").inc()
            self._logger.exception("ingestion_failed", source_type="url")
            raise
        INGESTION_REQUESTS.labels("url", result.status.value).inc()
        INGESTION_DURATION.labels("url").observe(perf_counter() - started)
        INGESTED_CHUNKS.labels("url").inc(result.chunk_count)
        DOCUMENTS_TOTAL.set(await self._vector_store.count())
        self._logger.info(
            "ingestion_completed",
            source_type="url",
            document_id=result.document_id,
            chunk_count=result.chunk_count,
            outcome=result.status.value,
        )
        return result


class InstrumentedDeleteDocument:
    def __init__(self, service: DeleteDocumentService, *, vector_store: VectorStore) -> None:
        self._service = service
        self._vector_store = vector_store
        self._logger = structlog.get_logger(__name__)

    async def delete(self, document_id: str) -> None:
        await self._service.delete(document_id)
        DOCUMENTS_TOTAL.set(await self._vector_store.count())
        self._logger.info("document_deleted", document_id=document_id)


def _upload_source_type(filename: str, content_type: str) -> str:
    lowered = filename.lower()
    if content_type.startswith("application/pdf") or lowered.endswith(".pdf"):
        return "pdf"
    if "markdown" in content_type or lowered.endswith(".md"):
        return "markdown"
    return "text"
