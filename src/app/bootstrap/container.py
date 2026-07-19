import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass

from app.adapters.chunking import LangChainTextChunker
from app.adapters.embeddings import build_embedding_model
from app.adapters.llm import build_chat_model
from app.adapters.parsers import HtmlDocumentParser, PdfDocumentParser, TextDocumentParser
from app.adapters.retrieval import QdrantVectorStoreAdapter
from app.adapters.web import HttpxWebDocumentFetcher
from app.application.chat import ChatService
from app.application.delete_document import DeleteDocumentService
from app.application.ingest_document import IngestUploadedDocumentService
from app.application.ingest_url import IngestUrlService
from app.application.rag import RagService
from app.application.retrieve import RetrieveService
from app.config import Settings
from app.domain.generation import LifecycleState
from app.observability.metrics import (
    DOCUMENTS_TOTAL,
    InstrumentedChatModel,
    InstrumentedEmbeddingModel,
    InstrumentedVectorStore,
)
from app.observability.services import (
    InstrumentedDeleteDocument,
    InstrumentedUploadIngestion,
    InstrumentedUrlIngestion,
)
from app.ports.documents import DocumentParser, WebDocumentFetcher
from app.ports.embeddings import EmbeddingModel
from app.ports.llm import ManagedChatModel
from app.ports.retrieval import VectorStore
from app.prompts import load_chat_system_prompt, load_rag_system_prompt


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    model: InstrumentedChatModel
    embeddings: InstrumentedEmbeddingModel
    vector_store: InstrumentedVectorStore
    web_fetcher: WebDocumentFetcher
    chat: ChatService
    ingest_upload: InstrumentedUploadIngestion
    ingest_url: InstrumentedUrlIngestion
    delete_document: InstrumentedDeleteDocument
    retrieve: RetrieveService
    rag: RagService
    _started: bool = False
    _start_lock: asyncio.Lock | None = None

    async def start(self) -> None:
        if self._started:
            return
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._started:
                return
            await self.embeddings.initialize()
            embedding_status = self.embeddings.status()
            if embedding_status.dimension is None or embedding_status.fingerprint is None:
                msg = "Embedding adapter did not expose dimension and fingerprint"
                raise RuntimeError(msg)
            await self.vector_store.initialize(
                dimension=embedding_status.dimension,
                embedding_fingerprint=embedding_status.fingerprint,
            )
            DOCUMENTS_TOTAL.set(await self.vector_store.count())
            if self.settings.model.load_on_startup:
                await self.model.load()
                await self.model.warmup()
            self._started = True

    async def readiness(self) -> tuple[bool, dict[str, str]]:
        model_state = self.model.status().state
        embedding_state = self.embeddings.status().state
        vector_ready = self._started and await self.vector_store.health_check()
        components = {
            "container": "ready" if self._started else "not_ready",
            "model": model_state.value,
            "embeddings": embedding_state.value,
            "vector_store": "ready" if vector_ready else "not_ready",
        }
        acceptable_model = model_state not in {LifecycleState.FAILED, LifecycleState.CLOSED}
        return (
            self._started
            and embedding_state is LifecycleState.READY
            and vector_ready
            and acceptable_model,
            components,
        )

    def safe_status(self) -> dict[str, str | bool | int | None]:
        model = self.model.status()
        embeddings = self.embeddings.status()
        return {
            "backend": model.backend,
            "model_alias": model.model_alias,
            "source": model.source,
            "device": model.device,
            "dtype": model.dtype,
            "loaded": model.loaded,
            "loading": model.loading,
            "local_files_only": model.local_files_only,
            "context_limit": model.max_input_tokens,
            "max_new_tokens": model.max_new_tokens,
            "embedding_backend": embeddings.backend,
            "embedding_model_alias": embeddings.model_alias,
            "embedding_dimension": embeddings.dimension,
            "qdrant_mode": self.settings.qdrant.mode.value,
            "retrieval_mode": self.settings.qdrant.retrieval_mode.value,
        }

    async def aclose(self) -> None:
        await self.web_fetcher.aclose()
        await self.vector_store.aclose()
        await self.embeddings.aclose()
        await self.model.aclose()
        self._started = False


def build_container(
    settings: Settings,
    *,
    model: ManagedChatModel | None = None,
    embeddings: EmbeddingModel | None = None,
    vector_store: VectorStore | None = None,
    web_fetcher: WebDocumentFetcher | None = None,
    parsers: Sequence[DocumentParser] | None = None,
) -> ApplicationContainer:
    if settings.offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
    raw_model = model or build_chat_model(settings.model)
    raw_embeddings = embeddings or build_embedding_model(settings.embeddings)
    raw_vector_store = vector_store or QdrantVectorStoreAdapter(settings.qdrant)
    instrumented_model = InstrumentedChatModel(raw_model)
    instrumented_embeddings = InstrumentedEmbeddingModel(raw_embeddings)
    instrumented_vector_store = InstrumentedVectorStore(
        raw_vector_store,
        retrieval_mode=settings.qdrant.retrieval_mode.value,
    )
    resolved_parsers: Sequence[DocumentParser] = parsers or (
        PdfDocumentParser(
            max_pages=settings.ingestion.max_pdf_pages,
            max_characters=settings.ingestion.max_extracted_characters,
        ),
        TextDocumentParser(max_characters=settings.ingestion.max_extracted_characters),
        HtmlDocumentParser(max_characters=settings.ingestion.max_extracted_characters),
    )
    chunker = LangChainTextChunker(settings.chunking)
    fetcher = web_fetcher or HttpxWebDocumentFetcher(settings.web)
    upload_service = IngestUploadedDocumentService(
        parsers=resolved_parsers,
        chunker=chunker,
        embeddings=instrumented_embeddings,
        vector_store=instrumented_vector_store,
        max_file_bytes=settings.ingestion.max_file_bytes,
    )
    url_service = IngestUrlService(
        fetcher=fetcher,
        parsers=resolved_parsers,
        chunker=chunker,
        embeddings=instrumented_embeddings,
        vector_store=instrumented_vector_store,
    )
    delete_service = DeleteDocumentService(vector_store=instrumented_vector_store)
    retrieve = RetrieveService(
        embeddings=instrumented_embeddings,
        vector_store=instrumented_vector_store,
        default_top_k=settings.qdrant.top_k,
        default_score_threshold=settings.qdrant.score_threshold,
    )
    return ApplicationContainer(
        settings=settings,
        model=instrumented_model,
        embeddings=instrumented_embeddings,
        vector_store=instrumented_vector_store,
        web_fetcher=fetcher,
        chat=ChatService(model=instrumented_model, system_prompt=load_chat_system_prompt()),
        ingest_upload=InstrumentedUploadIngestion(
            upload_service,
            vector_store=instrumented_vector_store,
        ),
        ingest_url=InstrumentedUrlIngestion(
            url_service,
            vector_store=instrumented_vector_store,
        ),
        delete_document=InstrumentedDeleteDocument(
            delete_service,
            vector_store=instrumented_vector_store,
        ),
        retrieve=retrieve,
        rag=RagService(
            retriever=retrieve,
            model=instrumented_model,
            system_prompt=load_rag_system_prompt(),
            top_k=settings.rag.top_k,
            max_context_chunks=settings.rag.max_context_chunks,
            max_context_characters=settings.rag.max_context_characters,
            max_context_tokens=settings.rag.max_context_tokens,
            min_relevant_chunks=settings.rag.min_relevant_chunks,
            snippet_characters=settings.rag.citation_snippet_characters,
            return_sources=settings.rag.return_sources,
            model_alias=settings.model.alias,
        ),
    )
