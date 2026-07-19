from collections.abc import Sequence
from time import perf_counter

from prometheus_client import Counter, Gauge, Histogram

from app.domain.chat import ChatMessage, GenerationResult
from app.domain.documents import DocumentChunk
from app.domain.generation import EmbeddingStatus, ModelStatus
from app.domain.retrieval import RetrievalFilter, RetrievedChunk
from app.ports.embeddings import EmbeddingModel
from app.ports.llm import ManagedChatModel
from app.ports.retrieval import VectorStore

HTTP_REQUESTS = Counter(
    "app_http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "app_http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "route"),
)
MODEL_LOADED = Gauge(
    "ai_model_loaded",
    "Whether the configured model is loaded",
    labelnames=("backend", "model_alias"),
)
MODEL_LOAD_DURATION = Histogram(
    "ai_model_load_duration_seconds",
    "Local model load duration",
    labelnames=("backend", "model_alias", "outcome"),
)
GENERATION_REQUESTS = Counter(
    "ai_generation_requests_total",
    "Local generation requests",
    labelnames=("backend", "model_alias", "outcome"),
)
GENERATION_DURATION = Histogram(
    "ai_generation_duration_seconds",
    "Local generation duration",
    labelnames=("backend", "model_alias"),
)
GENERATION_TOKENS = Counter(
    "ai_generation_tokens_total",
    "Locally tokenized generation usage",
    labelnames=("backend", "model_alias", "direction"),
)
EMBEDDING_REQUESTS = Counter(
    "ai_embedding_requests_total",
    "Embedding requests",
    labelnames=("backend", "model_alias", "operation", "outcome"),
)
EMBEDDING_DURATION = Histogram(
    "ai_embedding_duration_seconds",
    "Embedding duration",
    labelnames=("backend", "model_alias", "operation"),
)
EMBEDDING_TEXTS = Counter(
    "ai_embedding_texts_total",
    "Texts embedded",
    labelnames=("backend", "model_alias", "operation"),
)
RETRIEVAL_REQUESTS = Counter(
    "ai_retrieval_requests_total",
    "Retrieval requests",
    labelnames=("retrieval_mode", "outcome"),
)
RETRIEVAL_DURATION = Histogram(
    "ai_retrieval_duration_seconds",
    "Retrieval duration",
    labelnames=("retrieval_mode",),
)
RETRIEVED_CHUNKS = Counter(
    "ai_retrieved_chunks_total",
    "Chunks returned by retrieval",
    labelnames=("retrieval_mode",),
)
INGESTION_REQUESTS = Counter(
    "ai_ingestion_requests_total",
    "Document ingestion requests",
    labelnames=("source_type", "outcome"),
)
INGESTION_DURATION = Histogram(
    "ai_ingestion_duration_seconds",
    "Document ingestion duration",
    labelnames=("source_type",),
)
INGESTED_CHUNKS = Counter(
    "ai_ingested_chunks_total",
    "Chunks produced by ingestion",
    labelnames=("source_type",),
)
DOCUMENTS_TOTAL = Gauge("ai_documents_total", "Indexed local documents")


class InstrumentedChatModel:
    def __init__(self, client: ManagedChatModel) -> None:
        self._client = client

    def _labels(self) -> tuple[str, str]:
        status = self._client.status()
        return status.backend, status.model_alias

    async def load(self) -> None:
        backend, alias = self._labels()
        started = perf_counter()
        try:
            await self._client.load()
        except Exception:
            MODEL_LOAD_DURATION.labels(backend, alias, "error").observe(perf_counter() - started)
            MODEL_LOADED.labels(backend, alias).set(0)
            raise
        MODEL_LOAD_DURATION.labels(backend, alias, "success").observe(perf_counter() - started)
        MODEL_LOADED.labels(backend, alias).set(1)

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        backend, alias = self._labels()
        started = perf_counter()
        try:
            result = await self._client.generate(messages=messages, system_prompt=system_prompt)
        except Exception:
            GENERATION_REQUESTS.labels(backend, alias, "error").inc()
            raise
        GENERATION_REQUESTS.labels(backend, alias, "success").inc()
        if result.usage.input_tokens is not None:
            GENERATION_TOKENS.labels(backend, alias, "input").inc(result.usage.input_tokens)
        if result.usage.output_tokens is not None:
            GENERATION_TOKENS.labels(backend, alias, "output").inc(result.usage.output_tokens)
        GENERATION_DURATION.labels(backend, alias).observe(perf_counter() - started)
        return result

    async def count_tokens(self, text: str) -> int | None:
        return await self._client.count_tokens(text)

    async def warmup(self) -> None:
        await self._client.warmup()

    def status(self) -> ModelStatus:
        return self._client.status()

    async def aclose(self) -> None:
        await self._client.aclose()
        backend, alias = self._labels()
        MODEL_LOADED.labels(backend, alias).set(0)


InstrumentedLLMClient = InstrumentedChatModel


class InstrumentedEmbeddingModel:
    def __init__(self, client: EmbeddingModel) -> None:
        self._client = client

    async def initialize(self) -> None:
        await self._client.initialize()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed("documents", texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed("query", (text,)))[0]

    async def _embed(self, operation: str, texts: Sequence[str]) -> list[list[float]]:
        status = self._client.status()
        labels = (status.backend, status.model_alias, operation)
        started = perf_counter()
        try:
            if operation == "query":
                vectors = [await self._client.embed_query(texts[0])]
            else:
                vectors = await self._client.embed_documents(texts)
        except Exception:
            EMBEDDING_REQUESTS.labels(*labels, "error").inc()
            raise
        EMBEDDING_REQUESTS.labels(*labels, "success").inc()
        EMBEDDING_DURATION.labels(*labels).observe(perf_counter() - started)
        EMBEDDING_TEXTS.labels(*labels).inc(len(texts))
        return vectors

    def status(self) -> EmbeddingStatus:
        return self._client.status()

    async def aclose(self) -> None:
        await self._client.aclose()


class InstrumentedVectorStore:
    def __init__(self, client: VectorStore, *, retrieval_mode: str) -> None:
        self._client = client
        self._retrieval_mode = retrieval_mode

    async def initialize(self, *, dimension: int, embedding_fingerprint: str) -> None:
        await self._client.initialize(
            dimension=dimension,
            embedding_fingerprint=embedding_fingerprint,
        )

    async def health_check(self) -> bool:
        return await self._client.health_check()

    async def document_checksum(self, document_id: str) -> str | None:
        return await self._client.document_checksum(document_id)

    async def replace_document(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        embedding_fingerprint: str,
    ) -> None:
        await self._client.replace_document(
            chunks=chunks,
            vectors=vectors,
            embedding_fingerprint=embedding_fingerprint,
        )

    async def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None,
        filters: RetrievalFilter,
    ) -> tuple[RetrievedChunk, ...]:
        started = perf_counter()
        try:
            results = await self._client.search(
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
            )
        except Exception:
            RETRIEVAL_REQUESTS.labels(self._retrieval_mode, "error").inc()
            raise
        RETRIEVAL_REQUESTS.labels(self._retrieval_mode, "success").inc()
        RETRIEVAL_DURATION.labels(self._retrieval_mode).observe(perf_counter() - started)
        RETRIEVED_CHUNKS.labels(self._retrieval_mode).inc(len(results))
        return results

    async def delete_document(self, document_id: str) -> bool:
        return await self._client.delete_document(document_id)

    async def count(self) -> int:
        return await self._client.count()

    async def aclose(self) -> None:
        await self._client.aclose()
