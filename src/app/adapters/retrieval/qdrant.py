from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from math import ceil
from typing import Protocol, TypeVar, cast

import anyio
import structlog
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode as LangChainRetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import DistanceMetric, QdrantMode, QdrantSettings, RetrievalMode
from app.domain.documents import DocumentChunk, SourceType
from app.domain.errors import (
    CollectionCompatibilityError,
    ConfigurationError,
    VectorStoreError,
)
from app.domain.retrieval import RetrievalFilter, RetrievedChunk

T = TypeVar("T")
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


class _SparseEmbedding(Protocol):
    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]: ...

    def embed_query(self, text: str) -> models.SparseVector: ...


def _distance(metric: DistanceMetric) -> models.Distance:
    return {
        DistanceMetric.COSINE: models.Distance.COSINE,
        DistanceMetric.DOT: models.Distance.DOT,
        DistanceMetric.EUCLID: models.Distance.EUCLID,
    }[metric]


class QdrantVectorStoreAdapter:
    """Qdrant adapter for dense and optional FastEmbed sparse/hybrid retrieval."""

    def __init__(
        self,
        settings: QdrantSettings,
        *,
        client: QdrantClient | None = None,
        sparse_embedding: _SparseEmbedding | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or self._build_client(settings)
        self._sparse_embedding = sparse_embedding
        if settings.retrieval_mode is not RetrievalMode.DENSE and sparse_embedding is None:
            self._sparse_embedding = self._build_sparse_embedding(settings)
        self._store: QdrantVectorStore | None = None
        self._fingerprint: str | None = None
        self._logger = structlog.get_logger(__name__)

    @staticmethod
    def _build_sparse_embedding(
        settings: QdrantSettings,
    ) -> _SparseEmbedding:  # pragma: no cover - optional hybrid extra
        try:
            from langchain_qdrant import FastEmbedSparse
        except ImportError as exc:
            msg = "Hybrid retrieval requires `uv sync --extra hybrid`"
            raise ConfigurationError(msg) from exc
        return cast(
            _SparseEmbedding,
            FastEmbedSparse(
                model_name=settings.sparse_model_id,
                cache_dir=str(settings.sparse_cache_dir),
            ),
        )

    @staticmethod
    def _build_client(settings: QdrantSettings) -> QdrantClient:
        if settings.mode is QdrantMode.MEMORY:
            return QdrantClient(location=":memory:")
        if settings.mode is QdrantMode.LOCAL:
            settings.path.mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(settings.path))
        return QdrantClient(
            url=str(settings.url),
            prefer_grpc=settings.prefer_grpc,
            timeout=ceil(settings.request_timeout_seconds),
        )

    async def _run(self, call: Callable[[], T]) -> T:
        try:
            if self._settings.mode is QdrantMode.SERVER:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=0.25, max=2),
                    retry=retry_if_exception_type((ResponseHandlingException, UnexpectedResponse)),
                    reraise=True,
                ):
                    with attempt:
                        return await anyio.to_thread.run_sync(call)
            return await anyio.to_thread.run_sync(call)
        except CollectionCompatibilityError:
            raise
        except Exception as exc:
            raise VectorStoreError("Qdrant operation failed") from exc
        raise VectorStoreError("Qdrant retry loop completed without a result")

    def _vector_configs(
        self,
        dimension: int,
    ) -> tuple[dict[str, models.VectorParams], dict[str, models.SparseVectorParams]]:
        dense: dict[str, models.VectorParams] = {}
        sparse: dict[str, models.SparseVectorParams] = {}
        if self._settings.retrieval_mode is not RetrievalMode.SPARSE:
            dense[DENSE_VECTOR] = models.VectorParams(
                size=dimension,
                distance=_distance(self._settings.distance),
            )
        if self._settings.retrieval_mode is not RetrievalMode.DENSE:
            sparse[SPARSE_VECTOR] = models.SparseVectorParams(modifier=models.Modifier.IDF)
        return dense, sparse

    async def initialize(self, *, dimension: int, embedding_fingerprint: str) -> None:
        self._fingerprint = embedding_fingerprint
        dense, sparse = self._vector_configs(dimension)
        exists = await self._run(partial(self._client.collection_exists, self._settings.collection))
        if not exists:
            await self._run(
                partial(
                    self._client.create_collection,
                    collection_name=self._settings.collection,
                    vectors_config=dense,
                    sparse_vectors_config=sparse or None,
                    metadata={
                        "embedding_fingerprint": embedding_fingerprint,
                        "vector_dimension": dimension,
                        "distance": self._settings.distance.value,
                        "vector_names": [*dense, *sparse],
                        "retrieval_mode": self._settings.retrieval_mode.value,
                        "sparse_model_id": self._settings.sparse_model_id,
                    },
                )
            )
            self._logger.info(
                "collection_initialized",
                retrieval_mode=self._settings.retrieval_mode.value,
            )
        await self._check_compatibility(
            dimension=dimension,
            embedding_fingerprint=embedding_fingerprint,
        )
        if self._settings.retrieval_mode is RetrievalMode.DENSE:
            self._store = QdrantVectorStore(
                client=self._client,
                collection_name=self._settings.collection,
                embedding=None,
                retrieval_mode=LangChainRetrievalMode.DENSE,
                vector_name=DENSE_VECTOR,
                distance=_distance(self._settings.distance),
                validate_embeddings=False,
                validate_collection_config=False,
            )

    async def _check_compatibility(
        self,
        *,
        dimension: int,
        embedding_fingerprint: str,
    ) -> None:
        info = await self._run(partial(self._client.get_collection, self._settings.collection))
        actual_dense = info.config.params.vectors
        actual_sparse = info.config.params.sparse_vectors or {}
        expected_dense, expected_sparse = self._vector_configs(dimension)
        metadata = info.config.metadata or {}
        if isinstance(actual_dense, dict):
            compatible = set(actual_dense) == set(expected_dense)
            if compatible and DENSE_VECTOR in expected_dense:
                params = actual_dense[DENSE_VECTOR]
                compatible = params.size == dimension and params.distance == _distance(
                    self._settings.distance
                )
        else:
            compatible = False
        compatible = compatible and set(actual_sparse) == set(expected_sparse)
        compatible = compatible and metadata.get("embedding_fingerprint") == embedding_fingerprint
        compatible = (
            compatible and metadata.get("retrieval_mode") == self._settings.retrieval_mode.value
        )
        if not compatible:
            self._logger.error("collection_incompatible")
            raise CollectionCompatibilityError

    async def health_check(self) -> bool:
        try:
            info = await self._run(partial(self._client.get_collection, self._settings.collection))
        except VectorStoreError:
            return False
        return info.status in {models.CollectionStatus.GREEN, models.CollectionStatus.YELLOW}

    @staticmethod
    def _document_filter(document_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key="kind", match=models.MatchValue(value="chunk")),
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchValue(value=document_id),
                ),
            ]
        )

    async def document_checksum(self, document_id: str) -> str | None:
        records, _ = await self._run(
            partial(
                self._client.scroll,
                collection_name=self._settings.collection,
                scroll_filter=self._document_filter(document_id),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        )
        if not records or not records[0].payload:
            return None
        metadata = records[0].payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        checksum = metadata.get("document_checksum")
        return str(checksum) if checksum else None

    async def replace_document(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        embedding_fingerprint: str,
    ) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise VectorStoreError("Chunks and vectors must be non-empty and aligned")
        if embedding_fingerprint != self._fingerprint:
            raise CollectionCompatibilityError
        document_id = chunks[0].document_id
        old_ids = await self._point_ids_for_document(document_id)
        sparse_vectors = await self._embed_sparse_documents(chunks)
        point_vectors: list[dict[str, list[float] | models.SparseVector]] = []
        for index, vector in enumerate(vectors):
            named: dict[str, list[float] | models.SparseVector] = {}
            if self._settings.retrieval_mode is not RetrievalMode.SPARSE:
                named[DENSE_VECTOR] = list(vector)
            if sparse_vectors is not None:
                named[SPARSE_VECTOR] = sparse_vectors[index]
            point_vectors.append(named)
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=point_vector,
                payload={
                    "kind": "chunk",
                    "page_content": chunk.text,
                    "metadata": self._chunk_metadata(chunk, embedding_fingerprint),
                },
            )
            for chunk, point_vector in zip(chunks, point_vectors, strict=True)
        ]
        await self._run(
            partial(
                self._client.upsert,
                collection_name=self._settings.collection,
                points=points,
                wait=True,
            )
        )
        current_ids = {chunk.chunk_id for chunk in chunks}
        stale_ids = [point_id for point_id in old_ids if point_id not in current_ids]
        if stale_ids:
            await self._run(
                partial(
                    self._client.delete,
                    collection_name=self._settings.collection,
                    points_selector=models.PointIdsList(points=stale_ids),
                    wait=True,
                )
            )

    async def _embed_sparse_documents(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[models.SparseVector] | None:
        if self._settings.retrieval_mode is RetrievalMode.DENSE:
            return None
        sparse = self._require_sparse_embedding()
        return await anyio.to_thread.run_sync(
            sparse.embed_documents,
            [chunk.text for chunk in chunks],
        )

    async def _point_ids_for_document(self, document_id: str) -> list[str]:
        records, _ = await self._run(
            partial(
                self._client.scroll,
                collection_name=self._settings.collection,
                scroll_filter=self._document_filter(document_id),
                limit=10_000,
                with_payload=False,
                with_vectors=False,
            )
        )
        return [str(record.id) for record in records]

    @staticmethod
    def _chunk_metadata(chunk: DocumentChunk, fingerprint: str) -> dict[str, object]:
        return {
            "document_id": chunk.document_id,
            "document_version": chunk.document_version,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "chunk_checksum": chunk.chunk_checksum,
            "document_checksum": chunk.document_checksum,
            "source_type": chunk.source_type.value,
            "source": chunk.source,
            "title": chunk.title,
            "page_number": chunk.page_number,
            "content_type": chunk.content_type,
            "ingested_at": chunk.ingested_at.isoformat(),
            "embedding_fingerprint": fingerprint,
        }

    async def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None,
        filters: RetrievalFilter,
    ) -> tuple[RetrievedChunk, ...]:
        query_filter = self._search_filter(filters)
        if self._settings.retrieval_mode is not RetrievalMode.DENSE:
            return await self._search_sparse_or_hybrid(
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
        store = self._store
        if store is None:
            raise VectorStoreError("The Qdrant collection is not initialized")
        pairs = await self._run(
            partial(
                store.similarity_search_with_score_by_vector,
                list(query_vector),
                k=top_k,
                filter=query_filter,
                score_threshold=score_threshold,
            )
        )
        return tuple(
            RetrievedChunk(
                chunk=self._metadata_to_chunk(document.metadata, document.page_content),
                score=score,
            )
            for document, score in pairs
        )

    async def _search_sparse_or_hybrid(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None,
        query_filter: models.Filter,
    ) -> tuple[RetrievedChunk, ...]:  # pragma: no cover - optional hybrid extra
        sparse_query = await anyio.to_thread.run_sync(
            self._require_sparse_embedding().embed_query,
            query_text,
        )
        if self._settings.retrieval_mode is RetrievalMode.SPARSE:
            response = await self._run(
                partial(
                    self._client.query_points,
                    collection_name=self._settings.collection,
                    query=sparse_query,
                    using=SPARSE_VECTOR,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            )
        else:
            response = await self._run(
                partial(
                    self._client.query_points,
                    collection_name=self._settings.collection,
                    prefetch=[
                        models.Prefetch(
                            query=list(query_vector),
                            using=DENSE_VECTOR,
                            limit=top_k,
                            filter=query_filter,
                        ),
                        models.Prefetch(
                            query=sparse_query,
                            using=SPARSE_VECTOR,
                            limit=top_k,
                            filter=query_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            )
        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            metadata = payload.get("metadata")
            text = payload.get("page_content")
            if isinstance(metadata, dict) and isinstance(text, str):
                results.append(
                    RetrievedChunk(
                        chunk=self._metadata_to_chunk(metadata, text),
                        score=point.score,
                    )
                )
        return tuple(results)

    def _require_sparse_embedding(
        self,
    ) -> _SparseEmbedding:  # pragma: no cover - optional hybrid extra
        if self._sparse_embedding is None:
            raise ConfigurationError("Sparse embeddings are not configured")
        return self._sparse_embedding

    @staticmethod
    def _search_filter(filters: RetrievalFilter) -> models.Filter:
        must: list[models.Condition] = [
            models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))
        ]
        if filters.document_ids:
            must.append(
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchAny(any=list(filters.document_ids)),
                )
            )
        if filters.source_types:
            must.append(
                models.FieldCondition(
                    key="metadata.source_type",
                    match=models.MatchAny(any=[value.value for value in filters.source_types]),
                )
            )
        return models.Filter(must=must)

    @staticmethod
    def _metadata_to_chunk(metadata: dict[str, object], text: str) -> DocumentChunk:
        return DocumentChunk(
            document_id=str(metadata["document_id"]),
            document_version=str(metadata["document_version"]),
            chunk_id=str(metadata["chunk_id"]),
            chunk_index=int(cast(int, metadata["chunk_index"])),
            text=text,
            chunk_checksum=str(metadata["chunk_checksum"]),
            document_checksum=str(metadata["document_checksum"]),
            source_type=SourceType(str(metadata["source_type"])),
            source=str(metadata["source"]),
            title=str(metadata["title"]),
            page_number=(
                int(cast(int, metadata["page_number"]))
                if metadata.get("page_number") is not None
                else None
            ),
            content_type=str(metadata["content_type"]),
            ingested_at=datetime.fromisoformat(str(metadata["ingested_at"])),
        )

    async def delete_document(self, document_id: str) -> bool:
        exists = await self.document_checksum(document_id)
        if exists is None:
            return False
        await self._run(
            partial(
                self._client.delete,
                collection_name=self._settings.collection,
                points_selector=models.FilterSelector(filter=self._document_filter(document_id)),
                wait=True,
            )
        )
        return True

    async def count(self) -> int:
        result = await self._run(
            partial(
                self._client.count,
                collection_name=self._settings.collection,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="kind",
                            match=models.MatchValue(value="chunk"),
                        )
                    ]
                ),
                exact=True,
            )
        )
        return result.count

    async def aclose(self) -> None:
        await anyio.to_thread.run_sync(self._client.close)
