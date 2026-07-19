import asyncio
from collections.abc import Sequence
from typing import Protocol, cast

import anyio
import structlog

from app.adapters.embeddings.fingerprint import embedding_fingerprint
from app.adapters.model_runtime import select_device
from app.config import EmbeddingSettings, ModelSource
from app.domain.errors import EmbeddingError
from app.domain.generation import EmbeddingStatus, LifecycleState


class _EmbeddingClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HuggingFaceEmbeddingModel:
    """Local sentence-transformers adapter through LangChain HuggingFaceEmbeddings."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._client: _EmbeddingClient | None = None
        self._state = LifecycleState.UNLOADED
        self._device = settings.device.value
        self._dimension: int | None = None
        self._fingerprint: str | None = None
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(__name__)

    async def initialize(self) -> None:
        if self._state is LifecycleState.READY:
            return
        async with self._lock:
            if self._client is not None:
                self._state = LifecycleState.READY
                return
            self._state = LifecycleState.LOADING
            try:
                await anyio.to_thread.run_sync(self._load_sync)
            except Exception as exc:
                self._state = LifecycleState.FAILED
                self._logger.exception("embedding_load_failed", model_alias=self._settings.alias)
                raise EmbeddingError("Local embedding model loading failed") from exc
            self._state = LifecycleState.READY

    def _load_sync(self) -> None:  # pragma: no cover - exercised by opt-in local model smoke
        from langchain_huggingface import HuggingFaceEmbeddings

        reference = (
            str(self._settings.path.resolve())
            if self._settings.source is ModelSource.FILESYSTEM
            else self._settings.id
        )
        self._device = select_device(self._settings.device)
        model_kwargs: dict[str, object] = {
            "device": self._device,
            "local_files_only": self._settings.local_files_only,
            "trust_remote_code": False,
        }
        if self._settings.revision:
            model_kwargs["revision"] = self._settings.revision
        client = HuggingFaceEmbeddings(
            model_name=reference,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "normalize_embeddings": self._settings.normalize,
                "batch_size": self._settings.batch_size,
            },
        )
        probe = client.embed_query(f"{self._settings.query_prefix}dimension probe")
        self._dimension = len(probe)
        self._fingerprint = embedding_fingerprint(
            model_reference=(
                self._settings.path
                if self._settings.source is ModelSource.FILESYSTEM
                else self._settings.id
            ),
            revision=self._settings.revision,
            normalize=self._settings.normalize,
            dimension=self._dimension,
            query_prefix=self._settings.query_prefix,
            document_prefix=self._settings.document_prefix,
        )
        self._client = cast(_EmbeddingClient, client)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        await self.initialize()
        client = self._require_client()
        values = [f"{self._settings.document_prefix}{text}" for text in texts]
        self._logger.info("embedding_started", operation="documents", text_count=len(values))
        try:
            vectors = await anyio.to_thread.run_sync(client.embed_documents, values)
        except Exception as exc:
            self._logger.exception("embedding_failed", operation="documents")
            raise EmbeddingError("Document embedding failed") from exc
        self._logger.info("embedding_completed", operation="documents", text_count=len(values))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        await self.initialize()
        client = self._require_client()
        self._logger.info("embedding_started", operation="query", text_count=1)
        try:
            vector = await anyio.to_thread.run_sync(
                client.embed_query,
                f"{self._settings.query_prefix}{text}",
            )
        except Exception as exc:
            self._logger.exception("embedding_failed", operation="query")
            raise EmbeddingError("Query embedding failed") from exc
        self._logger.info("embedding_completed", operation="query", text_count=1)
        return vector

    def _require_client(self) -> _EmbeddingClient:
        if self._client is None:
            raise EmbeddingError("Embedding model is not initialized")
        return self._client

    def status(self) -> EmbeddingStatus:
        return EmbeddingStatus(
            backend="huggingface",
            model_alias=self._settings.alias,
            device=self._device,
            state=self._state,
            dimension=self._dimension,
            fingerprint=self._fingerprint,
        )

    async def aclose(self) -> None:
        self._client = None
        self._state = LifecycleState.CLOSED
