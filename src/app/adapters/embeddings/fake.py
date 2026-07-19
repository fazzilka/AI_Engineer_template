import hashlib
import math
import re
from collections.abc import Sequence

from app.adapters.embeddings.fingerprint import embedding_fingerprint
from app.domain.generation import EmbeddingStatus, LifecycleState

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class FakeEmbeddingModel:
    """Deterministic hashing embeddings with useful lexical relevance."""

    def __init__(self, *, dimension: int = 64, normalize: bool = True) -> None:
        self._dimension = dimension
        self._normalize = normalize
        self._state = LifecycleState.UNLOADED
        self._fingerprint = embedding_fingerprint(
            model_reference="built-in/fake-hashing-v1",
            revision="1",
            normalize=normalize,
            dimension=dimension,
            query_prefix="",
            document_prefix="",
        )

    async def initialize(self) -> None:
        self._state = LifecycleState.READY

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in TOKEN_PATTERN.findall(text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        if self._normalize and magnitude:
            return [value / magnitude for value in vector]
        return vector

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        await self.initialize()
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        await self.initialize()
        return self._embed(text)

    def status(self) -> EmbeddingStatus:
        return EmbeddingStatus(
            backend="fake",
            model_alias="fake-hashing-v1",
            device="cpu",
            state=self._state,
            dimension=self._dimension,
            fingerprint=self._fingerprint,
        )

    async def aclose(self) -> None:
        self._state = LifecycleState.CLOSED
