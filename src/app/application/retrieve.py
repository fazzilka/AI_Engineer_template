from app.domain.retrieval import RetrievalFilter, RetrievedChunk
from app.ports.embeddings import EmbeddingModel
from app.ports.retrieval import VectorStore


class RetrieveService:
    def __init__(
        self,
        *,
        embeddings: EmbeddingModel,
        vector_store: VectorStore,
        default_top_k: int,
        default_score_threshold: float | None,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._default_score_threshold = default_score_threshold

    async def search(
        self,
        *,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: RetrievalFilter | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        vector = await self._embeddings.embed_query(query)
        return await self._vector_store.search(
            query_text=query,
            query_vector=vector,
            top_k=top_k or self._default_top_k,
            score_threshold=(
                self._default_score_threshold if score_threshold is None else score_threshold
            ),
            filters=filters or RetrievalFilter(),
        )
