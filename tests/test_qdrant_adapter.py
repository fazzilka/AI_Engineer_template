from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient, models

from app.adapters.embeddings.fake import FakeEmbeddingModel
from app.adapters.retrieval.qdrant import QdrantVectorStoreAdapter
from app.config import QdrantMode, QdrantSettings, RetrievalMode
from app.domain.document_rules import content_checksum
from app.domain.documents import DocumentChunk, SourceType
from app.domain.errors import CollectionCompatibilityError, VectorStoreError
from app.domain.retrieval import RetrievalFilter


def chunk(*, version: str = "v1", text: str = "local qdrant vectors") -> DocumentChunk:
    checksum = content_checksum(text)
    return DocumentChunk(
        document_id="11111111-1111-5111-8111-111111111111",
        document_version=version,
        chunk_id=(
            "22222222-2222-5222-8222-222222222222"
            if version == "v1"
            else "33333333-3333-5333-8333-333333333333"
        ),
        chunk_index=0,
        text=text,
        chunk_checksum=checksum,
        document_checksum=content_checksum(version),
        source_type=SourceType.TEXT,
        source="notes.txt",
        title="notes",
        page_number=None,
        content_type="text/plain",
        ingested_at=datetime.now(UTC),
    )


class SparseEmbedding:
    @staticmethod
    def _embed(text: str) -> models.SparseVector:
        indices = sorted({sum(token.encode()) % 10_000 for token in text.casefold().split()})
        return models.SparseVector(indices=indices, values=[1.0] * len(indices))

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._embed(text)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_qdrant_contract_upsert_search_replace_delete() -> None:
    embeddings = FakeEmbeddingModel(dimension=16)
    await embeddings.initialize()
    status = embeddings.status()
    adapter = QdrantVectorStoreAdapter(QdrantSettings(mode=QdrantMode.MEMORY))
    assert status.dimension and status.fingerprint
    await adapter.initialize(
        dimension=status.dimension,
        embedding_fingerprint=status.fingerprint,
    )
    first = chunk()
    await adapter.replace_document(
        chunks=(first,),
        vectors=await embeddings.embed_documents((first.text,)),
        embedding_fingerprint=status.fingerprint,
    )

    assert await adapter.health_check()
    assert await adapter.document_checksum(first.document_id) == first.document_checksum
    assert await adapter.count() == 1
    results = await adapter.search(
        query_text="qdrant vectors",
        query_vector=await embeddings.embed_query("qdrant vectors"),
        top_k=5,
        score_threshold=None,
        filters=RetrievalFilter(document_ids=(first.document_id,)),
    )
    assert results[0].chunk.chunk_id == first.chunk_id

    replacement = chunk(version="v2", text="updated qdrant local vectors")
    await adapter.replace_document(
        chunks=(replacement,),
        vectors=await embeddings.embed_documents((replacement.text,)),
        embedding_fingerprint=status.fingerprint,
    )
    assert await adapter.count() == 1
    assert await adapter.document_checksum(first.document_id) == replacement.document_checksum
    assert await adapter.delete_document(first.document_id)
    assert not await adapter.delete_document(first.document_id)
    assert await adapter.count() == 0
    await adapter.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_qdrant_rejects_incompatible_collection_and_vectors() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "documents",
        vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.DOT)},
        metadata={"embedding_fingerprint": "different"},
    )
    adapter = QdrantVectorStoreAdapter(
        QdrantSettings(mode=QdrantMode.MEMORY),
        client=client,
    )

    with pytest.raises(CollectionCompatibilityError):
        await adapter.initialize(dimension=4, embedding_fingerprint="expected")

    compatible = QdrantVectorStoreAdapter(QdrantSettings(mode=QdrantMode.MEMORY))
    await compatible.initialize(dimension=4, embedding_fingerprint="expected")
    with pytest.raises(VectorStoreError, match="aligned"):
        await compatible.replace_document(
            chunks=(),
            vectors=(),
            embedding_fingerprint="expected",
        )
    with pytest.raises(CollectionCompatibilityError):
        item = chunk()
        await compatible.replace_document(
            chunks=(item,),
            vectors=([0.0] * 4,),
            embedding_fingerprint="wrong",
        )
    await compatible.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_qdrant_hybrid_mode_combines_dense_and_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.find_spec", lambda _name: object())
    settings = QdrantSettings(
        mode=QdrantMode.MEMORY,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    adapter = QdrantVectorStoreAdapter(
        settings,
        sparse_embedding=SparseEmbedding(),
    )
    embeddings = FakeEmbeddingModel(dimension=16)
    await embeddings.initialize()
    status = embeddings.status()
    assert status.dimension and status.fingerprint
    await adapter.initialize(
        dimension=status.dimension,
        embedding_fingerprint=status.fingerprint,
    )
    item = chunk()
    await adapter.replace_document(
        chunks=(item,),
        vectors=await embeddings.embed_documents((item.text,)),
        embedding_fingerprint=status.fingerprint,
    )

    results = await adapter.search(
        query_text="qdrant vectors",
        query_vector=await embeddings.embed_query("qdrant vectors"),
        top_k=3,
        score_threshold=None,
        filters=RetrievalFilter(),
    )

    assert results[0].chunk.document_id == item.document_id
    await adapter.aclose()
