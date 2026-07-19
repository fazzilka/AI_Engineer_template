from app.domain.errors import DocumentNotFoundError
from app.ports.retrieval import VectorStore


class DeleteDocumentService:
    def __init__(self, *, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    async def delete(self, document_id: str) -> None:
        if not await self._vector_store.delete_document(document_id):
            raise DocumentNotFoundError
