from typing import Self

from pydantic import AnyHttpUrl, BaseModel, ConfigDict

from app.domain.documents import IngestionResult, IngestionStatus


class UrlIngestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: AnyHttpUrl


class IngestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_version: str
    status: IngestionStatus
    chunk_count: int

    @classmethod
    def from_domain(cls, result: IngestionResult) -> Self:
        return cls(
            document_id=result.document_id,
            document_version=result.document_version,
            status=result.status,
            chunk_count=result.chunk_count,
        )


class DeleteDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    deleted: bool = True
