from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.dependencies import get_container
from app.api.schemas import DeleteDocumentResponse, IngestionResponse, UrlIngestionRequest
from app.bootstrap.container import ApplicationContainer
from app.domain.errors import DocumentTooLargeError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=IngestionResponse, status_code=status.HTTP_200_OK)
async def upload_document(
    container: Annotated[ApplicationContainer, Depends(get_container)],
    file: Annotated[UploadFile, File()],
) -> IngestionResponse:
    limit = container.settings.ingestion.max_file_bytes
    content = await file.read(limit + 1)
    await file.close()
    if len(content) > limit:
        raise DocumentTooLargeError
    result = await container.ingest_upload.ingest(
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        content=content,
    )
    return IngestionResponse.from_domain(result)


@router.post("/url", response_model=IngestionResponse, status_code=status.HTTP_200_OK)
async def ingest_url(
    payload: UrlIngestionRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> IngestionResponse:
    result = await container.ingest_url.ingest(str(payload.url))
    return IngestionResponse.from_domain(result)


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_document(
    document_id: str,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> DeleteDocumentResponse:
    await container.delete_document.delete(document_id)
    return DeleteDocumentResponse(document_id=document_id)
