from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_container
from app.api.schemas import RetrievalRequest, RetrievalResponse
from app.api.schemas.retrieval import RetrievedChunkResponse
from app.bootstrap.container import ApplicationContainer

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalResponse, status_code=status.HTTP_200_OK)
async def search_documents(
    payload: RetrievalRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> RetrievalResponse:
    results = await container.retrieve.search(
        query=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        filters=payload.filters.to_domain(),
    )
    return RetrievalResponse(
        results=[RetrievedChunkResponse.from_domain(result) for result in results]
    )
