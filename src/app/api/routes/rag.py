from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_container
from app.api.schemas import RagRequest, RagResponse
from app.bootstrap.container import ApplicationContainer

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RagResponse, status_code=status.HTTP_200_OK)
async def query_rag(
    payload: RagRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> RagResponse:
    result = await container.rag.answer(
        query=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        filters=payload.filters.to_domain(),
    )
    return RagResponse.from_domain(result)
