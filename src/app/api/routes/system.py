from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    return HealthResponse(status="ready")


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
