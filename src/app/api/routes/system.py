from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.dependencies import get_container
from app.api.schemas import HealthResponse, ModelStatusResponse
from app.bootstrap.container import ApplicationContainer

health_router = APIRouter(tags=["system"])
api_router = APIRouter(prefix="/system", tags=["system"])


@health_router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@health_router.get("/health/ready", response_model=HealthResponse)
async def readiness(
    container: Annotated[ApplicationContainer, Depends(get_container)],
    response: Response,
) -> HealthResponse:
    ready, components = await container.readiness()
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="not_ready", components=components)
    model_state = components["model"]
    ready_status = "ready_with_lazy_model" if model_state == "unloaded" else "ready"
    return HealthResponse(status=ready_status, components=components)


@health_router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@api_router.get("/model", response_model=ModelStatusResponse)
async def model_status(
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> ModelStatusResponse:
    return ModelStatusResponse(**container.safe_status())
