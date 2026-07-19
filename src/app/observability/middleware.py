import re
from time import perf_counter
from uuid import uuid4

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.metrics import HTTP_REQUEST_DURATION, HTTP_REQUESTS

REQUEST_ID_HEADER = "x-request-id"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestObservabilityMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._logger = structlog.get_logger(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method = scope["method"]
        request_id = self._request_id(scope)
        status_code = 500
        started_at = perf_counter()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        async def send_with_context(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self._app(scope, receive, send_with_context)
        except Exception:
            self._logger.exception("http_request_failed", method=method, path=scope["path"])
            raise
        finally:
            duration = perf_counter() - started_at
            route = getattr(scope.get("route"), "path", "unmatched")
            HTTP_REQUESTS.labels(method=method, route=route, status=str(status_code)).inc()
            HTTP_REQUEST_DURATION.labels(method=method, route=route).observe(duration)
            self._logger.info(
                "http_request_completed",
                method=method,
                route=route,
                status_code=status_code,
                duration_ms=round(duration * 1_000, 2),
            )
            structlog.contextvars.clear_contextvars()

    @staticmethod
    def _request_id(scope: Scope) -> str:
        candidate = Headers(scope=scope).get(REQUEST_ID_HEADER)
        if candidate and REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
        return uuid4().hex
