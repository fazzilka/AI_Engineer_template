import orjson
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self._app = app
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        content_length = Headers(scope=scope).get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = self._max_body_bytes + 1
            if declared_length > self._max_body_bytes:
                await self._reject(scope, send)
                return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_body_bytes:
                    msg = "request_body_too_large"
                    raise ValueError(msg)
            return message

        try:
            await self._app(scope, limited_receive, send)
        except ValueError as exc:
            if str(exc) != "request_body_too_large":
                raise
            await self._reject(scope, send)

    @staticmethod
    async def _reject(scope: Scope, send: Send) -> None:
        request_id = str(scope.get("state", {}).get("request_id", "unknown"))
        body = orjson.dumps(
            {
                "error": {
                    "code": "document_too_large",
                    "message": "The request body exceeds the configured size limit.",
                    "request_id": request_id,
                }
            }
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
