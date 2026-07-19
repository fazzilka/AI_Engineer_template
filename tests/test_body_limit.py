from collections.abc import Awaitable, Callable
from typing import cast

import pytest
from starlette.types import Message, Receive, Scope, Send

from app.api.body_limit import RequestBodyLimitMiddleware


@pytest.mark.unit
@pytest.mark.asyncio
async def test_body_limit_passes_non_http_scope() -> None:
    called = False

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal called
        called = True

    middleware = RequestBodyLimitMiddleware(app, max_body_bytes=10)
    await middleware(cast(Scope, {"type": "websocket"}), _empty_receive, _collect([]))

    assert called


@pytest.mark.unit
@pytest.mark.asyncio
async def test_body_limit_rejects_streamed_and_invalid_declared_lengths() -> None:
    async def consuming_app(
        _scope: Scope,
        receive: Receive,
        _send: Send,
    ) -> None:
        await receive()

    middleware = RequestBodyLimitMiddleware(consuming_app, max_body_bytes=3)
    streamed: list[Message] = []

    async def oversized_receive() -> Message:
        return {"type": "http.request", "body": b"four", "more_body": False}

    await middleware(_http_scope([]), oversized_receive, _collect(streamed))
    assert streamed[0]["status"] == 413

    invalid: list[Message] = []
    await middleware(
        _http_scope([(b"content-length", b"invalid")]),
        _empty_receive,
        _collect(invalid),
    )
    assert invalid[0]["status"] == 413


def _http_scope(headers: list[tuple[bytes, bytes]]) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "state": {"request_id": "request-id"},
    }


async def _empty_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


def _collect(messages: list[Message]) -> Callable[[Message], Awaitable[None]]:
    async def send(message: Message) -> None:
        messages.append(message)

    return send
