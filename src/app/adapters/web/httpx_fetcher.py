import ipaddress
import socket
from collections.abc import Awaitable, Callable
from functools import partial
from typing import cast
from urllib.parse import urljoin, urlsplit, urlunsplit

import anyio
import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import WebFetchSettings
from app.domain.documents import FetchedDocument
from app.domain.errors import (
    DocumentTooLargeError,
    RemoteDocumentFetchError,
    UnsafeUrlError,
)

Resolver = Callable[
    [str, int], Awaitable[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]]
]
ALLOWED_CONTENT_TYPES = {
    "application/xhtml+xml",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/x-markdown",
}
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
TRANSIENT_STATUSES = {429, 502, 503, 504}


class _TransientFetchError(Exception):
    pass


async def resolve_host(
    host: str,
    port: int,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        literal = ipaddress.ip_address(host.split("%", maxsplit=1)[0])
    except ValueError:
        try:
            records = await anyio.to_thread.run_sync(
                partial(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
            )
        except OSError as exc:
            raise UnsafeUrlError("The URL host could not be resolved") from exc
        addresses = {
            ipaddress.ip_address(str(record[4][0]).split("%", maxsplit=1)[0]) for record in records
        }
        if not addresses:
            raise UnsafeUrlError("The URL host did not resolve to an address") from None
        return tuple(sorted(addresses, key=str))
    return (literal,)


async def validate_url(
    url: str,
    *,
    allow_private_hosts: bool,
    resolver: Resolver = resolve_host,
) -> str:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise UnsafeUrlError("Only HTTP and HTTPS URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeUrlError("Embedded URL credentials are not allowed")
    if not parsed.hostname:
        raise UnsafeUrlError("The URL must include a host")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise UnsafeUrlError("The URL port is invalid") from exc
    addresses = await resolver(parsed.hostname, port)
    for address in addresses:
        if address.is_unspecified or address.is_multicast:
            raise UnsafeUrlError("The URL resolves to a prohibited address")
        if not allow_private_hosts and not address.is_global:
            raise UnsafeUrlError("The URL resolves to a non-public address")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


class HttpxWebDocumentFetcher:
    def __init__(
        self,
        settings: WebFetchSettings,
        *,
        client: httpx.AsyncClient | None = None,
        resolver: Resolver = resolve_host,
    ) -> None:
        self._settings = settings
        self._resolver = resolver
        self._owned_client = client is None
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(
                connect=settings.connect_timeout_seconds,
                read=settings.read_timeout_seconds,
                write=settings.read_timeout_seconds,
                pool=settings.connect_timeout_seconds,
            ),
            headers={"User-Agent": settings.user_agent},
            trust_env=False,
        )
        self._logger = structlog.get_logger(__name__)

    async def fetch(self, url: str) -> FetchedDocument:
        if not self._settings.enabled:
            raise UnsafeUrlError("URL ingestion is disabled")
        current_url = url
        for redirect_count in range(self._settings.max_redirects + 1):
            current_url = await validate_url(
                current_url,
                allow_private_hosts=self._settings.allow_private_hosts,
                resolver=self._resolver,
            )
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self._settings.max_retries + 1),
                    wait=wait_exponential(multiplier=0.25, max=2),
                    retry=retry_if_exception_type((_TransientFetchError, httpx.TransportError)),
                    reraise=True,
                ):
                    with attempt:
                        response = await self._request(current_url)
            except (httpx.TransportError, _TransientFetchError) as exc:
                raise RemoteDocumentFetchError(
                    "Remote request failed after bounded retries"
                ) from exc
            if isinstance(response, FetchedDocument):
                return response
            if redirect_count == self._settings.max_redirects:
                raise RemoteDocumentFetchError("The remote URL exceeded the redirect limit")
            current_url = urljoin(current_url, response)
        raise RemoteDocumentFetchError("The remote URL exceeded the redirect limit")

    async def _request(self, url: str) -> FetchedDocument | str:
        request = self._client.build_request("GET", url)
        response = await self._client.send(request, stream=True)
        try:
            if response.status_code in TRANSIENT_STATUSES:
                raise _TransientFetchError
            if response.status_code in REDIRECT_STATUSES:
                location = response.headers.get("location")
                if not location:
                    raise RemoteDocumentFetchError("Redirect response did not include a location")
                return cast(str, location)
            if response.is_error:
                raise RemoteDocumentFetchError(
                    f"Remote server returned HTTP {response.status_code}"
                )
            content_type = response.headers.get("content-type", "").partition(";")[0].lower()
            if content_type not in ALLOWED_CONTENT_TYPES:
                raise RemoteDocumentFetchError("Remote content type is not allowed")
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self._settings.max_response_bytes:
                raise DocumentTooLargeError("Remote response exceeds the configured size limit")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > self._settings.max_response_bytes:
                    raise DocumentTooLargeError("Remote response exceeds the configured size limit")
            return FetchedDocument(content=bytes(body), content_type=content_type, final_url=url)
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()
