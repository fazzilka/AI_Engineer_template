import ipaddress
from ipaddress import IPv4Address, IPv6Address

import httpx
import pytest

from app.adapters.web.httpx_fetcher import HttpxWebDocumentFetcher, validate_url
from app.config import WebFetchSettings
from app.domain.errors import (
    DocumentTooLargeError,
    RemoteDocumentFetchError,
    UnsafeUrlError,
)


async def public_resolver(
    _host: str,
    _port: int,
) -> tuple[IPv4Address | IPv6Address, ...]:
    return (ipaddress.ip_address("93.184.216.34"),)


async def private_resolver(
    _host: str,
    _port: int,
) -> tuple[IPv4Address | IPv6Address, ...]:
    return (ipaddress.ip_address("127.0.0.1"),)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/file", "https://user:pass@example.com/"],
)
async def test_url_validation_rejects_unsupported_or_credentialed_urls(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        await validate_url(url, allow_private_hosts=False, resolver=public_resolver)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_url_validation_checks_resolved_addresses() -> None:
    with pytest.raises(UnsafeUrlError, match="non-public"):
        await validate_url(
            "http://localhost/resource",
            allow_private_hosts=False,
            resolver=private_resolver,
        )

    result = await validate_url(
        "HTTPS://example.com/path#fragment",
        allow_private_hosts=False,
        resolver=public_resolver,
    )
    assert result == "https://example.com/path"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetcher_revalidates_redirect_destination() -> None:
    async def resolver(
        host: str,
        _port: int,
    ) -> tuple[IPv4Address | IPv6Address, ...]:
        address = "127.0.0.1" if host == "internal.test" else "93.184.216.34"
        return (ipaddress.ip_address(address),)

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(302, headers={"location": "http://internal.test/"})
    )
    client = httpx.AsyncClient(transport=transport)
    fetcher = HttpxWebDocumentFetcher(
        WebFetchSettings(),
        client=client,
        resolver=resolver,
    )

    with pytest.raises(UnsafeUrlError):
        await fetcher.fetch("https://example.com/start")
    await client.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetcher_streams_allowed_content_and_enforces_limits() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"<p>safe</p>",
                request=request,
            )
        )
    )
    fetcher = HttpxWebDocumentFetcher(
        WebFetchSettings(max_response_bytes=100),
        client=client,
        resolver=public_resolver,
    )
    result = await fetcher.fetch("https://example.com/")
    assert result.content == b"<p>safe</p>"

    limited = HttpxWebDocumentFetcher(
        WebFetchSettings(max_response_bytes=3),
        client=client,
        resolver=public_resolver,
    )
    with pytest.raises(DocumentTooLargeError):
        await limited.fetch("https://example.com/")
    await client.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetcher_retries_transient_status_and_rejects_content_type() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"{}",
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = HttpxWebDocumentFetcher(
        WebFetchSettings(max_retries=1),
        client=client,
        resolver=public_resolver,
    )

    with pytest.raises(RemoteDocumentFetchError, match="content type"):
        await fetcher.fetch("https://example.com/")
    assert calls == 2
    await client.aclose()
