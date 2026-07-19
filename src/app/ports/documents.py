from typing import Protocol

from app.domain.documents import DocumentDescriptor, FetchedDocument, ParsedDocument


class DocumentParser(Protocol):
    def supports(self, *, filename: str | None, content_type: str) -> bool: ...

    def parse(self, *, content: bytes, descriptor: DocumentDescriptor) -> ParsedDocument: ...


class WebDocumentFetcher(Protocol):
    async def fetch(self, url: str) -> FetchedDocument: ...

    async def aclose(self) -> None: ...
