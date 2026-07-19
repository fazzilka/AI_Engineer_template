from typing import Protocol

from app.domain.documents import DocumentChunk, ParsedDocument


class TextChunker(Protocol):
    def split(self, document: ParsedDocument) -> tuple[DocumentChunk, ...]: ...
