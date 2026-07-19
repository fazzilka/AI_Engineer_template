from typing import Protocol

from app.config import ChunkingSettings
from app.domain.document_rules import chunk_identity, content_checksum
from app.domain.documents import DocumentChunk, ParsedDocument


class _Splitter(Protocol):
    def split_text(self, text: str) -> list[str]: ...


class LangChainTextChunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self._settings = settings
        self._splitter: _Splitter | None = None

    def _get_splitter(self) -> _Splitter:
        if self._splitter is not None:
            return self._splitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            separators=(list(self._settings.separators) if self._settings.separators else None),
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            length_function=len,
            keep_separator=True,
            strip_whitespace=True,
        )
        return self._splitter

    def split(self, document: ParsedDocument) -> tuple[DocumentChunk, ...]:
        descriptor = document.descriptor
        chunks: list[DocumentChunk] = []
        for section in document.sections:
            for text in self._get_splitter().split_text(section.text):
                normalized = text.strip()
                if not normalized:
                    continue
                checksum = content_checksum(normalized)
                index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        document_id=descriptor.document_id,
                        document_version=descriptor.document_version,
                        chunk_id=chunk_identity(
                            document_id=descriptor.document_id,
                            version=descriptor.document_version,
                            chunk_index=index,
                            chunk_checksum=checksum,
                        ),
                        chunk_index=index,
                        text=normalized,
                        chunk_checksum=checksum,
                        document_checksum=descriptor.document_checksum,
                        source_type=descriptor.source.source_type,
                        source=descriptor.source.source,
                        title=descriptor.source.title,
                        page_number=section.page_number,
                        content_type=descriptor.source.content_type,
                        ingested_at=descriptor.ingested_at,
                    )
                )
        return tuple(chunks)
