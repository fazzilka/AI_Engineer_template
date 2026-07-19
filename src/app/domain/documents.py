from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SourceType(StrEnum):
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


class IngestionStatus(StrEnum):
    INDEXED = "indexed"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class DocumentSource:
    source_type: SourceType
    source: str
    title: str
    content_type: str
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentDescriptor:
    document_id: str
    document_version: str
    document_checksum: str
    source: DocumentSource
    ingested_at: datetime


@dataclass(frozen=True, slots=True)
class ParsedSection:
    text: str
    page_number: int | None = None
    heading: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    descriptor: DocumentDescriptor
    sections: tuple[ParsedSection, ...]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    document_id: str
    document_version: str
    chunk_id: str
    chunk_index: int
    text: str
    chunk_checksum: str
    document_checksum: str
    source_type: SourceType
    source: str
    title: str
    page_number: int | None
    content_type: str
    ingested_at: datetime


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    document_version: str
    status: IngestionStatus
    chunk_count: int


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    content: bytes
    content_type: str
    final_url: str
