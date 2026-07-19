from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from typing import ClassVar

import pytest
from pypdf import PdfWriter

from app.adapters.chunking import LangChainTextChunker
from app.adapters.parsers.html import HtmlDocumentParser
from app.adapters.parsers.pdf import PdfDocumentParser
from app.adapters.parsers.text import TextDocumentParser
from app.config import ChunkingSettings
from app.domain.document_rules import (
    chunk_identity,
    content_checksum,
    document_identity,
    document_version,
    sanitize_filename,
)
from app.domain.documents import DocumentDescriptor, DocumentSource, SourceType
from app.domain.errors import (
    DocumentParsingError,
    NoExtractableTextError,
    PdfCorruptedError,
    PdfEncryptedError,
)


def descriptor(source_type: SourceType = SourceType.TEXT) -> DocumentDescriptor:
    source = DocumentSource(
        source_type=source_type,
        source="notes.txt",
        title="notes",
        content_type="text/plain",
        filename="notes.txt",
    )
    checksum = content_checksum("content")
    return DocumentDescriptor(
        document_id=document_identity(source),
        document_version=document_version(checksum),
        document_checksum=checksum,
        source=source,
        ingested_at=datetime.now(UTC),
    )


@pytest.mark.unit
def test_document_rules_are_deterministic_and_sanitize_paths() -> None:
    source = descriptor().source
    checksum = content_checksum("same")

    assert content_checksum("same") == checksum
    assert document_identity(source) == document_identity(source)
    assert document_version(checksum) == checksum[:16]
    assert chunk_identity(
        document_id="doc",
        version="v1",
        chunk_index=0,
        chunk_checksum=checksum,
    ) == chunk_identity(
        document_id="doc",
        version="v1",
        chunk_index=0,
        chunk_checksum=checksum,
    )
    assert sanitize_filename("../../unsafe notes.md") == "unsafe notes.md"
    with pytest.raises(ValueError, match="null byte"):
        sanitize_filename("bad\x00name.txt")
    with pytest.raises(ValueError, match="empty"):
        sanitize_filename("../..")


@pytest.mark.unit
def test_text_parser_normalizes_utf8_and_markdown_title() -> None:
    parser = TextDocumentParser(max_characters=1_000)
    markdown_descriptor = replace(
        descriptor(SourceType.MARKDOWN),
        source=DocumentSource(
            source_type=SourceType.MARKDOWN,
            source="notes.md",
            title="notes",
            content_type="text/markdown",
            filename="notes.md",
        ),
    )

    parsed = parser.parse(
        content=b"# Heading\r\n\r\nBody   text",
        descriptor=markdown_descriptor,
    )

    assert parsed.descriptor.source.title == "Heading"
    assert parsed.sections[0].text == "# Heading\n\nBody text"
    assert parser.supports(filename="note.txt", content_type="application/octet-stream")
    with pytest.raises(DocumentParsingError, match="UTF-8"):
        parser.parse(content=b"\xff", descriptor=descriptor())
    with pytest.raises(NoExtractableTextError):
        parser.parse(content=b"  \n", descriptor=descriptor())
    with pytest.raises(DocumentParsingError, match="character limit"):
        TextDocumentParser(max_characters=3).parse(
            content=b"too long",
            descriptor=descriptor(),
        )


@pytest.mark.unit
def test_html_parser_removes_technical_content_and_keeps_boundaries() -> None:
    source = DocumentSource(
        source_type=SourceType.HTML,
        source="https://example.com/article",
        title="example.com",
        content_type="text/html",
    )
    html_descriptor = replace(descriptor(SourceType.HTML), source=source)
    parser = HtmlDocumentParser(max_characters=1_000)

    parsed = parser.parse(
        content=(
            b"<html><head><title>Article</title>"
            b"<link rel='canonical' href='/canonical'></head>"
            b"<body><script>secret()</script><h1>Topic</h1>"
            b"<p>First paragraph.</p><p hidden>hidden</p><p>Second paragraph.</p>"
            b"</body></html>"
        ),
        descriptor=html_descriptor,
    )

    assert parsed.descriptor.source.title == "Article"
    assert parsed.descriptor.source.source == "https://example.com/canonical"
    assert [section.text for section in parsed.sections] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert all(section.heading == "Topic" for section in parsed.sections)
    assert parser.supports(filename=None, content_type="text/html; charset=utf-8")
    with pytest.raises(NoExtractableTextError):
        parser.parse(content=b"<script>only()</script>", descriptor=html_descriptor)


@pytest.mark.unit
def test_pdf_parser_rejects_signature_encryption_and_empty_pages() -> None:
    parser = PdfDocumentParser(max_pages=2, max_characters=1_000)

    with pytest.raises(PdfCorruptedError, match="signature"):
        parser.parse(content=b"not-pdf", descriptor=descriptor(SourceType.PDF))

    encrypted = PdfWriter()
    encrypted.add_blank_page(width=72, height=72)
    encrypted.encrypt("password")
    encrypted_bytes = BytesIO()
    encrypted.write(encrypted_bytes)
    with pytest.raises(PdfEncryptedError):
        parser.parse(content=encrypted_bytes.getvalue(), descriptor=descriptor(SourceType.PDF))

    blank = PdfWriter()
    blank.add_blank_page(width=72, height=72)
    blank_bytes = BytesIO()
    blank.write(blank_bytes)
    with pytest.raises(NoExtractableTextError):
        parser.parse(content=blank_bytes.getvalue(), descriptor=descriptor(SourceType.PDF))


@pytest.mark.unit
def test_pdf_parser_preserves_page_number_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class Metadata:
        title = "PDF title"

    class Reader:
        is_encrypted = False
        pages: ClassVar[list[Page]] = [Page("Page one"), Page("   ")]
        metadata = Metadata()

    monkeypatch.setattr("app.adapters.parsers.pdf.PdfReader", lambda *_args, **_kwargs: Reader())
    parser = PdfDocumentParser(max_pages=2, max_characters=1_000)

    parsed = parser.parse(
        content=b"%PDF-mocked",
        descriptor=descriptor(SourceType.PDF),
    )

    assert parsed.descriptor.source.title == "PDF title"
    assert parsed.sections[0].page_number == 1
    assert parsed.sections[0].text == "Page one"


@pytest.mark.unit
def test_pdf_parser_enforces_page_and_character_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        def extract_text(self) -> str:
            return "long text"

    class Reader:
        is_encrypted = False
        pages: ClassVar[list[Page]] = [Page(), Page()]
        metadata = None

    monkeypatch.setattr("app.adapters.parsers.pdf.PdfReader", lambda *_args, **_kwargs: Reader())
    with pytest.raises(DocumentParsingError, match="page limit"):
        PdfDocumentParser(max_pages=1, max_characters=100).parse(
            content=b"%PDF-pages",
            descriptor=descriptor(SourceType.PDF),
        )
    with pytest.raises(DocumentParsingError, match="character limit"):
        PdfDocumentParser(max_pages=2, max_characters=5).parse(
            content=b"%PDF-chars",
            descriptor=descriptor(SourceType.PDF),
        )


@pytest.mark.unit
def test_chunker_is_deterministic_and_preserves_metadata() -> None:
    parser = TextDocumentParser(max_characters=2_000)
    document = parser.parse(
        content=("alpha beta gamma delta " * 20).encode(),
        descriptor=descriptor(),
    )
    chunker = LangChainTextChunker(ChunkingSettings(chunk_size=80, chunk_overlap=15))

    first = chunker.split(document)
    second = chunker.split(document)

    assert first == second
    assert first
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))
    assert all(chunk.text and len(chunk.text) <= 80 for chunk in first)
    assert all(chunk.document_id == document.descriptor.document_id for chunk in first)
