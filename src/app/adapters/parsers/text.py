from dataclasses import replace

from app.adapters.parsers.utils import normalize_text
from app.domain.documents import DocumentDescriptor, ParsedDocument, ParsedSection, SourceType
from app.domain.errors import DocumentParsingError, NoExtractableTextError

TEXT_CONTENT_TYPES = {"text/plain", "text/markdown", "text/x-markdown"}


class TextDocumentParser:
    def __init__(self, *, max_characters: int) -> None:
        self._max_characters = max_characters

    def supports(self, *, filename: str | None, content_type: str) -> bool:
        media_type = content_type.partition(";")[0].strip().lower()
        suffix = (filename or "").lower()
        return media_type in TEXT_CONTENT_TYPES or suffix.endswith((".txt", ".md"))

    def parse(self, *, content: bytes, descriptor: DocumentDescriptor) -> ParsedDocument:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParsingError("Text documents must be valid UTF-8") from exc
        text = normalize_text(text)
        if not text:
            raise NoExtractableTextError
        if len(text) > self._max_characters:
            raise DocumentParsingError("Extracted text exceeds the configured character limit")
        source = descriptor.source
        if source.source_type is SourceType.MARKDOWN:
            heading = next(
                (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")),
                "",
            )
            if heading:
                descriptor = replace(descriptor, source=replace(source, title=heading[:500]))
        return ParsedDocument(descriptor=descriptor, sections=(ParsedSection(text=text),))
