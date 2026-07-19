from dataclasses import replace
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.adapters.parsers.utils import normalize_text
from app.domain.documents import DocumentDescriptor, ParsedDocument, ParsedSection
from app.domain.errors import (
    DocumentParsingError,
    NoExtractableTextError,
    PdfCorruptedError,
    PdfEncryptedError,
)


class PdfDocumentParser:
    def __init__(self, *, max_pages: int, max_characters: int) -> None:
        self._max_pages = max_pages
        self._max_characters = max_characters

    def supports(self, *, filename: str | None, content_type: str) -> bool:
        media_type = content_type.partition(";")[0].strip().lower()
        return media_type == "application/pdf" or (filename or "").lower().endswith(".pdf")

    def parse(self, *, content: bytes, descriptor: DocumentDescriptor) -> ParsedDocument:
        if not content.startswith(b"%PDF-"):
            raise PdfCorruptedError("The file does not have a PDF signature")
        try:
            reader = PdfReader(BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise PdfEncryptedError
            if len(reader.pages) > self._max_pages:
                raise DocumentParsingError("The PDF exceeds the configured page limit")
            sections: list[ParsedSection] = []
            extracted_characters = 0
            for page_number, page in enumerate(reader.pages, start=1):
                text = normalize_text(page.extract_text() or "")
                if not text:
                    continue
                extracted_characters += len(text)
                if extracted_characters > self._max_characters:
                    raise DocumentParsingError(
                        "Extracted PDF text exceeds the configured character limit"
                    )
                sections.append(ParsedSection(text=text, page_number=page_number))
            if not sections:
                raise NoExtractableTextError
            title = normalize_text(str(reader.metadata.title or "")) if reader.metadata else ""
        except PdfEncryptedError, NoExtractableTextError, DocumentParsingError:
            raise
        except (PdfReadError, ValueError, TypeError, KeyError) as exc:
            raise PdfCorruptedError from exc
        except Exception as exc:
            raise PdfCorruptedError from exc
        if title:
            descriptor = replace(
                descriptor,
                source=replace(descriptor.source, title=title[:500]),
            )
        return ParsedDocument(descriptor=descriptor, sections=tuple(sections))
