from dataclasses import replace
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from app.adapters.parsers.utils import normalize_text
from app.domain.documents import DocumentDescriptor, ParsedDocument, ParsedSection
from app.domain.errors import DocumentParsingError, NoExtractableTextError


class HtmlDocumentParser:
    def __init__(self, *, max_characters: int) -> None:
        self._max_characters = max_characters

    def supports(self, *, filename: str | None, content_type: str) -> bool:
        del filename
        return content_type.partition(";")[0].strip().lower() in {
            "text/html",
            "application/xhtml+xml",
        }

    def parse(self, *, content: bytes, descriptor: DocumentDescriptor) -> ParsedDocument:
        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception as exc:
            raise DocumentParsingError("HTML parsing failed") from exc
        for element in soup.select(
            "script, style, noscript, template, [hidden], [aria-hidden='true']"
        ):
            element.decompose()
        sections: list[ParsedSection] = []
        extracted_characters = 0
        current_heading: str | None = None
        for element in soup.select("h1, h2, h3, h4, h5, h6, p, li, pre, blockquote"):
            text = normalize_text(element.get_text(" ", strip=True))
            if not text:
                continue
            if element.name and element.name.startswith("h"):
                current_heading = text
                continue
            extracted_characters += len(text)
            if extracted_characters > self._max_characters:
                raise DocumentParsingError("Extracted HTML exceeds the configured character limit")
            sections.append(ParsedSection(text=text, heading=current_heading))
        if not sections:
            fallback = normalize_text(soup.get_text("\n", strip=True))
            if fallback:
                sections.append(ParsedSection(text=fallback))
        if not sections:
            raise NoExtractableTextError
        title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
        source = descriptor.source
        canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
        if canonical and canonical.get("href"):
            candidate = urljoin(source.source, str(canonical["href"]))
            if urlsplit(candidate).scheme in {"http", "https"}:
                source = replace(source, source=candidate)
        if title:
            source = replace(source, title=title[:500])
        return ParsedDocument(
            descriptor=replace(descriptor, source=source),
            sections=tuple(sections),
        )
