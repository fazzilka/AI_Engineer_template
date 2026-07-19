import re
import unicodedata

INLINE_WHITESPACE = re.compile(r"[^\S\n]+")
EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    normalized = INLINE_WHITESPACE.sub(" ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    return EXCESS_BLANK_LINES.sub("\n\n", normalized).strip()
