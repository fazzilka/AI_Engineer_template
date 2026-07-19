import hashlib
import re
import unicodedata
from pathlib import PurePath
from uuid import NAMESPACE_URL, uuid5

from app.domain.documents import DocumentSource

SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")


def content_checksum(content: bytes | str) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def document_identity(source: DocumentSource) -> str:
    stable_source = f"{source.source_type.value}:{source.source}"
    return str(uuid5(NAMESPACE_URL, stable_source))


def document_version(checksum: str) -> str:
    return checksum[:16]


def chunk_identity(
    *,
    document_id: str,
    version: str,
    chunk_index: int,
    chunk_checksum: str,
) -> str:
    value = f"{document_id}:{version}:{chunk_index}:{chunk_checksum}"
    return str(uuid5(NAMESPACE_URL, value))


def sanitize_filename(value: str) -> str:
    if "\x00" in value:
        msg = "Filename contains a null byte"
        raise ValueError(msg)
    name = unicodedata.normalize("NFKC", PurePath(value).name).strip()
    name = SAFE_FILENAME.sub("_", name).strip(" .")
    if not name or name in {".", ".."}:
        msg = "Filename is empty after sanitization"
        raise ValueError(msg)
    return name[:255]
