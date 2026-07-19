import hashlib
import json
from pathlib import Path


def embedding_fingerprint(
    *,
    model_reference: str | Path,
    revision: str,
    normalize: bool,
    dimension: int,
    query_prefix: str,
    document_prefix: str,
) -> str:
    canonical_reference = (
        str(model_reference.resolve()) if isinstance(model_reference, Path) else model_reference
    )
    payload = {
        "document_prefix": document_prefix,
        "dimension": dimension,
        "model": canonical_reference,
        "normalize": normalize,
        "query_prefix": query_prefix,
        "revision": revision,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()
