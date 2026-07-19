# Extension recipes

These capabilities are intentionally optional.

## OCR for scanned PDF

Add an OCR adapter behind `DocumentParser` or a dedicated narrow port. Keep OCR binaries and language
packs in a separate image/profile, enforce page/time limits, and add malformed/scanned PDF tests. The
baseline returns `document_contains_no_extractable_text` when no text layer exists.

## Reranking

Add a `Reranker` port between Qdrant retrieval and context budgeting. Keep it local, bounded, and covered
by retrieval evals. Do not replace source scores or citations without documenting the semantics.

## Authentication and authorization

Authenticate at an API gateway or FastAPI dependency. Apply document-level authorization before search
and deletion; never accept raw Qdrant filter DSL from clients.

## Background ingestion

Invoke existing ingestion services from a queue adapter. Preserve document identity, idempotency,
request/correlation IDs, timeouts, and deletion semantics. Redis, PostgreSQL, Celery, and other job
systems are intentionally absent from the baseline.

## Alternative local runtimes

An alternative runtime requires an architectural decision, an implementation of `ChatModel`, lifecycle
and thread-boundary tests, offline behavior, token usage semantics, and model-license documentation.
External model APIs are not a drop-in extension.

## Multiple replicas

Use one model-owning worker per replica and distribute requests at the service layer. Use Qdrant server
mode for shared retrieval. Plan Prometheus aggregation, graceful rollout memory spikes, and GPU affinity.
