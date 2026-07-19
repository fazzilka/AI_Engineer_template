# Repository instructions for coding agents

## Purpose

This repository is a provider-free, production-oriented foundation for fully local AI and RAG projects.
Keep the core portable, offline-safe after model download, and explicit about security and operational
limits.

## Required workflow

1. Inspect the affected boundary before editing and preserve unrelated work.
2. Keep changes small and add/update tests for behavior changes.
3. External model APIs are prohibited without a separate architectural decision.
4. Real model downloads are prohibited in unit tests, default evals, and CI.
5. LangChain types must not enter domain or application models.
6. Model selection, paths, revisions, devices, and generation settings remain server-side.
7. Model code must cross a thread/process boundary and must not block the event loop.
8. Prompt/model parameter changes require eval updates.
9. Chunking changes require retrieval eval updates.
10. Embedding changes require a collection migration/reindex note.
11. Qdrant schema changes require compatibility tests.
12. URL ingestion changes require SSRF and redirect tests.
13. PDF parser changes require malformed, encrypted, oversized, and textless PDF tests.
14. Run `make check` before handoff.
15. Run `make security` for security-sensitive and dependency changes.
16. Do not push, rewrite history, or perform destructive Git operations unless explicitly asked.
17. Never commit model files, Hugging Face caches, vector data, secrets, or generated artifacts.

## Architecture rules

- `domain/`: immutable provider/framework-independent models, rules, and errors.
- `ports/`: narrow Protocol contracts used by application services.
- `application/`: orchestration; no FastAPI, Qdrant, LangChain, Transformers, Torch, HTTPX, Prometheus,
  structlog, or concrete adapters.
- `adapters/`: local models, embeddings, parsers, fetcher, chunker, and vector store implementations.
- `api/`: input validation and transport/domain mapping; no RAG/model/storage logic.
- `bootstrap/`: composition root and resource lifecycle.
- `observability/`: safe boundary wrappers with low-cardinality metrics and no content logging.

Unit/integration tests must not require network access, secrets, real weights, or a Qdrant server.

## Commit convention

Create focused Conventional Commits:

```text
<type>(<scope>): <short English summary>
```

Allowed types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `style`.

For incompatible changes, add an English `BREAKING CHANGE:` footer with migration guidance.

## Canonical commands

- Install: `make install`
- Install optional features: `make install-all`
- Run locally: `make dev`
- Format: `make format`
- Full offline verification: `make check`
- Dependency audit: `make security`
- Build artifacts: `make build`
- Build container: `make docker-build`
- Opt-in local model check: `make model-smoke`

Use uv only and keep `pyproject.toml` and `uv.lock` synchronized.
