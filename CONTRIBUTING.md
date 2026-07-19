# Contributing

## Local setup

Requirements: Python 3.14, uv 0.11.x, and GNU Make. Docker is optional.

```bash
cp .env.example .env
make install
make check
make dev
```

Default tests and evals must remain deterministic and offline. They use fake model/embeddings and Qdrant
memory mode; they must not download weights, require secrets, or start Docker.

## Making a change

- Keep domain and application independent of FastAPI, LangChain, Qdrant, Transformers, Torch, HTTPX,
  Prometheus, and structlog.
- Add external behavior behind a narrow port and concrete adapter.
- Keep model selection, paths, revisions, devices, and generation parameters server-side.
- Add tests for behavior and adapter contracts.
- Prompt/generation changes require RAG/chat/security eval updates.
- Chunking or retrieval changes require retrieval eval updates.
- Embedding or Qdrant schema changes require a collection migration note and compatibility tests.
- URL changes require SSRF/redirect tests; PDF changes require malformed/encrypted/textless tests.
- Run `make security` for security-sensitive or dependency changes.
- Update documentation and `.env.example` when public contracts or configuration change.

## Commands

```bash
make format
make lint
make typecheck
make test-unit
make test-integration
make eval
make check
make security
make build
```

`make test-model` and `make model-smoke` are opt-in and require pre-downloaded local files.

## Commits

Use focused Conventional Commits with an English summary:

```text
<type>(<scope>): <short summary>
```

Supported types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `style`.

Use a second paragraph for incompatible changes:

```bash
git commit -m "refactor(model): replace legacy runtime configuration" \
  -m "BREAKING CHANGE: configure an in-process local model through MODEL__* settings."
```

Do not commit model weights, caches, vector data, `.env`, coverage/build artifacts, or IDE state.

## Pull requests

Explain the affected boundary, security implications, migration needs, tests/evals, and actual verification
commands. Do not claim real-model, audit, Docker, or hardware behavior that was not executed.
