# Repository instructions for coding agents

## Purpose

This repository is a reusable, production-oriented foundation for AI engineering projects.
Keep the core provider-neutral and make optional capabilities easy to add without forcing them on
every generated project.

## Required workflow

1. Inspect the affected boundary before editing.
2. Keep changes small and scoped. Preserve unrelated user work.
3. Add or update tests for behavior changes.
4. When prompts, model parameters, retrieval, or generation behavior changes, update `evals/`.
5. Run `make check` before handing work back.
6. Do not push, rewrite history, or perform destructive Git operations unless explicitly asked.

## Architecture rules

- `domain/` contains provider- and framework-independent models.
- `ports/` defines interfaces used by the application.
- `application/` orchestrates use cases and depends only on domain models and ports.
- `adapters/` implements external systems such as LLM providers, stores, queues, and vector databases.
- `api/` maps HTTP contracts to application use cases; it does not contain business logic.
- `observability/` wraps boundaries without leaking Prometheus or logging concerns into domain code.
- Keep model selection and credentials server-side. Never accept arbitrary provider credentials from an
  API request.
- Unit and integration tests must not require network access or real secrets.

## Commit convention

Create focused commits, preferably one independently reviewable file or concern at a time. Use:

```text
<type>(<scope>): <short English summary>
```

Allowed types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `style`.

For an incompatible change, add an English footer in a second commit message paragraph:

```text
BREAKING CHANGE: <clear description and migration guidance>
```

## Canonical commands

- Install: `make install`
- Run locally: `make dev`
- Format: `make format`
- Full verification: `make check`
- Build artifacts: `make build`
- Build container: `make docker-build`

Use `uv` for dependencies and keep `uv.lock` synchronized with `pyproject.toml`.
