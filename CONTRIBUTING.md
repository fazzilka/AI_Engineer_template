# Contributing

## Local setup

Requirements: Python 3.12+, `uv` 0.11.x, and GNU Make. Docker is optional.

```bash
cp .env.example .env
make install
make check
make dev
```

Keep tests offline by default. The configured fake LLM adapter makes the complete API path available
without credentials.

## Making a change

- Put business rules in `application/` or `domain/`, not in HTTP routes.
- Add external integrations behind a port and an adapter.
- Add tests for new behavior and regressions.
- Update `evals/cases.jsonl` when generation behavior or prompts change.
- Update documentation when configuration or public contracts change.
- Run `make check` before opening a pull request.

## Commits

Use small Conventional Commits with an English summary:

```text
<type>(<scope>): <short summary>
```

Supported types are `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, and `style`.

Use a second paragraph for incompatible changes:

```bash
git commit -m "feat(api): replace legacy completion contract" \
  -m "BREAKING CHANGE: clients must send a messages array instead of a prompt string"
```

## Pull requests

Explain the problem and the chosen boundary, keep unrelated changes out, and include verification
evidence. A pull request that changes AI behavior should describe both the eval dataset change and the
observed result.
