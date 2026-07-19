# Security model

This template minimizes local AI and RAG attack surfaces but still requires deployment-specific access
control, TLS, rate limiting, capacity limits, and monitoring.

## Enforced boundaries

- No model API credentials or external inference providers.
- HTTP requests cannot select a model, revision, device, local path, Qdrant URL, or collection path.
- `trust_remote_code` defaults to false and is never enabled automatically; explicit use emits a warning.
- Uploads are bounded by bytes, PDF pages, and extracted characters. PDF signature is checked.
- Filenames are normalized to a basename; upload bytes never become an arbitrary filesystem read.
- Encrypted, corrupted, textless, oversized, and unsupported documents produce typed public errors.
- URL ingestion allows HTTP(S) only, rejects embedded credentials, resolves DNS, blocks non-public
  addresses by default, and revalidates every redirect.
- Remote responses use connect/read timeouts, bounded transient retries, content-type allowlisting,
  streamed size enforcement, and redirect limits.
- HTML parsing executes no JavaScript and removes script/style/noscript/hidden technical elements.
- System prompts remain server-side. Retrieved documents are escaped and treated as untrusted data.
- Public errors contain stable codes and request IDs without stack traces, local paths, or exception text.
- Logs omit prompts, answers, retrieved context, document bodies, upload bytes, URLs, filenames, tokens,
  and secrets.
- Prometheus labels remain low-cardinality and exclude request/document IDs and user data.
- Models, Hugging Face caches, vector data, env files, and generated artifacts are ignored by Git and the
  Docker build context.
- Container runtime is non-root, read-only, capability-free, and `no-new-privileges`; only explicit
  volumes and tmpfs are writable.
- `make security` audits Python dependencies; CI runs it separately from offline `make check`.

## SSRF residual risk

DNS is validated before each request and after redirects. Network policy should additionally deny access
to cloud metadata, management, and private ranges at the container/host/egress layer. DNS rebinding and
proxy behavior are best controlled with egress firewall rules; the HTTP client ignores ambient proxy
environment variables by default.

Set `WEB__ENABLED=false` in isolated deployments.

## Prompt injection

Prompt boundaries reduce instruction confusion but model output is never authorization. Do not connect
RAG answers directly to destructive tools, secrets, or privileged actions. Validate and authorize every
side effect outside the model. Add product-specific attack fixtures to `evals/cases/security.jsonl`.

## Model supply chain

- Pin immutable revisions during download.
- Review model code, files, license, and provenance.
- Keep `MODEL__TRUST_REMOTE_CODE=false` unless a documented review accepts the code-execution risk.
- Verify artifacts using organization-approved checksums/signatures when available.
- Store weights in read-only runtime volumes.

## Deployment checklist

1. Add authentication and per-document authorization.
2. Enforce ingress body/rate/concurrency limits in addition to application limits.
3. Deny unsafe egress at the network layer.
4. Protect `/metrics`, health, and API documentation as required.
5. Define retention, deletion, backup, and incident-response policies for vector data.
6. Size RAM/VRAM and set orchestrator requests/limits from measured model behavior.
7. Alert on load failures, timeouts, ingestion errors, retrieval regressions, and storage growth.
