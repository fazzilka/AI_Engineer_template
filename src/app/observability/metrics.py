from collections.abc import Sequence
from time import perf_counter

from prometheus_client import Counter, Histogram

from app.domain.chat import ChatMessage, GenerationResult
from app.ports.llm import LLMClient

HTTP_REQUESTS = Counter(
    "app_http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "app_http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "route"),
)
LLM_REQUESTS = Counter(
    "app_llm_requests_total",
    "Total LLM generation requests",
    labelnames=("provider", "status"),
)
LLM_REQUEST_DURATION = Histogram(
    "app_llm_request_duration_seconds",
    "LLM request duration in seconds",
    labelnames=("provider",),
)
LLM_TOKENS = Counter(
    "app_llm_tokens_total",
    "Tokens reported by the LLM provider",
    labelnames=("provider", "direction"),
)


class InstrumentedLLMClient:
    def __init__(self, *, client: LLMClient, provider: str) -> None:
        self._client = client
        self._provider = provider

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        started_at = perf_counter()
        try:
            result = await self._client.generate(messages=messages, system_prompt=system_prompt)
        except Exception:
            LLM_REQUESTS.labels(provider=self._provider, status="error").inc()
            raise
        else:
            LLM_REQUESTS.labels(provider=self._provider, status="success").inc()
            LLM_TOKENS.labels(provider=self._provider, direction="input").inc(
                result.usage.input_tokens
            )
            LLM_TOKENS.labels(provider=self._provider, direction="output").inc(
                result.usage.output_tokens
            )
            return result
        finally:
            LLM_REQUEST_DURATION.labels(provider=self._provider).observe(
                perf_counter() - started_at
            )
