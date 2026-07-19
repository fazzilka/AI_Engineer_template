import asyncio
import time

import pytest

from app.adapters.embeddings.fake import FakeEmbeddingModel
from app.adapters.embeddings.huggingface import HuggingFaceEmbeddingModel
from app.adapters.llm.fake import FakeChatModel
from app.adapters.llm.huggingface import HuggingFaceChatModel
from app.adapters.model_runtime import select_device
from app.config import Device, EmbeddingSettings, ModelSettings
from app.domain.chat import ChatMessage, MessageRole
from app.domain.errors import EmbeddingError, ModelTimeoutError, ModelUnavailableError
from app.domain.generation import LifecycleState


class Tokenizer:
    def __init__(self, *, chat_template: str | None = None) -> None:
        self.chat_template = chat_template

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt
        return "|".join(item["content"] for item in conversation)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


class Pipeline:
    def __init__(self, output: str = "local answer", *, delay: float = 0) -> None:
        self.output = output
        self.delay = delay
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        return self.output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fake_model_contract_context_usage_and_controlled_failure() -> None:
    model = FakeChatModel(fail_on="FAIL")
    await model.warmup()
    response = await model.generate(
        messages=(
            ChatMessage(
                role=MessageRole.USER,
                content='<source id="source-1">Grounded local fact</source>',
            ),
        ),
        system_prompt="safe system prompt",
    )

    assert response.content == "Fake grounded answer: Grounded local fact"
    assert response.usage.total_tokens is not None
    assert model.status().state is LifecycleState.READY
    with pytest.raises(ModelUnavailableError, match="Controlled"):
        await model.generate(
            messages=(ChatMessage(role=MessageRole.USER, content="FAIL"),),
            system_prompt="safe",
        )
    await model.aclose()
    assert model.status().state is LifecycleState.CLOSED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fake_embeddings_are_deterministic_and_lexically_relevant() -> None:
    model = FakeEmbeddingModel(dimension=32)
    first, second, unrelated = await model.embed_documents(
        ["local qdrant vectors", "local qdrant vectors", "banana orchard"]
    )
    query = await model.embed_query("qdrant vectors")

    assert first == second
    assert len(first) == 32
    assert sum(a * b for a, b in zip(first, query, strict=True)) > sum(
        a * b for a, b in zip(unrelated, query, strict=True)
    )
    assert model.status().fingerprint
    await model.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_huggingface_model_uses_fallback_and_chat_templates() -> None:
    settings = ModelSettings(timeout_seconds=1)
    model = HuggingFaceChatModel(settings)
    pipeline = Pipeline()
    model._state = LifecycleState.READY
    model._pipeline = pipeline
    model._tokenizer = Tokenizer()

    result = await model.generate(
        messages=(ChatMessage(role=MessageRole.USER, content="question"),),
        system_prompt="system",
    )

    assert result.content == "local answer"
    assert pipeline.prompts[0] == "System: system\n\nUser: question\n\nAssistant:"
    assert result.usage.input_tokens == 5

    model._tokenizer = Tokenizer(chat_template="configured")
    prompt = model._format_prompt(
        messages=(ChatMessage(role=MessageRole.USER, content="question"),),
        system_prompt="system",
    )
    assert prompt == "system|question"
    await model.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_huggingface_model_load_is_singleton_and_wraps_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HuggingFaceChatModel(ModelSettings())
    loads = 0

    def load_sync() -> None:
        nonlocal loads
        loads += 1
        model._pipeline = Pipeline()
        model._tokenizer = Tokenizer()

    monkeypatch.setattr(model, "_load_sync", load_sync)
    await asyncio.gather(model.load(), model.load())
    assert loads == 1
    assert model.status().loaded
    await model.aclose()

    failing = HuggingFaceChatModel(ModelSettings())
    monkeypatch.setattr(failing, "_load_sync", lambda: (_ for _ in ()).throw(OSError()))
    with pytest.raises(ModelUnavailableError, match="model-download"):
        await failing.load()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_huggingface_timeout_does_not_cancel_worker_cleanup() -> None:
    model = HuggingFaceChatModel(ModelSettings(timeout_seconds=0.001))
    model._state = LifecycleState.READY
    model._pipeline = Pipeline(delay=0.02)
    model._tokenizer = Tokenizer()

    with pytest.raises(ModelTimeoutError):
        await model.generate(
            messages=(ChatMessage(role=MessageRole.USER, content="slow"),),
            system_prompt="system",
        )
    await model.aclose()


class EmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_huggingface_embedding_adapter_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HuggingFaceEmbeddingModel(EmbeddingSettings(query_prefix="q: "))

    def load_sync() -> None:
        model._client = EmbeddingClient()
        model._dimension = 2
        model._fingerprint = "fingerprint"

    monkeypatch.setattr(model, "_load_sync", load_sync)
    await model.initialize()
    assert await model.embed_query("text") == [7.0, 1.0]
    assert await model.embed_documents(("a", "bb")) == [[1.0, 1.0], [2.0, 1.0]]
    assert model.status().dimension == 2
    await model.aclose()

    failing = HuggingFaceEmbeddingModel(EmbeddingSettings())
    monkeypatch.setattr(failing, "_load_sync", lambda: (_ for _ in ()).throw(OSError()))
    with pytest.raises(EmbeddingError, match="loading failed"):
        await failing.initialize()


@pytest.mark.unit
def test_device_selection_auto_and_explicit_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device(Device.AUTO) == "cpu"
    assert select_device(Device.CPU) == "cpu"
    with pytest.raises(Exception, match="CUDA"):
        select_device(Device.CUDA)
