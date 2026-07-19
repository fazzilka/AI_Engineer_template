import asyncio
from collections.abc import Sequence
from typing import Any, Protocol, cast

import anyio
import structlog

from app.adapters.model_runtime import select_device
from app.config import ModelSettings, ModelSource
from app.domain.chat import ChatMessage, GenerationResult, MessageRole, TokenUsage
from app.domain.errors import ModelTimeoutError, ModelUnavailableError
from app.domain.generation import LifecycleState, ModelStatus


class _TextPipeline(Protocol):
    def invoke(self, prompt: str) -> str: ...


class _Tokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...


class HuggingFaceChatModel:
    """In-process Transformers model wrapped by LangChain's HuggingFacePipeline."""

    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings
        self._state = LifecycleState.UNLOADED
        self._device = settings.device.value
        self._pipeline: _TextPipeline | None = None
        self._tokenizer: _Tokenizer | None = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.concurrency)
        self._running: set[asyncio.Task[str]] = set()
        self._logger = structlog.get_logger(__name__)

    async def load(self) -> None:
        if self._state is LifecycleState.READY:
            return
        async with self._load_lock:
            if self._pipeline is not None:
                self._state = LifecycleState.READY
                return
            self._state = LifecycleState.LOADING
            self._logger.info("model_load_started", model_alias=self._settings.alias)
            if self._settings.trust_remote_code:
                self._logger.warning(
                    "model_trust_remote_code_enabled",
                    model_alias=self._settings.alias,
                )
            try:
                await anyio.to_thread.run_sync(self._load_sync)
            except Exception as exc:
                self._state = LifecycleState.FAILED
                self._logger.exception("model_load_failed", model_alias=self._settings.alias)
                msg = (
                    "Local model loading failed. Pre-download it with "
                    "`make model-download` and verify MODEL__PATH or the Hugging Face cache."
                )
                raise ModelUnavailableError(msg) from exc
            self._state = LifecycleState.READY
            self._logger.info(
                "model_load_completed",
                model_alias=self._settings.alias,
                device=self._device,
            )

    def _load_sync(self) -> None:  # pragma: no cover - exercised by opt-in local model smoke
        import torch
        from langchain_huggingface import HuggingFacePipeline
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )

        reference = (
            str(self._settings.path.resolve())
            if self._settings.source is ModelSource.FILESYSTEM
            else self._settings.id
        )
        revision = self._settings.revision or None
        tokenizer = AutoTokenizer.from_pretrained(
            reference,
            revision=revision,
            local_files_only=self._settings.local_files_only,
            trust_remote_code=self._settings.trust_remote_code,
        )
        dtype: str | torch.dtype = self._settings.dtype
        if self._settings.dtype != "auto":
            dtype = getattr(torch, self._settings.dtype)
        model_factory = (
            AutoModelForCausalLM
            if self._settings.task == "text-generation"
            else AutoModelForSeq2SeqLM
        )
        model: Any = model_factory.from_pretrained(
            reference,
            revision=revision,
            local_files_only=self._settings.local_files_only,
            trust_remote_code=self._settings.trust_remote_code,
            dtype=dtype,
        )
        self._device = select_device(self._settings.device)
        model.to(self._device)
        pipeline_factory: Any = pipeline
        generator: Any = pipeline_factory(
            task=self._settings.task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self._settings.max_new_tokens,
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            do_sample=self._settings.do_sample,
            repetition_penalty=self._settings.repetition_penalty,
            return_full_text=False,
        )
        self._pipeline = cast(_TextPipeline, HuggingFacePipeline(pipeline=generator))
        self._tokenizer = cast(_Tokenizer, tokenizer)

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> GenerationResult:
        await self.load()
        prompt = self._format_prompt(messages=messages, system_prompt=system_prompt)
        input_tokens = await self.count_tokens(prompt)
        if input_tokens is not None and input_tokens > self._settings.max_input_tokens:
            raise ModelUnavailableError("The prompt exceeds the configured model context limit")
        pipeline = self._pipeline
        if pipeline is None:
            raise ModelUnavailableError("The local model is not loaded")

        await self._semaphore.acquire()
        task = asyncio.create_task(anyio.to_thread.run_sync(pipeline.invoke, prompt))
        self._running.add(task)

        def release(completed: asyncio.Task[str]) -> None:
            self._running.discard(completed)
            self._semaphore.release()

        task.add_done_callback(release)
        self._logger.info(
            "generation_started",
            model_alias=self._settings.alias,
            device=self._device,
        )
        try:
            async with asyncio.timeout(self._settings.timeout_seconds):
                content = await asyncio.shield(task)
        except TimeoutError as exc:
            self._logger.warning("generation_failed", outcome="timeout")
            raise ModelTimeoutError from exc
        except Exception as exc:
            self._logger.exception("generation_failed", outcome="error")
            raise ModelUnavailableError("Local model inference failed") from exc
        output_tokens = await self.count_tokens(content)
        self._logger.info("generation_completed", model_alias=self._settings.alias)
        return GenerationResult(
            content=content.strip(),
            model=self._settings.alias,
            finish_reason="stop",
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated=False,
            ),
        )

    def _format_prompt(
        self,
        *,
        messages: Sequence[ChatMessage],
        system_prompt: str,
    ) -> str:
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise ModelUnavailableError("The tokenizer is not loaded")
        conversation = [{"role": "system", "content": system_prompt}]
        conversation.extend(
            {
                "role": "user" if message.role is MessageRole.USER else "assistant",
                "content": message.content,
            }
            for message in messages
        )
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = [f"System: {system_prompt}"]
        lines.extend(
            f"{'User' if message.role is MessageRole.USER else 'Assistant'}: {message.content}"
            for message in messages
        )
        lines.append("Assistant:")
        return "\n\n".join(lines)

    async def count_tokens(self, text: str) -> int | None:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return None
        return await anyio.to_thread.run_sync(
            lambda: len(tokenizer.encode(text, add_special_tokens=True))
        )

    async def warmup(self) -> None:
        await self.load()
        await self.generate(
            messages=(ChatMessage(role=MessageRole.USER, content="Hello"),),
            system_prompt="Answer briefly.",
        )

    def status(self) -> ModelStatus:
        source = "filesystem" if self._settings.source is ModelSource.FILESYSTEM else "cache"
        return ModelStatus(
            backend="huggingface",
            model_alias=self._settings.alias,
            source=source,
            device=self._device,
            dtype=self._settings.dtype,
            state=self._state,
            local_files_only=self._settings.local_files_only,
            max_input_tokens=self._settings.max_input_tokens,
            max_new_tokens=self._settings.max_new_tokens,
        )

    async def aclose(self) -> None:
        if self._running:
            await asyncio.gather(*tuple(self._running), return_exceptions=True)
        await anyio.to_thread.run_sync(self._cleanup_sync)
        self._state = LifecycleState.CLOSED

    def _cleanup_sync(self) -> None:
        self._pipeline = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError, RuntimeError:
            return
