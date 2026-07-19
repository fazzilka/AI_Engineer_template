from dataclasses import dataclass
from enum import StrEnum


class LifecycleState(StrEnum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ModelStatus:
    backend: str
    model_alias: str
    source: str
    device: str
    dtype: str
    state: LifecycleState
    local_files_only: bool
    max_input_tokens: int
    max_new_tokens: int

    @property
    def loaded(self) -> bool:
        return self.state is LifecycleState.READY

    @property
    def loading(self) -> bool:
        return self.state is LifecycleState.LOADING


@dataclass(frozen=True, slots=True)
class EmbeddingStatus:
    backend: str
    model_alias: str
    device: str
    state: LifecycleState
    dimension: int | None
    fingerprint: str | None
