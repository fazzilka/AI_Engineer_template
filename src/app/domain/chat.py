from dataclasses import dataclass
from enum import StrEnum


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not self.content or len(self.content) > 32_000:
            msg = "Chat message content must contain between 1 and 32000 characters"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated: bool = False

    def __post_init__(self) -> None:
        values = (self.input_tokens, self.output_tokens, self.total_tokens)
        if any(value is not None and value < 0 for value in values):
            msg = "Token counts cannot be negative"
            raise ValueError(msg)
        if (
            self.total_tokens is None
            and self.input_tokens is not None
            and self.output_tokens is not None
        ):
            object.__setattr__(
                self,
                "total_tokens",
                self.input_tokens + self.output_tokens,
            )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    content: str
    model: str
    finish_reason: str | None = None
    usage: TokenUsage = TokenUsage()

    def __post_init__(self) -> None:
        if not self.content:
            msg = "Generation content cannot be empty"
            raise ValueError(msg)
        if not self.model:
            msg = "Model alias cannot be empty"
            raise ValueError(msg)
