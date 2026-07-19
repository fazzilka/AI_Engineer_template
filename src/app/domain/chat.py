from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str = Field(min_length=1, max_length=32_000)


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
