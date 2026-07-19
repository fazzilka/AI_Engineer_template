from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.chat import ChatMessage, GenerationResult, MessageRole, TokenUsage


class ChatMessagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1, max_length=32_000)

    def to_domain(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessagePayload] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def require_final_user_message(self) -> Self:
        if self.messages[-1].role is not MessageRole.USER:
            msg = "The final message must have the user role"
            raise ValueError(msg)
        return self


class ChatResponse(BaseModel):
    content: str
    model: str
    finish_reason: str | None
    usage: TokenUsage

    @classmethod
    def from_result(cls, result: GenerationResult) -> Self:
        return cls(
            content=result.content,
            model=result.model,
            finish_reason=result.finish_reason,
            usage=result.usage,
        )


class HealthResponse(BaseModel):
    status: str
