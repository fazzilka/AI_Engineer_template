from functools import lru_cache
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    """Configuration shared by every LLM adapter."""

    provider: Literal["fake", "openai_compatible"] = "fake"
    api_key: SecretStr | None = None
    base_url: str | None = None
    model: str = "fake-model"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1_024, ge=1)
    timeout_seconds: float = Field(default=30, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def require_credentials_for_remote_provider(self) -> Self:
        if self.provider == "openai_compatible" and self.api_key is None:
            msg = "LLM__API_KEY is required when LLM__PROVIDER=openai_compatible"
            raise ValueError(msg)
        return self


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "AI Engineer Template"
    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_prefix: str = "/api/v1"
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
