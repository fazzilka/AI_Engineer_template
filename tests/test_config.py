import pytest
from pydantic import ValidationError

from app.config import LLMSettings, Settings, get_settings


def test_settings_load_nested_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("LLM__PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM__API_KEY", "secret")
    monkeypatch.setenv("LLM__MODEL", "custom-model")

    settings = Settings()

    assert settings.app_env == "staging"
    assert settings.llm.provider == "openai_compatible"
    assert settings.llm.model == "custom-model"
    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "secret"


def test_remote_provider_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="LLM__API_KEY"):
        LLMSettings(provider="openai_compatible")


def test_empty_environment_secret_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM__API_KEY", "")

    with pytest.raises(ValidationError, match="LLM__API_KEY"):
        Settings()


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()
