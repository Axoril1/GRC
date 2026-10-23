"""Tests for configuration parsing, environment loading, and settings."""

from pathlib import Path

import pytest

from repo2video.config import Settings, get_settings
from repo2video.errors import ConfigError

ENV_KEYS = [
    "GITHUB_TOKEN",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_TIMEOUT_SECONDS",
    "LLM_TEMPERATURE",
    "LLM_NUM_CTX",
    "PIPER_VOICE_PATH",
    "OUTPUT_DIR",
    "VIDEO_WIDTH",
    "VIDEO_HEIGHT",
    "VIDEO_FPS",
    "SCENE_PAUSE_SECONDS",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a clean environment for each test and clear cached settings."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


def test_settings_defaults() -> None:
    """Verify default values of Settings when no environment variables are set."""
    settings = get_settings()

    assert settings.github_token is None
    assert settings.llm_provider == "ollama"
    assert settings.llm_base_url == "http://localhost:11434"
    assert settings.llm_model == "qwen2.5:7b"
    assert settings.llm_api_key is None
    assert settings.llm_timeout_seconds == 300
    assert settings.llm_temperature == 0.4
    assert settings.llm_num_ctx == 8192
    assert settings.piper_voice_path == Path("voices/en_US-lessac-medium.onnx")
    assert settings.output_dir == Path("output")
    assert settings.video_width == 1920
    assert settings.video_height == 1080
    assert settings.video_fps == 30
    assert settings.scene_pause_seconds == 0.6


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variables override defaults."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_mocktoken123")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY", "sk-mockkey456")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("LLM_NUM_CTX", "4096")
    monkeypatch.setenv("PIPER_VOICE_PATH", "custom/voice.onnx")
    monkeypatch.setenv("OUTPUT_DIR", "custom_build")
    monkeypatch.setenv("VIDEO_WIDTH", "1280")
    monkeypatch.setenv("VIDEO_HEIGHT", "720")
    monkeypatch.setenv("VIDEO_FPS", "60")
    monkeypatch.setenv("SCENE_PAUSE_SECONDS", "1.5")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.github_token == "ghp_mocktoken123"
    assert settings.llm_provider == "openai"
    assert settings.llm_base_url == "https://api.example.com/v1"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_api_key == "sk-mockkey456"
    assert settings.llm_timeout_seconds == 180
    assert settings.llm_temperature == 0.7
    assert settings.llm_num_ctx == 4096
    assert settings.piper_voice_path == Path("custom/voice.onnx")
    assert settings.output_dir == Path("custom_build")
    assert settings.video_width == 1280
    assert settings.video_height == 720
    assert settings.video_fps == 60
    assert settings.scene_pause_seconds == 1.5


def test_dotenv_loading_without_overwriting_real_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify .env loads variables, strips quotes, and does not overwrite real environment."""
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        """
        # Leading comment
        
        LLM_MODEL=dotenv-model
        GITHUB_TOKEN="dotenv-secret-token"
        VIDEO_WIDTH='720'
        LLM_PROVIDER=openai
        OUTPUT_DIR=dotenv-output
        """,
        encoding="utf-8",
    )

    # Set real environment variable before loading .env
    monkeypatch.setenv("LLM_MODEL", "real-env-model")
    monkeypatch.chdir(tmp_path)

    get_settings.cache_clear()
    settings = get_settings()

    # Pre-existing real environment variable must NOT be overwritten
    assert settings.llm_model == "real-env-model"
    # Variables from .env should be loaded and quotes stripped
    assert settings.github_token == "dotenv-secret-token"
    assert settings.video_width == 720
    assert settings.llm_provider == "openai"
    assert settings.output_dir == Path("dotenv-output")


def test_invalid_number_text_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify non-numeric text for numeric settings raises ConfigError with informative hint."""
    monkeypatch.setenv("VIDEO_WIDTH", "full-hd")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as exc_info:
        get_settings()
    assert "VIDEO_WIDTH" in exc_info.value.message
    assert exc_info.value.hint is not None
    assert "VIDEO_WIDTH" in exc_info.value.hint


def test_invalid_negative_number_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify negative numbers raise ConfigError naming the variable in the hint."""
    monkeypatch.setenv("VIDEO_WIDTH", "-1920")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as exc_info:
        get_settings()
    assert "VIDEO_WIDTH" in exc_info.value.message
    assert "negative" in exc_info.value.message
    assert exc_info.value.hint is not None
    assert "VIDEO_WIDTH" in exc_info.value.hint


def test_invalid_float_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify invalid float text raises ConfigError."""
    monkeypatch.setenv("LLM_TEMPERATURE", "very-hot")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as exc_info:
        get_settings()
    assert "LLM_TEMPERATURE" in exc_info.value.message
    assert exc_info.value.hint is not None
    assert "LLM_TEMPERATURE" in exc_info.value.hint


def test_invalid_provider_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify unknown provider raises ConfigError whose hint names allowed values."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as exc_info:
        get_settings()
    assert "LLM_PROVIDER" in exc_info.value.hint
    assert "ollama" in exc_info.value.hint
    assert "openai" in exc_info.value.hint


def test_secrets_hidden_in_repr() -> None:
    """Verify github_token and llm_api_key are masked in repr."""
    # When secrets are provided
    settings_with_secrets = Settings(
        github_token="ghp_super_secret_github_token",
        llm_api_key="sk_super_secret_openai_key",
    )
    repr_str = repr(settings_with_secrets)
    assert "ghp_super_secret_github_token" not in repr_str
    assert "sk_super_secret_openai_key" not in repr_str
    assert "github_token='***'" in repr_str
    assert "llm_api_key='***'" in repr_str

    # When secrets are None
    settings_no_secrets = Settings(github_token=None, llm_api_key=None)
    repr_no_secrets = repr(settings_no_secrets)
    assert "github_token=None" in repr_no_secrets
    assert "llm_api_key=None" in repr_no_secrets
