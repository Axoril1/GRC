"""Pytest configuration and global fixtures for isolated testing."""

from collections.abc import Generator
from pathlib import Path

import pytest

import repo2video.config
from repo2video.config import get_settings

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
    "VOICE_SPEED",
]

REPO_ROOT_ENV = (Path(__file__).parent.parent / ".env").resolve()
ORIGINAL_LOAD_DOTENV = repo2video.config.load_dotenv


def _test_load_dotenv(dotenv_path: Path | None = None) -> None:
    target = (dotenv_path if dotenv_path is not None else Path.cwd() / ".env").resolve()
    if target == REPO_ROOT_ENV:
        return
    ORIGINAL_LOAD_DOTENV(dotenv_path)


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Prevent tests from reading local root .env while allowing custom test .env loading."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("repo2video.config.load_dotenv", _test_load_dotenv)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
