"""Tests for custom exception classes in repo2video."""

import pytest

from repo2video.errors import (
    AnalyzerError,
    ConfigError,
    LLMError,
    MediaError,
    RenderError,
    Repo2VideoError,
    SceneError,
    ScriptError,
    VoiceError,
)


def test_repo2video_error_without_hint() -> None:
    """Verify error string representation without a hint."""
    err = Repo2VideoError("Something failed")
    assert str(err) == "Something failed"
    assert err.message == "Something failed"
    assert err.hint is None


def test_repo2video_error_with_hint() -> None:
    """Verify error string representation with a hint."""
    err = Repo2VideoError("Something failed", hint="Try turning it off and on again.")
    assert str(err) == "Something failed\nHint: Try turning it off and on again."
    assert err.message == "Something failed"
    assert err.hint == "Try turning it off and on again."


@pytest.mark.parametrize(
    "exc_cls",
    [
        ConfigError,
        AnalyzerError,
        LLMError,
        ScriptError,
        SceneError,
        VoiceError,
        MediaError,
        RenderError,
    ],
)
def test_subclass_inheritance(exc_cls: type[Repo2VideoError]) -> None:
    """Verify all custom exceptions inherit from Repo2VideoError and format correctly."""
    err = exc_cls("Failure message", hint="Resolution hint")
    assert isinstance(err, Repo2VideoError)
    assert str(err) == "Failure message\nHint: Resolution hint"
