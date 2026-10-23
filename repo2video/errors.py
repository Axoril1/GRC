"""Exception hierarchy for the repo2video application."""

from __future__ import annotations


class Repo2VideoError(Exception):
    """Base exception for all repo2video errors.

    Attributes:
        message: Explanation of what went wrong.
        hint: Optional short sentence suggesting how to fix the issue.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        """Initialize the error with a message and an optional hint.

        Args:
            message: Explanation of what went wrong.
            hint: Optional short sentence suggesting how to resolve the error.
        """
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        """Return a formatted string representation of the error.

        Returns:
            The error message, followed by a newline and 'Hint: ...' if a hint exists.
        """
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ConfigError(Repo2VideoError):
    """Raised when configuration values are invalid or missing.

    Attributes:
        message: Explanation of the configuration problem.
        hint: Optional short sentence suggesting how to fix the configuration.
    """


class AnalyzerError(Repo2VideoError):
    """Raised when repository analysis fails.

    Attributes:
        message: Explanation of the analysis failure.
        hint: Optional short sentence suggesting how to resolve the issue.
    """


class LLMError(Repo2VideoError):
    """Raised when LLM interaction or generation fails.

    Attributes:
        message: Explanation of the LLM failure.
        hint: Optional short sentence suggesting how to resolve the issue.
    """


class ScriptError(Repo2VideoError):
    """Raised when script generation or validation fails.

    Attributes:
        message: Explanation of the script failure.
        hint: Optional short sentence suggesting how to resolve the issue.
    """


class SceneError(Repo2VideoError):
    """Raised when scene building or slide rendering fails.

    Attributes:
        message: Explanation of the scene building failure.
        hint: Optional short sentence suggesting how to resolve the issue.
    """


class VoiceError(Repo2VideoError):
    """Raised when voice narration synthesis fails.

    Attributes:
        message: Explanation of the voice synthesis failure.
        hint: Optional short sentence suggesting how to resolve the issue.
    """


class RenderError(Repo2VideoError):
    """Raised when video rendering or composition fails.

    Attributes:
        message: Explanation of the rendering failure.
        hint: Optional short sentence suggesting how to resolve the issue.
    """
