"""Application configuration and environment settings for repo2video."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from repo2video.errors import ConfigError


def load_dotenv(dotenv_path: Path | None = None) -> None:
    """Load key-value pairs from a .env file into os.environ.

    Ignores blank lines and comments starting with '#'. Strips matching
    surrounding single or double quotes. Does not overwrite variables already
    set in os.environ.

    Args:
        dotenv_path: Path to the .env file. Defaults to .env in the current directory.
    """
    path = dotenv_path if dotenv_path is not None else Path.cwd() / ".env"
    if not path.is_file():
        return

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ) and len(value) >= 2:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_int(var_name: str, value_str: str | None, default: int) -> int:
    """Parse and validate an integer configuration variable.

    Args:
        var_name: Name of the environment variable.
        value_str: Raw string value from the environment.
        default: Default value used when value_str is None.

    Returns:
        The validated integer.

    Raises:
        ConfigError: If value_str is not a valid integer or is negative.
    """
    if value_str is None:
        return default
    try:
        val = int(value_str)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid integer for {var_name}: {value_str!r}.",
            hint=f"Set {var_name} to a non-negative integer.",
        ) from exc
    if val < 0:
        raise ConfigError(
            f"{var_name} cannot be negative, got {val}.",
            hint=f"Set {var_name} to a non-negative integer.",
        )
    return val


def _parse_float(var_name: str, value_str: str | None, default: float) -> float:
    """Parse and validate a float configuration variable.

    Args:
        var_name: Name of the environment variable.
        value_str: Raw string value from the environment.
        default: Default value used when value_str is None.

    Returns:
        The validated float.

    Raises:
        ConfigError: If value_str is not a valid float or is negative.
    """
    if value_str is None:
        return default
    try:
        val = float(value_str)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid number for {var_name}: {value_str!r}.",
            hint=f"Set {var_name} to a non-negative number.",
        ) from exc
    if val < 0.0:
        raise ConfigError(
            f"{var_name} cannot be negative, got {val}.",
            hint=f"Set {var_name} to a non-negative number.",
        )
    return val


def _parse_float_range(
    var_name: str,
    value_str: str | None,
    default: float,
    min_val: float = 0.5,
    max_val: float = 1.5,
) -> float:
    """Parse and validate a float configuration variable within a bounded range.

    Args:
        var_name: Name of the environment variable.
        value_str: Raw string value from the environment.
        default: Default value used when value_str is None.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        The validated float within [min_val, max_val].

    Raises:
        ConfigError: If value_str is not a valid float or outside the range.
    """
    if value_str is None:
        return default
    try:
        val = float(value_str)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid number for {var_name}: {value_str!r}.",
            hint=f"Set {var_name} to a number between {min_val} and {max_val}.",
        ) from exc
    if not (min_val <= val <= max_val):
        raise ConfigError(
            f"{var_name} must be between {min_val} and {max_val}, got {val}.",
            hint=f"Set {var_name} to a number between {min_val} and {max_val}.",
        )
    return val


def _parse_anim_fps(var_name: str, value_str: str | None, default: int = 24) -> int:
    """Parse and validate ANIM_FPS configuration variable (between 10 and 30).

    Args:
        var_name: Name of the environment variable.
        value_str: Raw string value from the environment.
        default: Default value used when value_str is None.

    Returns:
        The validated integer between 10 and 30.

    Raises:
        ConfigError: If value_str is not an integer or outside [10, 30].
    """
    if value_str is None:
        return default
    try:
        val = int(value_str)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid integer for {var_name}: {value_str!r}.",
            hint=f"Set {var_name} to an integer between 10 and 30.",
        ) from exc
    if not (10 <= val <= 30):
        raise ConfigError(
            f"{var_name} must be between 10 and 30, got {val}.",
            hint=f"Set {var_name} to an integer between 10 and 30.",
        )
    return val


def _parse_anim_quality(var_name: str, value_str: str | None, default: str = "normal") -> str:
    """Parse and validate ANIM_QUALITY configuration variable ('normal' or 'fast').

    Args:
        var_name: Name of the environment variable.
        value_str: Raw string value from the environment.
        default: Default quality used when value_str is None.

    Returns:
        The validated quality string ('normal' or 'fast').

    Raises:
        ConfigError: If the quality is not 'normal' or 'fast'.
    """
    if value_str is None:
        return default
    val = value_str.strip().lower()
    if val not in ("normal", "fast"):
        raise ConfigError(
            f"{var_name} must be 'normal' or 'fast', got {value_str!r}.",
            hint=f"Set {var_name} to 'normal' or 'fast'.",
        )
    return val


def _parse_provider(var_name: str, value_str: str | None, default: str) -> str:
    """Parse and validate the LLM provider configuration variable.

    Args:
        var_name: Name of the environment variable.
        value_str: Raw string value from the environment.
        default: Default provider used when value_str is None.

    Returns:
        The validated provider string ('ollama' or 'openai').

    Raises:
        ConfigError: If the provider is not 'ollama' or 'openai'.
    """
    if value_str is None:
        return default
    val = value_str.strip().lower()
    allowed = ("ollama", "openai")
    if val not in allowed:
        raise ConfigError(
            f"Unknown LLM provider: {value_str!r}.",
            hint=f"Set {var_name} to one of: {', '.join(allowed)}.",
        )
    return val


@dataclass(frozen=True)
class Settings:
    """Application settings for repo2video.

    Attributes:
        github_token: GitHub personal access token (optional).
        llm_provider: LLM provider service ('ollama' or 'openai').
        llm_base_url: Base URL for LLM API calls.
        llm_model: Model identifier for LLM generation.
        llm_api_key: API key for the LLM provider (optional).
        llm_timeout_seconds: Request timeout in seconds for LLM calls.
        llm_temperature: Sampling temperature for LLM generation.
        llm_num_ctx: Context window size in tokens for LLM generation.
        piper_voice_path: File path to the Piper TTS ONNX model.
        output_dir: Directory where generated outputs will be stored.
        video_width: Output video width in pixels.
        video_height: Output video height in pixels.
        video_fps: Output video frame rate.
        scene_pause_seconds: Duration in seconds to pause between scenes.
        voice_speed: Speech rate multiplier for TTS audio (between 0.5 and 1.5).
        anim_fps: Frame rate for animation capture (between 10 and 30, default 24).
        anim_quality: Quality preset for animation ('normal' or 'fast', default 'normal').
    """

    github_token: str | None = None
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    llm_api_key: str | None = None
    llm_timeout_seconds: int = 300
    llm_temperature: float = 0.4
    llm_num_ctx: int = 8192
    piper_voice_path: Path = Path("voices/en_US-lessac-medium.onnx")
    output_dir: Path = Path("output")
    video_width: int = 1920
    video_height: int = 1080
    video_fps: int = 30
    scene_pause_seconds: float = 0.6
    voice_speed: float = 1.0
    anim_fps: int = 24
    anim_quality: str = "normal"

    def __post_init__(self) -> None:
        """Validate settings constraints upon initialization.

        Raises:
            ConfigError: If any configuration value is invalid.
        """
        if self.llm_provider not in ("ollama", "openai"):
            raise ConfigError(
                f"Unknown LLM provider: {self.llm_provider!r}.",
                hint="Set LLM_PROVIDER to one of: ollama, openai.",
            )
        if not isinstance(self.llm_timeout_seconds, int) or isinstance(
            self.llm_timeout_seconds, bool
        ):
            raise ConfigError(
                f"Invalid integer for LLM_TIMEOUT_SECONDS: {self.llm_timeout_seconds!r}.",
                hint="Set LLM_TIMEOUT_SECONDS to a non-negative integer.",
            )
        if self.llm_timeout_seconds < 0:
            raise ConfigError(
                f"LLM_TIMEOUT_SECONDS cannot be negative, got {self.llm_timeout_seconds}.",
                hint="Set LLM_TIMEOUT_SECONDS to a non-negative integer.",
            )
        if not isinstance(self.llm_temperature, (int, float)) or isinstance(
            self.llm_temperature, bool
        ):
            raise ConfigError(
                f"Invalid number for LLM_TEMPERATURE: {self.llm_temperature!r}.",
                hint="Set LLM_TEMPERATURE to a non-negative number.",
            )
        if self.llm_temperature < 0.0:
            raise ConfigError(
                f"LLM_TEMPERATURE cannot be negative, got {self.llm_temperature}.",
                hint="Set LLM_TEMPERATURE to a non-negative number.",
            )
        if not isinstance(self.llm_num_ctx, int) or isinstance(self.llm_num_ctx, bool):
            raise ConfigError(
                f"Invalid integer for LLM_NUM_CTX: {self.llm_num_ctx!r}.",
                hint="Set LLM_NUM_CTX to a non-negative integer.",
            )
        if self.llm_num_ctx < 0:
            raise ConfigError(
                f"LLM_NUM_CTX cannot be negative, got {self.llm_num_ctx}.",
                hint="Set LLM_NUM_CTX to a non-negative integer.",
            )
        if not isinstance(self.video_width, int) or isinstance(self.video_width, bool):
            raise ConfigError(
                f"Invalid integer for VIDEO_WIDTH: {self.video_width!r}.",
                hint="Set VIDEO_WIDTH to a non-negative integer.",
            )
        if self.video_width < 0:
            raise ConfigError(
                f"VIDEO_WIDTH cannot be negative, got {self.video_width}.",
                hint="Set VIDEO_WIDTH to a non-negative integer.",
            )
        if not isinstance(self.video_height, int) or isinstance(self.video_height, bool):
            raise ConfigError(
                f"Invalid integer for VIDEO_HEIGHT: {self.video_height!r}.",
                hint="Set VIDEO_HEIGHT to a non-negative integer.",
            )
        if self.video_height < 0:
            raise ConfigError(
                f"VIDEO_HEIGHT cannot be negative, got {self.video_height}.",
                hint="Set VIDEO_HEIGHT to a non-negative integer.",
            )
        if not isinstance(self.video_fps, int) or isinstance(self.video_fps, bool):
            raise ConfigError(
                f"Invalid integer for VIDEO_FPS: {self.video_fps!r}.",
                hint="Set VIDEO_FPS to a non-negative integer.",
            )
        if self.video_fps < 0:
            raise ConfigError(
                f"VIDEO_FPS cannot be negative, got {self.video_fps}.",
                hint="Set VIDEO_FPS to a non-negative integer.",
            )
        if not isinstance(self.scene_pause_seconds, (int, float)) or isinstance(
            self.scene_pause_seconds, bool
        ):
            raise ConfigError(
                f"Invalid number for SCENE_PAUSE_SECONDS: {self.scene_pause_seconds!r}.",
                hint="Set SCENE_PAUSE_SECONDS to a non-negative number.",
            )
        if self.scene_pause_seconds < 0.0:
            raise ConfigError(
                f"SCENE_PAUSE_SECONDS cannot be negative, got {self.scene_pause_seconds}.",
                hint="Set SCENE_PAUSE_SECONDS to a non-negative number.",
            )
        if not isinstance(self.voice_speed, (int, float)) or isinstance(self.voice_speed, bool):
            raise ConfigError(
                f"Invalid number for VOICE_SPEED: {self.voice_speed!r}.",
                hint="Set VOICE_SPEED to a number between 0.5 and 1.5.",
            )
        if not (0.5 <= float(self.voice_speed) <= 1.5):
            raise ConfigError(
                f"VOICE_SPEED must be between 0.5 and 1.5, got {self.voice_speed}.",
                hint="Set VOICE_SPEED to a number between 0.5 and 1.5.",
            )
        if not isinstance(self.anim_fps, int) or isinstance(self.anim_fps, bool):
            raise ConfigError(
                f"Invalid integer for ANIM_FPS: {self.anim_fps!r}.",
                hint="Set ANIM_FPS to an integer between 10 and 30.",
            )
        if not (10 <= self.anim_fps <= 30):
            raise ConfigError(
                f"ANIM_FPS must be between 10 and 30, got {self.anim_fps}.",
                hint="Set ANIM_FPS to an integer between 10 and 30.",
            )
        if not isinstance(self.anim_quality, str):
            raise ConfigError(
                f"Invalid value for ANIM_QUALITY: {self.anim_quality!r}.",
                hint="Set ANIM_QUALITY to 'normal' or 'fast'.",
            )
        if self.anim_quality.strip().lower() not in ("normal", "fast"):
            raise ConfigError(
                f"ANIM_QUALITY must be 'normal' or 'fast', got {self.anim_quality!r}.",
                hint="Set ANIM_QUALITY to 'normal' or 'fast'.",
            )

    def __repr__(self) -> str:
        """Return a string representation of Settings with secrets masked.

        Returns:
            String representation with github_token and llm_api_key masked as '***' when set.
        """
        token_repr = "'***'" if self.github_token is not None else "None"
        api_key_repr = "'***'" if self.llm_api_key is not None else "None"
        return (
            f"Settings("
            f"github_token={token_repr}, "
            f"llm_provider={self.llm_provider!r}, "
            f"llm_base_url={self.llm_base_url!r}, "
            f"llm_model={self.llm_model!r}, "
            f"llm_api_key={api_key_repr}, "
            f"llm_timeout_seconds={self.llm_timeout_seconds!r}, "
            f"llm_temperature={self.llm_temperature!r}, "
            f"llm_num_ctx={self.llm_num_ctx!r}, "
            f"piper_voice_path={self.piper_voice_path!r}, "
            f"output_dir={self.output_dir!r}, "
            f"video_width={self.video_width!r}, "
            f"video_height={self.video_height!r}, "
            f"video_fps={self.video_fps!r}, "
            f"scene_pause_seconds={self.scene_pause_seconds!r}, "
            f"voice_speed={self.voice_speed!r}, "
            f"anim_fps={self.anim_fps!r}, "
            f"anim_quality={self.anim_quality!r}"
            f")"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load, validate, and return cached application settings.

    Reads configuration from .env in the current directory if present without
    overwriting existing environment variables, reads environment variables,
    validates values, and returns a frozen Settings instance.

    Returns:
        The validated Settings instance.

    Raises:
        ConfigError: If any configuration variable is invalid.
    """
    load_dotenv()

    github_token = os.environ.get("GITHUB_TOKEN") or None
    llm_provider = _parse_provider("LLM_PROVIDER", os.environ.get("LLM_PROVIDER"), "ollama")
    llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
    llm_model = os.environ.get("LLM_MODEL", "qwen2.5:7b")
    llm_api_key = os.environ.get("LLM_API_KEY") or None
    llm_timeout_seconds = _parse_int(
        "LLM_TIMEOUT_SECONDS", os.environ.get("LLM_TIMEOUT_SECONDS"), 300
    )
    llm_temperature = _parse_float("LLM_TEMPERATURE", os.environ.get("LLM_TEMPERATURE"), 0.4)
    llm_num_ctx = _parse_int("LLM_NUM_CTX", os.environ.get("LLM_NUM_CTX"), 8192)
    piper_voice_path = Path(os.environ.get("PIPER_VOICE_PATH", "voices/en_US-lessac-medium.onnx"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "output"))
    video_width = _parse_int("VIDEO_WIDTH", os.environ.get("VIDEO_WIDTH"), 1920)
    video_height = _parse_int("VIDEO_HEIGHT", os.environ.get("VIDEO_HEIGHT"), 1080)
    video_fps = _parse_int("VIDEO_FPS", os.environ.get("VIDEO_FPS"), 30)
    scene_pause_seconds = _parse_float(
        "SCENE_PAUSE_SECONDS", os.environ.get("SCENE_PAUSE_SECONDS"), 0.6
    )
    voice_speed = _parse_float_range("VOICE_SPEED", os.environ.get("VOICE_SPEED"), 1.0, 0.5, 1.5)
    anim_fps = _parse_anim_fps("ANIM_FPS", os.environ.get("ANIM_FPS"), 24)
    anim_quality = _parse_anim_quality("ANIM_QUALITY", os.environ.get("ANIM_QUALITY"), "normal")

    return Settings(
        github_token=github_token,
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_temperature=llm_temperature,
        llm_num_ctx=llm_num_ctx,
        piper_voice_path=piper_voice_path,
        output_dir=output_dir,
        video_width=video_width,
        video_height=video_height,
        video_fps=video_fps,
        scene_pause_seconds=scene_pause_seconds,
        voice_speed=voice_speed,
        anim_fps=anim_fps,
        anim_quality=anim_quality,
    )
