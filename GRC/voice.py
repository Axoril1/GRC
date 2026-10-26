"""Voice synthesis stage that converts scene narration text into WAV audio using Piper TTS."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from GRC.config import Settings, get_settings
from GRC.errors import Repo2VideoError, VoiceError
from GRC.models import ScriptData

logger = logging.getLogger(__name__)

DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    "GitHub": "Git Hub",
    "APIs": "A P Is",
    "API": "A P I",
    "CLI": "C L I",
    "LLMs": "L L Ms",
    "LLM": "L L M",
    "TTS": "T T S",
    "FFmpeg": "F F mpeg",
    "JSON": "Jason",
    "YAML": "yammel",
    "SQL": "sequel",
    "npm": "N P M",
    "PyPI": "pie P I",
    "macOS": "mac O S",
    "GPU": "G P U",
    "CPU": "C P U",
    "URL": "U R L",
    "UI": "U I",
    "README": "read me",
    "MIT": "M I T",
    "OSS": "O S S",
    "open-source": "open source",
}


@dataclass(frozen=True)
class SceneAudio:
    """Audio metadata for a synthesized scene slide.

    Attributes:
        scene_id: Numeric identifier of the scene (1-based).
        path: Path to the generated WAV audio file.
        duration_seconds: Measured duration of the audio in seconds.
    """

    scene_id: int
    path: Path
    duration_seconds: float


class TTSEngine(Protocol):
    """Protocol defining the interface for text-to-speech engines."""

    def synthesize(self, text: str, wav_path: Path) -> None:
        """Synthesize speech audio from text and save to a WAV file.

        Args:
            text: Prepared narration text to speak.
            wav_path: Target path for the output WAV file.

        Raises:
            VoiceError: If speech synthesis fails.
        """
        ...

    def check_ready(self) -> None:
        """Verify that the TTS engine, model, and dependencies are ready.

        Raises:
            VoiceError: If the engine is missing, unconfigured, or cannot run.
        """
        ...


class PiperEngine:
    """Text-to-speech engine using local Piper TTS binary or module."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        voice_speed: float | None = None,
        *,
        cmd: list[str] | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the Piper TTS engine.

        Args:
            model_path: Path to the Piper ONNX voice model. Defaults to Settings.piper_voice_path.
            voice_speed: Speech rate multiplier (0.5 to 1.5). Defaults to Settings.voice_speed.
            cmd: Custom command or executable path for Piper. Detected automatically if None.
            settings: Optional Settings instance to read default configuration from.
        """
        resolved_settings = settings if settings is not None else get_settings()

        self.model_path: Path = (
            Path(model_path) if model_path is not None else resolved_settings.piper_voice_path
        )
        self.voice_speed: float = (
            float(voice_speed) if voice_speed is not None else resolved_settings.voice_speed
        )

        if cmd is not None:
            self._cmd: list[str] = [cmd] if isinstance(cmd, str) else list(cmd)
        else:
            piper_binary = shutil.which("piper")
            if piper_binary:
                self._cmd = [piper_binary]
            else:
                self._cmd = [sys.executable, "-m", "piper"]

    def check_ready(self) -> None:
        """Verify that the Piper model, model config, and executable are available.

        Raises:
            VoiceError: If the voice model or companion config is missing, or Piper cannot run.
        """
        if not self.model_path.is_file():
            raise VoiceError(
                f"Piper voice model not found at {self.model_path}.",
                hint=(
                    "Download a voice (for example en_US-lessac-medium) from the Piper "
                    "voices page, put both the .onnx and .onnx.json files in the voices "
                    "folder, or set PIPER_VOICE_PATH. If your Piper version has it, "
                    "python -m piper.download_voices can fetch voices."
                ),
            )

        config_path = Path(f"{self.model_path}.json")
        if not config_path.is_file():
            raise VoiceError(
                f"Piper voice model config not found at {config_path}.",
                hint=(
                    "Download a voice (for example en_US-lessac-medium) from the Piper "
                    "voices page, put both the .onnx and .onnx.json files in the voices "
                    "folder, or set PIPER_VOICE_PATH. If your Piper version has it, "
                    "python -m piper.download_voices can fetch voices."
                ),
            )

        try:
            res = subprocess.run(
                [*self._cmd, "--help"],
                capture_output=True,
                check=False,
                text=True,
                timeout=20,
            )
            if res.returncode != 0:
                raise VoiceError(
                    "Piper is not installed.",
                    hint="Run: pip install piper-tts",
                )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            raise VoiceError(
                "Piper is not installed.",
                hint="Run: pip install piper-tts",
            ) from exc

    def synthesize(self, text: str, wav_path: Path) -> None:
        """Synthesize speech audio from text using Piper and validate the output.

        Args:
            text: Normalized narration text to speak.
            wav_path: Target path for the output WAV file.

        Raises:
            VoiceError: If synthesis fails, times out, or output violates quality constraints.
        """
        wav_path = Path(wav_path)
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        length_scale = f"{1.0 / self.voice_speed:.3f}"
        cmd = [
            *self._cmd,
            "--model",
            str(self.model_path),
            "--output_file",
            str(wav_path),
            "--length_scale",
            length_scale,
            "--sentence_silence",
            "0.25",
        ]
        logger.debug("Executing Piper command: %s", cmd)

        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise VoiceError(
                "Piper synthesis timed out after 120 seconds.",
                hint="Try shorter narration or a smaller/faster voice model.",
            ) from exc
        except FileNotFoundError as exc:
            raise VoiceError(
                "Piper is not installed.",
                hint="Run: pip install piper-tts",
            ) from exc
        except subprocess.SubprocessError as exc:
            raise VoiceError(
                f"Piper synthesis subprocess failed: {exc}",
                hint="Ensure Piper and audio dependencies are properly installed.",
            ) from exc

        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
            last_stderr = stderr_text[-500:] if len(stderr_text) > 500 else stderr_text
            raise VoiceError(
                f"Piper synthesis failed with exit code {result.returncode}: {last_stderr}",
                hint="Check Piper voice model validity and narration text.",
            )

        if not wav_path.is_file():
            raise VoiceError(
                f"Synthesis output file was not created at {wav_path}.",
                hint="Check disk space and write permissions.",
            )

        file_size = wav_path.stat().st_size
        if file_size <= 1024:
            raise VoiceError(
                f"Synthesis output {wav_path} is too small ({file_size} bytes <= 1 KB).",
                hint="Ensure narration text is valid and voice model produces audio.",
            )

        duration = wav_duration(wav_path)
        if duration < 0.5:
            raise VoiceError(
                f"Synthesis output duration {duration:.2f}s is too short (minimum is 0.5s).",
                hint="Provide longer narration text for this scene.",
            )
        if duration > 90.0:
            raise VoiceError(
                f"Synthesis output duration {duration:.2f}s is too long (maximum is 90s).",
                hint="Shorten narration text for this scene.",
            )


def prepare_for_speech(text: str, extra_pronunciations: dict[str, str] | None = None) -> str:
    """Clean and normalize narration text for speech synthesis.

    Applies whole-word pronunciation expansions for technical terms,
    normalizes hyphens/slashes/underscores, strips unsupported characters,
    collapses whitespace, and ensures proper terminal punctuation.

    Args:
        text: Raw narration text.
        extra_pronunciations: Optional dictionary of additional or overriding
            word-to-spoken-form mappings.

    Returns:
        Deterministic, sanitized text ready for TTS synthesis.
    """
    merged = dict(DEFAULT_PRONUNCIATIONS)
    if extra_pronunciations:
        merged.update(extra_pronunciations)

    # 1. Whole-word case-sensitive pronunciation replacements
    if merged:
        sorted_keys = sorted(merged.keys(), key=len, reverse=True)
        pattern = re.compile(
            rf"(?<![A-Za-z0-9])({'|'.join(re.escape(k) for k in sorted_keys)})(?![A-Za-z0-9])"
        )
        text = pattern.sub(lambda m: merged[m.group(0)], text)

    # 2. Replace slashes between letters with a space
    text = re.sub(r"(?<=[a-zA-Z])/(?=[a-zA-Z])", " ", text)

    # 3. Replace underscores and hyphens inside words with a space
    text = re.sub(r"(?<=[a-zA-Z0-9])[_-](?=[a-zA-Z0-9])", " ", text)

    # 4. Remove leftover characters that are not letters, digits, spaces,
    # or allowed punctuation: . , ? ! ; : ' and -
    text = re.sub(r"[^a-zA-Z0-9\s.,?!;:'\-]", "", text)

    # 5. Clean whitespace before punctuation
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)

    # 6. Collapse multiple whitespace characters into single space and strip
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Ensure text ends with a full stop, question mark, or exclamation mark
    if text and not text.endswith((".", "?", "!")):
        text += "."
    elif not text:
        text = "."

    return text


def wav_duration(path: Path) -> float:
    """Measure the duration in seconds of a WAV audio file.

    Attempts measurement using the standard library wave module first. If that
    fails due to an unusual WAV encoding or header error, falls back to ffprobe.

    Args:
        path: Path to the WAV file.

    Returns:
        Duration of the WAV file in seconds.

    Raises:
        VoiceError: If the file does not exist, or neither wave nor ffprobe can read it.
    """
    path = Path(path)
    if not path.is_file():
        raise VoiceError(
            f"WAV audio file not found at {path}.",
            hint="Ensure the TTS engine synthesized the audio file successfully.",
        )

    # Try standard library wave module first
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except (wave.Error, OSError, EOFError) as wave_exc:
        logger.debug(
            "Standard wave module failed to inspect %s: %s; trying ffprobe fallback",
            path,
            wave_exc,
        )

    # Fallback to ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        duration_str = result.stdout.strip()
        duration = float(duration_str)
        if duration >= 0.0:
            return duration
    except Exception as ffprobe_exc:
        raise VoiceError(
            f"Failed to determine duration of WAV file {path}.",
            hint="Ensure the file is a valid WAV audio file and that ffprobe is installed.",
        ) from ffprobe_exc

    raise VoiceError(
        f"Invalid duration obtained for WAV file {path}.",
        hint="Check that the generated audio file is not corrupt.",
    )


def synthesize_all(
    script: ScriptData,
    out_dir: Path,
    engine: TTSEngine | None = None,
    settings: Settings | None = None,
) -> list[SceneAudio]:
    """Synthesize audio narration for each scene in a script and return SceneAudio list.

    Caches results by SHA-256 hash in sidecar .key files to skip re-synthesis when
    text, model, and speed match. Synthesizes atomically to a temporary file before
    renaming to prevent partially written WAV files.

    Args:
        script: Validated ScriptData containing scenes to synthesize.
        out_dir: Directory where generated WAV files and sidecar keys are saved.
        engine: Optional TTSEngine instance. Defaults to PiperEngine.
        settings: Optional Settings instance. Defaults to get_settings().

    Returns:
        List of SceneAudio objects in scene order.

    Raises:
        VoiceError: If readiness check fails or synthesis for any scene fails.
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_engine = engine if engine is not None else PiperEngine(settings=resolved_settings)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load extra pronunciations from pronunciations.json if present
    extra_pronunciations: dict[str, str] = {}
    pron_file = Path.cwd() / "pronunciations.json"
    if pron_file.is_file():
        try:
            content = pron_file.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in data.items()
            ):
                extra_pronunciations = data
                logger.debug(
                    "Loaded %d custom pronunciations from %s",
                    len(extra_pronunciations),
                    pron_file,
                )
            else:
                logger.warning(
                    "Invalid %s: expected a flat JSON object of string-to-string mappings. Ignoring file.",
                    pron_file,
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to parse %s: %s. Ignoring custom pronunciations.",
                pron_file,
                exc,
            )

    # 2. Check readiness once before synthesizing
    resolved_engine.check_ready()

    # Determine voice model name and speed for cache key
    model_name = getattr(resolved_engine, "model_name", None)
    if model_name is None:
        model_path_attr = getattr(resolved_engine, "model_path", None)
        model_name = (
            model_path_attr.name
            if model_path_attr is not None
            else resolved_settings.piper_voice_path.name
        )
    voice_speed = getattr(resolved_engine, "voice_speed", resolved_settings.voice_speed)

    scene_audios: list[SceneAudio] = []

    for scene in script.scenes:
        wav_filename = f"scene_{scene.id:02d}.wav"
        wav_path = out_dir / wav_filename
        key_path = out_dir / f"{wav_filename}.key"

        prepared_text = prepare_for_speech(
            scene.narration, extra_pronunciations=extra_pronunciations
        )
        cache_payload = f"{prepared_text}|{model_name}|{voice_speed:.3f}"
        cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()

        # Check cache
        if wav_path.is_file() and key_path.is_file():
            try:
                stored_key = key_path.read_text(encoding="utf-8").strip()
                if stored_key == cache_key:
                    duration = wav_duration(wav_path)
                    if 0.5 <= duration <= 90.0:
                        logger.info(
                            "Reusing cached audio for scene %d from %s (%.2fs)",
                            scene.id,
                            wav_path,
                            duration,
                        )
                        scene_audios.append(
                            SceneAudio(
                                scene_id=scene.id,
                                path=wav_path,
                                duration_seconds=duration,
                            )
                        )
                        continue
            except (OSError, VoiceError) as exc:
                logger.debug(
                    "Cache validation failed for scene %d (%s): %s; re-synthesizing.",
                    scene.id,
                    wav_path,
                    exc,
                )

        # Synthesize atomically via a temporary file
        temp_path = out_dir / f".scene_{scene.id:02d}.wav.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        try:
            try:
                resolved_engine.synthesize(prepared_text, temp_path)
                os.replace(temp_path, wav_path)
                key_path.write_text(cache_key, encoding="utf-8")
                duration = wav_duration(wav_path)
                scene_audios.append(
                    SceneAudio(
                        scene_id=scene.id,
                        path=wav_path,
                        duration_seconds=duration,
                    )
                )
                logger.info(
                    "Synthesized audio for scene %d -> %s (%.2fs)",
                    scene.id,
                    wav_path,
                    duration,
                )
            except Exception as exc:
                if isinstance(exc, VoiceError):
                    err_msg = exc.message
                    if (
                        f"scene {scene.id}" not in err_msg.lower()
                        and f"scene_{scene.id:02d}" not in err_msg.lower()
                    ):
                        raise VoiceError(
                            f"Failed to synthesize audio for scene {scene.id}: {err_msg}",
                            hint=exc.hint,
                        ) from exc
                    raise
                raise VoiceError(
                    f"Failed to synthesize audio for scene {scene.id}: {exc}",
                    hint="Ensure Piper TTS dependencies and voice model are functioning.",
                ) from exc
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    total_duration = sum(sa.duration_seconds for sa in scene_audios)
    logger.info(
        "Total audio duration for %d scenes: %.2f seconds",
        len(scene_audios),
        total_duration,
    )
    if total_duration < 30.0 or total_duration > 100.0:
        logger.warning(
            "Total audio duration %.2fs is outside recommended video length (30-100s).",
            total_duration,
        )

    return scene_audios


def main() -> None:
    """CLI entrypoint for checking voice readiness or synthesizing a script."""
    parser = argparse.ArgumentParser(
        description="GRC voice stage: synthesize narration WAV audio using Piper TTS."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether Piper TTS and the configured voice model are ready.",
    )
    parser.add_argument(
        "script_path",
        nargs="?",
        type=Path,
        help="Path to scenes.json script file to synthesize.",
    )
    args = parser.parse_args()

    settings = get_settings()
    engine = PiperEngine(settings=settings)

    if args.check:
        try:
            engine.check_ready()
            print("Piper TTS and voice model are ready.")
            sys.exit(0)
        except VoiceError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.script_path:
        script_file = Path(args.script_path)
        if not script_file.is_file():
            print(f"Error: File not found: {script_file}", file=sys.stderr)
            sys.exit(1)
        try:
            script = ScriptData.load(script_file)
            out_dir = script_file.resolve().parent / "audio"
            audios = synthesize_all(script, out_dir, engine=engine, settings=settings)
            for sa in audios:
                print(f"Scene {sa.scene_id}: {sa.duration_seconds:.2f}s ({sa.path})")
            total = sum(sa.duration_seconds for sa in audios)
            print(f"Total duration: {total:.2f}s across {len(audios)} scenes.")
            sys.exit(0)
        except Repo2VideoError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
