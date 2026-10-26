"""Unit tests for the repo2video voice narration synthesis stage."""

from __future__ import annotations

import json
import logging
import subprocess
import wave
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from GRC.errors import VoiceError
from GRC.models import Scene, ScriptData
from GRC.voice import (
    DEFAULT_PRONUNCIATIONS,
    PiperEngine,
    SceneAudio,
    main,
    prepare_for_speech,
    synthesize_all,
    wav_duration,
)


def create_dummy_wav(
    path: Path,
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    num_channels: int = 1,
    sampwidth: int = 2,
) -> Path:
    """Create a real PCM WAV file using the standard library wave module.

    Args:
        path: Target file path.
        duration_seconds: Audio duration in seconds.
        sample_rate: Audio sampling rate in Hz.
        num_channels: Number of audio channels (1 = mono).
        sampwidth: Bytes per sample (2 = 16-bit).

    Returns:
        The created file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * duration_seconds)
    data = b"\x00" * (num_frames * num_channels * sampwidth)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return path


# -----------------------------------------------------------------------------
# 1. prepare_for_speech tests
# -----------------------------------------------------------------------------

PREPARE_TABLE = [
    ("GitHub repository", "Git Hub repository."),
    ("REST API and GraphQL APIs", "REST A P I and GraphQL A P Is."),
    ("Modern CLI tool", "Modern C L I tool."),
    ("Local LLM and large LLMs", "Local L L M and large L L Ms."),
    ("Piper TTS synthesis", "Piper T T S synthesis."),
    ("Rendered with FFmpeg", "Rendered with F F mpeg."),
    ("Parses JSON and YAML files", "Parses Jason and yammel files."),
    ("PostgreSQL and SQL queries", "PostgreSQL and sequel queries."),
    ("Install via npm or PyPI", "Install via N P M or pie P I."),
    ("Works on macOS and Linux", "Works on mac O S and Linux."),
    ("Accelerated by GPU or CPU", "Accelerated by G P U or C P U."),
    ("Enter the repo URL", "Enter the repo U R L."),
    ("Clean user UI", "Clean user U I."),
    ("Check the project README", "Check the project read me."),
    ("Released under MIT and OSS licenses", "Released under M I T and O S S licenses."),
    ("Fully open-source software", "Fully open source software."),
    ("client/server architecture and/or bridges", "client server architecture and or bridges."),
    ("Supports snake_case_identifier", "Supports snake case identifier."),
    ("High-performance cross-platform build", "High performance cross platform build."),
    (
        "Clean code & comments with $dollars and #hashtags!",
        "Clean code comments with dollars and hashtags!",
    ),
    ("Is this question preserved?", "Is this question preserved?"),
    ("Excited announcement!", "Excited announcement!"),
    ("Trailing spaces without punct   ", "Trailing spaces without punct."),
    ("Colon: semicolon; comma, period.", "Colon: semicolon; comma, period."),
]


def test_scene_audio_dataclass(tmp_path: Path) -> None:
    """Verify SceneAudio is a frozen dataclass with expected attributes."""
    audio_path = tmp_path / "scene_01.wav"
    sa = SceneAudio(scene_id=1, path=audio_path, duration_seconds=4.5)
    assert sa.scene_id == 1
    assert sa.path == audio_path
    assert sa.duration_seconds == 4.5

    # Verify frozen
    with pytest.raises(FrozenInstanceError):
        sa.scene_id = 2  # type: ignore[misc]


def test_default_pronunciations_built_in() -> None:
    """Verify DEFAULT_PRONUNCIATIONS dictionary contains required mappings."""
    assert DEFAULT_PRONUNCIATIONS["GitHub"] == "Git Hub"
    assert DEFAULT_PRONUNCIATIONS["API"] == "A P I"
    assert DEFAULT_PRONUNCIATIONS["APIs"] == "A P Is"
    assert DEFAULT_PRONUNCIATIONS["CLI"] == "C L I"
    assert DEFAULT_PRONUNCIATIONS["LLM"] == "L L M"
    assert DEFAULT_PRONUNCIATIONS["open-source"] == "open source"


@pytest.mark.parametrize(("raw_input", "expected_output"), PREPARE_TABLE)
def test_prepare_for_speech_table(raw_input: str, expected_output: str) -> None:
    """Verify prepare_for_speech normalizes a wide variety of inputs deterministically."""
    result = prepare_for_speech(raw_input)
    assert result == expected_output


def test_prepare_for_speech_custom_pronunciations() -> None:
    """Verify extra_pronunciations merges over built-in mappings."""
    text = "Deploying Docker container to Kubernetes with PyPI package"
    custom = {
        "Docker": "Dock er",
        "Kubernetes": "Kube",
        "PyPI": "pie pie",  # Overrides built-in "pie P I"
    }
    result = prepare_for_speech(text, extra_pronunciations=custom)
    assert result == "Deploying Dock er container to Kube with pie pie package."


def test_prepare_for_speech_empty_string() -> None:
    """Verify empty string returns a single period."""
    assert prepare_for_speech("") == "."
    assert prepare_for_speech("   ") == "."


# -----------------------------------------------------------------------------
# 2. wav_duration tests
# -----------------------------------------------------------------------------


def test_wav_duration_generated_file(tmp_path: Path) -> None:
    """Verify wav_duration measures duration accurately on a generated 2.5-second file."""
    wav_file = tmp_path / "test_2_5s.wav"
    create_dummy_wav(wav_file, duration_seconds=2.5)

    measured = wav_duration(wav_file)
    assert measured == pytest.approx(2.5, abs=0.01)


def test_wav_duration_missing_file(tmp_path: Path) -> None:
    """Verify wav_duration raises VoiceError when audio file does not exist."""
    missing = tmp_path / "non_existent.wav"
    with pytest.raises(VoiceError) as exc_info:
        wav_duration(missing)
    assert "not found" in exc_info.value.message


def test_wav_duration_wave_fails_falls_back_to_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify wav_duration falls back to ffprobe when wave module fails."""
    wav_file = tmp_path / "custom_format.wav"
    wav_file.write_bytes(b"RIFF dummy unusual wave header")

    # Mock wave.open to fail
    def fake_wave_open(*args: Any, **kwargs: Any) -> Any:
        raise wave.Error("Unrecognized WAV chunk format")

    monkeypatch.setattr(wave, "open", fake_wave_open)

    # Mock ffprobe subprocess call
    def fake_subprocess_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="3.14159\n", stderr="")
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    duration = wav_duration(wav_file)
    assert duration == pytest.approx(3.14, abs=0.01)


def test_wav_duration_both_wave_and_ffprobe_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify wav_duration raises VoiceError when both wave and ffprobe fail."""
    wav_file = tmp_path / "corrupt.wav"
    wav_file.write_bytes(b"garbage")

    def fake_wave_open(*args: Any, **kwargs: Any) -> Any:
        raise wave.Error("Corrupt data")

    def fake_subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.SubprocessError("ffprobe execution failed")

    monkeypatch.setattr(wave, "open", fake_wave_open)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    with pytest.raises(VoiceError) as exc_info:
        wav_duration(wav_file)
    assert "Failed to determine duration" in exc_info.value.message


# -----------------------------------------------------------------------------
# 3. PiperEngine command construction & readiness tests
# -----------------------------------------------------------------------------


def test_piper_engine_command_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify command argument list, length_scale formatting, and stdin handling."""
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    out_wav = tmp_path / "out.wav"

    commands_executed: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        commands_executed.append({"cmd": cmd, "kwargs": kwargs})
        # Create output file larger than 1KB with 2.0s duration
        create_dummy_wav(out_wav, duration_seconds=2.0)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Test speed 0.8 -> length_scale = 1 / 0.8 = 1.250
    engine_08 = PiperEngine(model_path=model_path, voice_speed=0.8, cmd=["mock_piper"])
    engine_08.synthesize("Narration text for 0.8 speed.", out_wav)

    assert len(commands_executed) == 1
    call_08 = commands_executed[0]
    assert call_08["cmd"] == [
        "mock_piper",
        "--model",
        str(model_path),
        "--output_file",
        str(out_wav),
        "--length_scale",
        "1.250",
        "--sentence_silence",
        "0.25",
    ]
    assert call_08["kwargs"]["input"] == b"Narration text for 0.8 speed."
    assert call_08["kwargs"]["timeout"] == 120

    # Test speed 1.0 -> length_scale = 1.000
    engine_10 = PiperEngine(model_path=model_path, voice_speed=1.0, cmd=["mock_piper"])
    engine_10.synthesize("Narration text for 1.0 speed.", out_wav)
    assert commands_executed[1]["cmd"][6] == "1.000"

    # Test speed 1.25 -> length_scale = 1 / 1.25 = 0.800
    engine_125 = PiperEngine(model_path=model_path, voice_speed=1.25, cmd=["mock_piper"])
    engine_125.synthesize("Narration text for 1.25 speed.", out_wav)
    assert commands_executed[2]["cmd"][6] == "0.800"


def test_piper_engine_check_ready_missing_model(tmp_path: Path) -> None:
    """Verify check_ready raises VoiceError when the .onnx model file is missing."""
    engine = PiperEngine(model_path=tmp_path / "missing.onnx")
    with pytest.raises(VoiceError) as exc_info:
        engine.check_ready()
    assert "Piper voice model not found" in exc_info.value.message
    assert "Download a voice" in (exc_info.value.hint or "")


def test_piper_engine_check_ready_missing_config(tmp_path: Path) -> None:
    """Verify check_ready raises VoiceError when companion .onnx.json is missing."""
    model = tmp_path / "voice.onnx"
    model.touch()
    # Notice: companion voice.onnx.json is NOT created

    engine = PiperEngine(model_path=model)
    with pytest.raises(VoiceError) as exc_info:
        engine.check_ready()
    assert "Piper voice model config not found" in exc_info.value.message
    assert "Download a voice" in (exc_info.value.hint or "")


def test_piper_engine_check_ready_missing_piper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify check_ready raises VoiceError when piper command fails to execute."""
    model = tmp_path / "voice.onnx"
    model.touch()
    Path(f"{model}.json").touch()

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Emulate piper binary missing or non-zero exit code
        return subprocess.CompletedProcess(
            args[0], returncode=1, stdout="", stderr="command not found"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine = PiperEngine(model_path=model, cmd=["non_existent_piper"])
    with pytest.raises(VoiceError) as exc_info:
        engine.check_ready()
    assert "Piper is not installed" in exc_info.value.message
    assert "pip install piper-tts" in (exc_info.value.hint or "")


def test_piper_engine_check_ready_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify check_ready succeeds when model, companion config, and piper binary exist."""
    model = tmp_path / "voice.onnx"
    model.touch()
    Path(f"{model}.json").touch()

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args[0], returncode=0, stdout="Piper TTS 1.2.0", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine = PiperEngine(model_path=model, cmd=["mock_piper"])
    engine.check_ready()  # Should not raise


# -----------------------------------------------------------------------------
# 4. PiperEngine synthesis errors and output validation
# -----------------------------------------------------------------------------


def test_piper_engine_synthesis_failure_with_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify non-zero returncode raises VoiceError with the last 500 chars of stderr."""
    model = tmp_path / "voice.onnx"
    model.touch()
    out_wav = tmp_path / "out.wav"

    err_text = "Detailed onnx error: " + ("x" * 600) + "end_of_error"

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args[0], returncode=2, stdout=b"", stderr=err_text.encode("utf-8")
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine = PiperEngine(model_path=model, cmd=["mock_piper"])
    with pytest.raises(VoiceError) as exc_info:
        engine.synthesize("Some text.", out_wav)

    assert "Piper synthesis failed with exit code 2" in exc_info.value.message
    assert "end_of_error" in exc_info.value.message
    # Check that error is capped at roughly 500 chars
    assert len(exc_info.value.message) < 700


def test_piper_engine_output_validation_file_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify VoiceError raised when Piper exits 0 but creates no output file."""
    model = tmp_path / "voice.onnx"
    model.touch()
    out_wav = tmp_path / "out.wav"

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        # Does not create out_wav
        return subprocess.CompletedProcess(args[0], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine = PiperEngine(model_path=model, cmd=["mock_piper"])
    with pytest.raises(VoiceError) as exc_info:
        engine.synthesize("Some text.", out_wav)
    assert "was not created" in exc_info.value.message


def test_piper_engine_output_validation_file_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify VoiceError raised when output file is <= 1 KB (1024 bytes)."""
    model = tmp_path / "voice.onnx"
    model.touch()
    out_wav = tmp_path / "out.wav"

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        out_wav.write_bytes(b"\x00" * 500)  # 500 bytes <= 1024
        return subprocess.CompletedProcess(args[0], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine = PiperEngine(model_path=model, cmd=["mock_piper"])
    with pytest.raises(VoiceError) as exc_info:
        engine.synthesize("Some text.", out_wav)
    assert "too small" in exc_info.value.message


def test_piper_engine_output_validation_duration_too_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify VoiceError raised when measured duration is < 0.5 seconds."""
    model = tmp_path / "voice.onnx"
    model.touch()
    out_wav = tmp_path / "out.wav"

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        # 0.2 seconds at 16000Hz 16-bit = 6400 bytes (> 1KB), duration 0.2s (< 0.5s)
        create_dummy_wav(out_wav, duration_seconds=0.2)
        return subprocess.CompletedProcess(args[0], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine = PiperEngine(model_path=model, cmd=["mock_piper"])
    with pytest.raises(VoiceError) as exc_info:
        engine.synthesize("Some text.", out_wav)
    assert "too short" in exc_info.value.message


def test_piper_engine_output_validation_duration_too_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify VoiceError raised when measured duration is > 90 seconds."""
    model = tmp_path / "voice.onnx"
    model.touch()
    out_wav = tmp_path / "out.wav"

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        create_dummy_wav(out_wav, duration_seconds=95.0)
        return subprocess.CompletedProcess(args[0], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine = PiperEngine(model_path=model, cmd=["mock_piper"])
    with pytest.raises(VoiceError) as exc_info:
        engine.synthesize("Some text.", out_wav)
    assert "too long" in exc_info.value.message


# -----------------------------------------------------------------------------
# 5. synthesize_all, caching, atomic write, and pronunciations.json
# -----------------------------------------------------------------------------


def make_valid_scene(
    scene_id: int,
    scene_type: str,
    heading: str = "Project Overview",
    bullets: list[str] | None = None,
    narration: str = (
        "This is an automated narration sentence designed for testing the voice pipeline "
        "and audio synthesis functionality thoroughly across stages."
    ),
) -> Scene:
    """Create a validated Scene object for tests."""
    return Scene(
        id=scene_id,
        type=scene_type,
        heading=heading,
        bullets=bullets if bullets is not None else ["Key bullet point"],
        narration=narration,
    )


def _make_dummy_script() -> ScriptData:
    """Create a minimal valid ScriptData object for tests."""
    return ScriptData(
        title="Sample Repo Video",
        scenes=[
            make_valid_scene(1, "title", "Project Overview"),
            make_valid_scene(2, "problem", "The Challenge"),
            make_valid_scene(3, "feature", "Key Features"),
            make_valid_scene(4, "howto", "Quick Setup"),
            make_valid_scene(5, "outro", "Get Started"),
        ],
    )


class MockTTSEngine:
    """Mock TTS Engine for pipeline testing without external subprocess calls."""

    def __init__(self, duration_per_scene: float = 12.0) -> None:
        self.duration_per_scene = duration_per_scene
        self.calls: list[tuple[str, Path]] = []
        self.ready_checked = False
        self.model_name = "test-voice.onnx"
        self.voice_speed = 1.0

    def check_ready(self) -> None:
        self.ready_checked = True

    def synthesize(self, text: str, wav_path: Path) -> None:
        self.calls.append((text, wav_path))
        create_dummy_wav(wav_path, duration_seconds=self.duration_per_scene)


def test_synthesize_all_caching(tmp_path: Path) -> None:
    """Verify second run reuses cached files and does not call engine.synthesize."""
    script = _make_dummy_script()
    engine = MockTTSEngine(duration_per_scene=12.0)
    out_dir = tmp_path / "audio"

    # First run
    results1 = synthesize_all(script, out_dir, engine=engine)
    assert engine.ready_checked is True
    assert len(engine.calls) == 5
    assert len(results1) == 5
    assert (out_dir / "scene_01.wav").is_file()
    assert (out_dir / "scene_01.wav.key").is_file()

    # Second run with same parameters: engine.synthesize must NOT be called again
    engine.calls.clear()
    results2 = synthesize_all(script, out_dir, engine=engine)
    assert len(engine.calls) == 0
    assert len(results2) == 5
    assert [r.scene_id for r in results2] == [1, 2, 3, 4, 5]

    # Third run with changed speed -> cache invalidated, re-synthesizes
    engine.voice_speed = 1.25
    engine.calls.clear()
    results3 = synthesize_all(script, out_dir, engine=engine)
    assert len(engine.calls) == 5
    assert len(results3) == 5


def test_synthesize_all_atomic_write_on_failure(tmp_path: Path) -> None:
    """Verify that a synthesis failure leaves no final WAV or temporary file."""
    script = _make_dummy_script()
    out_dir = tmp_path / "audio"

    class FailingEngine:
        def __init__(self) -> None:
            self.model_name = "voice.onnx"
            self.voice_speed = 1.0

        def check_ready(self) -> None:
            pass

        def synthesize(self, text: str, wav_path: Path) -> None:
            # Leave partial file and fail
            wav_path.write_bytes(b"partial unfinished data")
            raise VoiceError("Synthesis crashed")

    engine = FailingEngine()
    with pytest.raises(VoiceError) as exc_info:
        synthesize_all(script, out_dir, engine=engine)

    assert "scene 1" in exc_info.value.message.lower()
    # Ensure final WAV does not exist
    assert not (out_dir / "scene_01.wav").exists()
    # Ensure no temporary file left behind
    temp_files = list(out_dir.glob(".scene_01.wav.tmp_*"))
    assert len(temp_files) == 0


def test_synthesize_all_duration_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify warning is logged when total audio duration is < 30s or > 100s."""
    script = _make_dummy_script()  # 5 scenes

    # Total duration = 5 * 5.0s = 25s (< 30s)
    engine_short = MockTTSEngine(duration_per_scene=5.0)
    with caplog.at_level(logging.WARNING):
        synthesize_all(script, tmp_path / "audio_short", engine=engine_short)
    assert "outside recommended video length" in caplog.text

    caplog.clear()

    # Total duration = 5 * 25.0s = 125s (> 100s)
    engine_long = MockTTSEngine(duration_per_scene=25.0)
    with caplog.at_level(logging.WARNING):
        synthesize_all(script, tmp_path / "audio_long", engine=engine_long)
    assert "outside recommended video length" in caplog.text


def test_synthesize_all_pronunciations_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify valid pronunciations.json is loaded and invalid file is ignored with warning."""
    monkeypatch.chdir(tmp_path)
    pron_file = tmp_path / "pronunciations.json"

    # Write valid custom pronunciations
    pron_file.write_text(json.dumps({"repo2video": "repo to video"}), encoding="utf-8")

    captured_texts: list[str] = []

    class CapturingEngine:
        def __init__(self) -> None:
            self.model_name = "voice.onnx"
            self.voice_speed = 1.0

        def check_ready(self) -> None:
            pass

        def synthesize(self, text: str, wav_path: Path) -> None:
            captured_texts.append(text)
            create_dummy_wav(wav_path, duration_seconds=12.0)

    script = ScriptData(
        title="Custom Pron Script",
        scenes=[
            make_valid_scene(
                1,
                "title",
                "Overview",
                narration="Welcome to the repo2video project today, turning public repositories into demo videos automatically for developers worldwide.",
            ),
            make_valid_scene(2, "problem", "The Challenge"),
            make_valid_scene(3, "feature", "Key Features"),
            make_valid_scene(4, "howto", "Quick Setup"),
            make_valid_scene(
                5,
                "outro",
                "Get Started",
                narration="Install the repo2video package today and start creating narrated videos for your repository right away now.",
            ),
        ],
    )

    synthesize_all(script, tmp_path / "audio", engine=CapturingEngine())
    assert "repo to video" in captured_texts[0]
    assert "repo to video" in captured_texts[4]

    # Corrupt pronunciations.json: should be ignored with warning, not crash
    pron_file.write_text("{invalid json", encoding="utf-8")
    captured_texts.clear()
    synthesize_all(script, tmp_path / "audio2", engine=CapturingEngine())
    # Built-ins still work, no crash
    assert len(captured_texts) == 5


# -----------------------------------------------------------------------------
# 6. CLI entrypoint main() tests
# -----------------------------------------------------------------------------


def test_main_check_ready_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify main --check prints ready message and exits 0."""
    monkeypatch.setattr("sys.argv", ["GRC.voice", "--check"])

    def mock_check_ready(self: PiperEngine) -> None:
        pass

    monkeypatch.setattr(PiperEngine, "check_ready", mock_check_ready)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "ready" in captured.out.lower()


def test_main_check_ready_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify main --check prints error to stderr and exits 1 when unready."""
    monkeypatch.setattr("sys.argv", ["GRC.voice", "--check"])

    def mock_check_ready(self: PiperEngine) -> None:
        raise VoiceError(
            "Piper voice model not found at voices/voice.onnx.", hint="Download voice."
        )

    monkeypatch.setattr(PiperEngine, "check_ready", mock_check_ready)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_main_synthesize_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify main path/to/scenes.json synthesizes WAVs and prints durations."""
    script = _make_dummy_script()
    script_file = tmp_path / "scenes.json"
    script.save(script_file)

    monkeypatch.setattr("sys.argv", ["GRC.voice", str(script_file)])

    def mock_check_ready(self: PiperEngine) -> None:
        pass

    def mock_synthesize(self: PiperEngine, text: str, wav_path: Path) -> None:
        create_dummy_wav(wav_path, duration_seconds=10.0)

    monkeypatch.setattr(PiperEngine, "check_ready", mock_check_ready)
    monkeypatch.setattr(PiperEngine, "synthesize", mock_synthesize)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Scene 1: 10.00s" in captured.out
    assert "Scene 2: 10.00s" in captured.out
    assert "Scene 5: 10.00s" in captured.out
    assert "Total duration: 50.00s across 5 scenes." in captured.out
