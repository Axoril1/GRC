"""Tests for the Typer command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from repo2video import __version__
from repo2video.cli import app
from repo2video.config import Settings
from repo2video.errors import Repo2VideoError
from repo2video.models import RepoData, Scene, ScriptData
from repo2video.pipeline import DoctorCheck, PipelineOptions, PipelineResult

runner = CliRunner()


@pytest.fixture
def fake_script_data() -> ScriptData:
    """Fixture providing a valid 5-scene ScriptData instance."""
    narr = "This is a valid narration passage designed to meet the minimum length requirements for testing purposes."
    return ScriptData(
        title="owner/test-repo",
        scenes=[
            Scene(id=1, type="title", heading="Title", bullets=["b1"], narration=narr, code=""),
            Scene(id=2, type="problem", heading="Problem", bullets=["b2"], narration=narr, code=""),
            Scene(id=3, type="feature", heading="Feature", bullets=["b3"], narration=narr, code=""),
            Scene(id=4, type="howto", heading="Howto", bullets=["b4"], narration=narr, code=""),
            Scene(id=5, type="outro", heading="Outro", bullets=["b5"], narration=narr, code=""),
        ],
    )


@pytest.fixture
def mock_pipeline_result(tmp_path: Path) -> PipelineResult:
    """Fixture providing a mock successful PipelineResult."""
    work_dir = tmp_path / "owner__test-repo"
    work_dir.mkdir(parents=True, exist_ok=True)
    video = work_dir / "demo.mp4"
    video.write_bytes(b"MP4_CONTENT")
    captions = work_dir / "demo.srt"
    captions.write_text("1\n00:00:00,000 --> 00:00:05,000\nCaption", encoding="utf-8")
    repo_json = work_dir / "repo.json"
    repo_json.write_text("{}", encoding="utf-8")
    scenes_json = work_dir / "scenes.json"
    scenes_json.write_text("{}", encoding="utf-8")

    return PipelineResult(
        work_dir=work_dir,
        repo_json=repo_json,
        scenes_json=scenes_json,
        video=video,
        captions=captions,
        duration_seconds=15.0,
        timings={"analyze": 1.0, "script": 2.0, "slides": 3.0, "voice": 2.5, "render": 1.5},
    )


def test_version() -> None:
    """Verify --version option outputs application version and exits 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"repo2video {__version__}" in result.output

    # Short flag -v
    res_short = runner.invoke(app, ["-v"])
    assert res_short.exit_code == 0
    assert f"repo2video {__version__}" in res_short.output


def test_make_success(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline_result: PipelineResult,
) -> None:
    """Verify make command runs full pipeline, printing progress to stderr and video path to stdout."""
    recorded: dict[str, Any] = {}

    def fake_run(
        url: str, out: Path | None, settings: Settings, options: PipelineOptions
    ) -> PipelineResult:
        recorded["url"] = url
        recorded["out"] = out
        recorded["settings"] = settings
        recorded["options"] = options
        # Simulate stage progress callbacks
        options.on_stage("analyze", 1, 5)
        options.on_stage("script", 2, 5)
        options.on_stage("slides", 3, 5)
        options.on_stage("voice", 4, 5)
        options.on_stage("render", 5, 5)
        return mock_pipeline_result

    monkeypatch.setattr("repo2video.cli.run_pipeline", fake_run)

    result = runner.invoke(app, ["make", "https://github.com/owner/test-repo"])
    assert result.exit_code == 0
    assert recorded["url"] == "https://github.com/owner/test-repo"

    # stdout should contain the final video path
    assert str(mock_pipeline_result.video) in result.stdout

    # stderr should contain progress and summary
    assert "[1/5] Analyzing repository..." in result.stderr
    assert "[5/5] Compositing video with FFmpeg..." in result.stderr
    assert "Done in" in result.stderr
    assert "Summary:" in result.stderr
    assert "Duration: 15.0s" in result.stderr


def test_make_script_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify make --script-only outputs the scenes.json path to stdout."""
    work_dir = tmp_path / "owner__test-repo"
    work_dir.mkdir(parents=True, exist_ok=True)
    scenes_json = work_dir / "scenes.json"
    scenes_json.write_text("{}", encoding="utf-8")

    script_result = PipelineResult(
        work_dir=work_dir,
        repo_json=work_dir / "repo.json",
        scenes_json=scenes_json,
        video=None,
        captions=None,
        duration_seconds=None,
        timings={"analyze": 1.0, "script": 2.0},
    )

    recorded_opts: list[PipelineOptions] = []

    def fake_run(
        url: str, out: Path | None, settings: Settings, options: PipelineOptions
    ) -> PipelineResult:
        recorded_opts.append(options)
        options.on_stage("analyze", 1, 2)
        options.on_stage("script", 2, 2)
        return script_result

    monkeypatch.setattr("repo2video.cli.run_pipeline", fake_run)

    result = runner.invoke(app, ["make", "https://github.com/owner/test-repo", "--script-only"])
    assert result.exit_code == 0
    assert recorded_opts[0].script_only is True
    assert str(scenes_json) in result.stdout
    assert "Summary:" in result.stderr


def test_make_option_overrides(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline_result: PipelineResult,
    tmp_path: Path,
) -> None:
    """Verify CLI options override settings without mutating the base Settings object."""
    recorded: dict[str, Any] = {}

    def fake_run(
        url: str, out: Path | None, settings: Settings, options: PipelineOptions
    ) -> PipelineResult:
        recorded["settings"] = settings
        recorded["options"] = options
        return mock_pipeline_result

    monkeypatch.setattr("repo2video.cli.run_pipeline", fake_run)

    custom_out = tmp_path / "custom.mp4"
    result = runner.invoke(
        app,
        [
            "make",
            "https://github.com/owner/test-repo",
            "--out",
            str(custom_out),
            "--provider",
            "openai",
            "--model",
            "gpt-4o",
            "--voice-speed",
            "1.2",
            "--theme",
            "light",
            "--accent",
            "#ff0000",
            "--reuse",
            "--force",
            "--keep-temp",
        ],
    )
    assert result.exit_code == 0
    settings: Settings = recorded["settings"]
    options: PipelineOptions = recorded["options"]

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o"
    assert settings.voice_speed == 1.2

    assert options.theme == "light"
    assert options.accent == "#ff0000"
    assert options.reuse is True
    assert options.force is True
    assert options.keep_temp is True


def test_make_invalid_options() -> None:
    """Verify friendly error messages and non-zero exit for invalid option inputs."""
    # Unknown theme
    res_theme = runner.invoke(
        app, ["make", "https://github.com/owner/test-repo", "--theme", "neon"]
    )
    assert res_theme.exit_code != 0
    assert "Unknown theme 'neon'" in res_theme.output

    # Invalid accent color
    res_accent = runner.invoke(
        app, ["make", "https://github.com/owner/test-repo", "--accent", "blue"]
    )
    assert res_accent.exit_code != 0
    assert "Invalid accent color" in res_accent.output

    # Unknown LLM provider
    res_prov = runner.invoke(
        app, ["make", "https://github.com/owner/test-repo", "--provider", "claude"]
    )
    assert res_prov.exit_code != 0
    assert "Unknown provider 'claude'" in res_prov.output

    # Voice speed out of bounds
    res_speed = runner.invoke(
        app, ["make", "https://github.com/owner/test-repo", "--voice-speed", "2.5"]
    )
    assert res_speed.exit_code != 0
    assert "voice_speed must be between 0.5 and 1.5" in res_speed.output


def test_make_expected_error_formatting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Repo2VideoError prints Error: and Hint: without a Python traceback."""

    def fake_run(
        url: str, out: Path | None, settings: Settings, options: PipelineOptions
    ) -> PipelineResult:
        raise Repo2VideoError(
            "GitHub repository psf/requests was not found.",
            hint="Check repository URL and privacy.",
        )

    monkeypatch.setattr("repo2video.cli.run_pipeline", fake_run)

    result = runner.invoke(app, ["make", "https://github.com/psf/requests"])
    assert result.exit_code == 1
    assert "Error: GitHub repository psf/requests was not found." in result.stderr
    assert "Hint: Check repository URL and privacy." in result.stderr
    assert "Traceback (most recent call last)" not in result.stderr


def test_make_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify KeyboardInterrupt prints Cancelled. and exits with code 130."""
    target_out = tmp_path / "cancelled.mp4"
    target_out.write_bytes(b"")

    def fake_run(
        url: str, out: Path | None, settings: Settings, options: PipelineOptions
    ) -> PipelineResult:
        raise KeyboardInterrupt()

    monkeypatch.setattr("repo2video.cli.run_pipeline", fake_run)

    result = runner.invoke(
        app, ["make", "https://github.com/owner/test-repo", "--out", str(target_out)]
    )
    assert result.exit_code == 130
    assert "Cancelled." in result.stderr
    # Incomplete 0-byte output file removed
    assert not target_out.exists()


def test_make_unexpected_exception_writes_crash_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify unexpected exceptions generate a crash.log with traceback and exit 1."""
    out_dir = tmp_path / "test_out"

    def fake_get_settings() -> Settings:
        return Settings(output_dir=out_dir)

    monkeypatch.setattr("repo2video.cli.get_settings", fake_get_settings)

    def fake_run(
        url: str, out: Path | None, settings: Settings, options: PipelineOptions
    ) -> PipelineResult:
        raise RuntimeError("Unexpected internal crash during pipeline orchestration")

    monkeypatch.setattr("repo2video.cli.run_pipeline", fake_run)

    result = runner.invoke(app, ["make", "https://github.com/owner/test-repo"])
    assert result.exit_code == 1
    assert "Unexpected error: Unexpected internal crash" in result.stderr
    assert "Traceback written to" in result.stderr
    assert "crash.log" in result.stderr

    crash_log = out_dir / "crash.log"
    assert crash_log.is_file()
    crash_text = crash_log.read_text(encoding="utf-8")
    assert "RuntimeError: Unexpected internal crash" in crash_text
    assert "Traceback" in crash_text


def test_render_command_accepts_file_and_folder(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline_result: PipelineResult,
    tmp_path: Path,
) -> None:
    """Verify render command accepts both scenes.json path and containing directory."""
    work_dir = tmp_path / "manual_dir"
    work_dir.mkdir(parents=True, exist_ok=True)
    scenes_file = work_dir / "scenes.json"
    scenes_file.write_text("{}", encoding="utf-8")

    recorded_paths: list[Path] = []

    def fake_render_only(
        scenes_json: Path,
        repo_label: str,
        out: Path | None,
        settings: Settings,
        options: PipelineOptions,
    ) -> PipelineResult:
        recorded_paths.append(scenes_json)
        return mock_pipeline_result

    monkeypatch.setattr("repo2video.cli.render_only", fake_render_only)

    # 1. Passing scenes.json file directly
    res1 = runner.invoke(app, ["render", str(scenes_file), "--label", "owner/manual"])
    assert res1.exit_code == 0
    assert str(mock_pipeline_result.video) in res1.stdout
    assert recorded_paths[-1] == scenes_file

    # 2. Passing directory containing scenes.json
    res2 = runner.invoke(app, ["render", str(work_dir), "--label", "owner/manual"])
    assert res2.exit_code == 0
    assert str(mock_pipeline_result.video) in res2.stdout
    assert recorded_paths[-1] == scenes_file


def test_render_command_label_inference(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipeline_result: PipelineResult,
    tmp_path: Path,
) -> None:
    """Verify render command infers repository label from repo.json when --label is omitted."""
    work_dir = tmp_path / "inferred_dir"
    work_dir.mkdir(parents=True, exist_ok=True)
    scenes_file = work_dir / "scenes.json"
    scenes_file.write_text("{}", encoding="utf-8")

    repo_data = RepoData(
        name="inferred-repo",
        owner="inferred-owner",
        description="Desc",
        stars=10,
        languages=["Python"],
        readme="# Readme",
        top_level=["main.py"],
    )
    repo_data.save(work_dir / "repo.json")

    recorded_label: list[str] = []

    def fake_render_only(
        scenes_json: Path,
        repo_label: str,
        out: Path | None,
        settings: Settings,
        options: PipelineOptions,
    ) -> PipelineResult:
        recorded_label.append(repo_label)
        return mock_pipeline_result

    monkeypatch.setattr("repo2video.cli.render_only", fake_render_only)

    res = runner.invoke(app, ["render", str(scenes_file)])
    assert res.exit_code == 0
    assert recorded_label[0] == "inferred-owner/inferred-repo"


def test_doctor_command_all_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify doctor prints OK for all checks and exits 0 when dependencies are available."""
    checks = [
        DoctorCheck(name="Python version (3.12.3)", ok=True, status="OK"),
        DoctorCheck(name="FFmpeg and encoders", ok=True, status="OK"),
        DoctorCheck(name="Playwright Chromium", ok=True, status="OK"),
        DoctorCheck(name="Piper and voice file", ok=True, status="OK"),
        DoctorCheck(name="LLM server and model (ollama: mistral:latest)", ok=True, status="OK"),
        DoctorCheck(name="GITHUB_TOKEN", ok=True, status="set", required=False),
    ]

    monkeypatch.setattr("repo2video.cli.check_doctor", lambda s: checks)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python version (3.12.3): OK" in result.output
    assert "FFmpeg and encoders: OK" in result.output
    assert "Playwright Chromium: OK" in result.output
    assert "Piper and voice file: OK" in result.output
    assert "LLM server and model" in result.output
    assert "GITHUB_TOKEN: set" in result.output
    assert "FAIL" not in result.output


def test_doctor_command_mixed_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify doctor prints FAIL with hints and exits 1 when a dependency is missing."""
    checks = [
        DoctorCheck(name="Python version (3.12.3)", ok=True, status="OK"),
        DoctorCheck(name="FFmpeg and encoders", ok=True, status="OK"),
        DoctorCheck(name="Playwright Chromium", ok=True, status="OK"),
        DoctorCheck(
            name="Piper and voice file",
            ok=False,
            status="FAIL",
            hint="Download voice model",
            required=True,
        ),
        DoctorCheck(
            name="LLM server and model (ollama: mistral:latest)",
            ok=False,
            status="FAIL",
            hint="Run: ollama serve",
            required=True,
        ),
        DoctorCheck(name="GITHUB_TOKEN", ok=True, status="not set", required=False),
    ]

    monkeypatch.setattr("repo2video.cli.check_doctor", lambda s: checks)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "Python version (3.12.3): OK" in result.output
    assert "Piper and voice file: FAIL - Hint: Download voice model" in result.output
    assert (
        "LLM server and model (ollama: mistral:latest): FAIL - Hint: Run: ollama serve"
        in result.output
    )
    assert "GITHUB_TOKEN: not set" in result.output
