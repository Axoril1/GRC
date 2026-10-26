"""Tests for CLI v2 flags, progress reporting, validation, summary formatting, and doctor additions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from GRC.cli import app
from GRC.config import Settings
from GRC.pipeline import DoctorCheck, PipelineOptions, PipelineResult

runner = CliRunner()


@pytest.fixture
def dummy_result(tmp_path: Path) -> PipelineResult:
    """Fixture providing a PipelineResult with realistic media, fallbacks, and timings."""
    work_dir = tmp_path / "test-owner__demo-repo"
    work_dir.mkdir(parents=True, exist_ok=True)
    video = work_dir / "demo.mp4"
    video.write_bytes(b"MP4_DATA")
    captions = work_dir / "demo.srt"
    captions.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello", encoding="utf-8")
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
        duration_seconds=12.5,
        timings={
            "analyze": 0.5,
            "media": 0.8,
            "script": 1.2,
            "voice": 2.0,
            "visuals": 4.5,
            "render": 1.5,
        },
        scene_counts={"title": 1, "feature": 2, "code": 1, "outro": 1},
        fallbacks=[],
        media_count=2,
    )


# ==============================================================================
# Progress and Stage Order Tests
# ==============================================================================


def test_cli_six_stage_progress_output(
    monkeypatch: pytest.MonkeyPatch, dummy_result: PipelineResult
) -> None:
    """Verify standard make command prints all 6 stage progress lines [1/6] to [6/6]."""

    def fake_run(
        url: str, out: Any = None, settings: Any = None, options: Any = None
    ) -> PipelineResult:
        if options and options.on_stage:
            options.on_stage("analyze", 1, 6)
            options.on_stage("media", 2, 6)
            options.on_stage("script", 3, 6)
            options.on_stage("voice", 4, 6)
            options.on_stage("visuals", 5, 6)
            options.on_stage("render", 6, 6)
        return dummy_result

    monkeypatch.setattr("GRC.cli.run_pipeline", fake_run)

    res = runner.invoke(app, ["make", "https://github.com/test-owner/demo-repo"])
    assert res.exit_code == 0
    stderr = res.stderr
    assert "[1/6] Analyzing repository..." in stderr
    assert "[2/6] Collecting media assets..." in stderr
    assert "[3/6] Writing the script..." in stderr
    assert "[4/6] Synthesizing voice narration..." in stderr
    assert "[5/6] Rendering visual scenes..." in stderr
    assert "[6/6] Compositing video with FFmpeg..." in stderr


def test_cli_script_only_three_stage_progress_output(
    monkeypatch: pytest.MonkeyPatch, dummy_result: PipelineResult
) -> None:
    """Verify --script-only prints 3 stage progress lines [1/3] to [3/3]."""

    def fake_run(
        url: str, out: Any = None, settings: Any = None, options: Any = None
    ) -> PipelineResult:
        if options and options.on_stage:
            options.on_stage("analyze", 1, 3)
            options.on_stage("media", 2, 3)
            options.on_stage("script", 3, 3)
        return dummy_result

    monkeypatch.setattr("GRC.cli.run_pipeline", fake_run)

    res = runner.invoke(app, ["make", "https://github.com/test-owner/demo-repo", "--script-only"])
    assert res.exit_code == 0
    stderr = res.stderr
    assert "[1/3] Analyzing repository..." in stderr
    assert "[2/3] Collecting media assets..." in stderr
    assert "[3/3] Writing the script..." in stderr
    assert "[4/6]" not in stderr


# ==============================================================================
# Option Validation Tests
# ==============================================================================


def test_cli_captions_validation() -> None:
    """Verify invalid --captions mode raises BadParameter error."""
    res = runner.invoke(
        app, ["make", "https://github.com/owner/repo", "--captions", "invalid_mode"]
    )
    assert res.exit_code != 0
    assert "Unknown captions mode 'invalid_mode'" in (res.stderr or res.stdout)


def test_cli_music_missing_file_validation(tmp_path: Path) -> None:
    """Verify missing --music file path raises BadParameter error."""
    missing = tmp_path / "missing_track.mp3"
    res = runner.invoke(app, ["make", "https://github.com/owner/repo", "--music", str(missing)])
    assert res.exit_code != 0
    assert "Music file not found" in (res.stderr or res.stdout)


def test_cli_music_unsupported_extension_validation(tmp_path: Path) -> None:
    """Verify --music with unsupported extension raises BadParameter error."""
    bad_music = tmp_path / "track.exe"
    bad_music.write_bytes(b"DATA")
    res = runner.invoke(app, ["make", "https://github.com/owner/repo", "--music", str(bad_music)])
    assert res.exit_code != 0
    assert "Unsupported music file extension '.exe'" in (res.stderr or res.stdout)


def test_cli_music_rights_warning_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_result: PipelineResult
) -> None:
    """Verify valid --music prints the copyright notice warning to stderr."""
    good_music = tmp_path / "ambient.mp3"
    good_music.write_bytes(b"MP3_DATA")

    monkeypatch.setattr("GRC.cli.run_pipeline", lambda *args, **kwargs: dummy_result)

    res = runner.invoke(app, ["make", "https://github.com/owner/repo", "--music", str(good_music)])
    assert res.exit_code == 0
    assert "Make sure you have the right to use this music." in res.stderr


def test_cli_anim_fps_out_of_range_validation() -> None:
    """Verify --anim-fps outside 10-30 raises BadParameter error."""
    res_low = runner.invoke(app, ["make", "https://github.com/owner/repo", "--anim-fps", "5"])
    assert res_low.exit_code != 0
    assert "anim-fps must be between 10 and 30" in (res_low.stderr or res_low.stdout)

    res_high = runner.invoke(app, ["make", "https://github.com/owner/repo", "--anim-fps", "45"])
    assert res_high.exit_code != 0
    assert "anim-fps must be between 10 and 30" in (res_high.stderr or res_high.stdout)


def test_cli_fast_flag_overrides(
    monkeypatch: pytest.MonkeyPatch, dummy_result: PipelineResult
) -> None:
    """Verify --fast sets anim_quality to fast and anim_fps to 15 in settings overrides."""
    captured_settings: list[Settings] = []
    captured_options: list[PipelineOptions] = []

    def fake_run(
        url: str, out: Any = None, settings: Any = None, options: Any = None
    ) -> PipelineResult:
        if settings:
            captured_settings.append(settings)
        if options:
            captured_options.append(options)
        return dummy_result

    monkeypatch.setattr("GRC.cli.run_pipeline", fake_run)

    res = runner.invoke(app, ["make", "https://github.com/owner/repo", "--fast"])
    assert res.exit_code == 0
    assert len(captured_settings) == 1
    assert captured_settings[0].anim_quality == "fast"
    assert captured_settings[0].anim_fps == 15
    assert captured_options[0].fast is True


# ==============================================================================
# Summary Formatting Tests
# ==============================================================================


def test_cli_summary_reports_scene_types_and_counts(
    monkeypatch: pytest.MonkeyPatch, dummy_result: PipelineResult
) -> None:
    """Verify summary output prints scene types with their counts."""
    monkeypatch.setattr("GRC.cli.run_pipeline", lambda *args, **kwargs: dummy_result)

    res = runner.invoke(app, ["make", "https://github.com/owner/repo"])
    assert res.exit_code == 0
    stderr = res.stderr
    assert "Scenes:   1x code, 2x feature, 1x outro, 1x title" in stderr


def test_cli_summary_reports_fallbacks_in_yellow(
    monkeypatch: pytest.MonkeyPatch, dummy_result: PipelineResult
) -> None:
    """Verify summary output displays yellow fallback notice and scene lines."""
    dummy_result.fallbacks = [
        {"scene_id": 5, "type": "code", "reason": "Pygments formatting error"},
        {"scene_id": 7, "type": "terminal", "reason": "Terminal replay parse failure"},
    ]
    monkeypatch.setattr("GRC.cli.run_pipeline", lambda *args, **kwargs: dummy_result)

    res = runner.invoke(app, ["make", "https://github.com/owner/repo"])
    assert res.exit_code == 0
    stderr = res.stderr
    assert "Some scenes fell back to plain static slides:" in stderr
    assert "Scene 5 (code): Pygments formatting error" in stderr
    assert "Scene 7 (terminal): Terminal replay parse failure" in stderr


def test_cli_summary_reports_no_media_hint_unless_disabled(
    monkeypatch: pytest.MonkeyPatch, dummy_result: PipelineResult
) -> None:
    """Verify no-media hint prints when media_count is 0, but is omitted when --no-media is passed."""
    dummy_result.media_count = 0
    monkeypatch.setattr("GRC.cli.run_pipeline", lambda *args, **kwargs: dummy_result)

    # 1. Normal run with 0 media: hint must appear
    res_normal = runner.invoke(app, ["make", "https://github.com/owner/repo"])
    assert res_normal.exit_code == 0
    assert "This repository has no usable screenshots." in res_normal.stderr

    # 2. Run with --no-media: hint must NOT appear
    res_no_media = runner.invoke(app, ["make", "https://github.com/owner/repo", "--no-media"])
    assert res_no_media.exit_code == 0
    assert "This repository has no usable screenshots." not in res_no_media.stderr


# ==============================================================================
# Doctor Command Tests
# ==============================================================================


def test_cli_doctor_missing_xfade_shows_warn_and_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify doctor check with missing xfade reports WARN and exits 0."""
    checks = [
        DoctorCheck(name="Pillow image library", ok=True, status="OK"),
        DoctorCheck(name="Pygments syntax highlighter", ok=True, status="OK"),
        DoctorCheck(name="FFmpeg overlay filter", ok=True, status="OK"),
        DoctorCheck(
            name="FFmpeg xfade filter",
            ok=False,
            status="WARN",
            hint="Scene transitions will fall back to jump cuts.",
            required=False,
        ),
    ]
    monkeypatch.setattr("GRC.cli.check_doctor", lambda settings: checks)

    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0
    output = res.stdout
    assert "FFmpeg xfade filter: WARN" in output
    assert "Scene transitions will fall back to jump cuts." in output


def test_cli_doctor_missing_pillow_shows_fail_and_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify doctor check with missing Pillow reports FAIL and exits 1."""
    checks = [
        DoctorCheck(
            name="Pillow image library",
            ok=False,
            status="FAIL",
            hint="Install Pillow: `pip install pillow`.",
            required=True,
        ),
        DoctorCheck(name="Pygments syntax highlighter", ok=True, status="OK"),
    ]
    monkeypatch.setattr("GRC.cli.check_doctor", lambda settings: checks)

    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 1
    output = res.stdout
    assert "Pillow image library: FAIL" in output
    assert "Install Pillow: `pip install pillow`." in output


# ==============================================================================
# Render-Only Command Tests with V2 Flags
# ==============================================================================


def test_cli_render_command_passes_v2_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_result: PipelineResult
) -> None:
    """Verify the render CLI command parses and forwards all v2 options to render_only."""
    scenes_path = tmp_path / "scenes.json"
    scenes_path.write_text("{}", encoding="utf-8")

    captured_options: list[PipelineOptions] = []

    def fake_render_only(*args: Any, **kwargs: Any) -> PipelineResult:
        options = kwargs.get("options")
        if options:
            captured_options.append(options)
        return dummy_result

    monkeypatch.setattr("GRC.cli.render_only", fake_render_only)

    res = runner.invoke(
        app,
        [
            "render",
            str(scenes_path),
            "--no-animation",
            "--no-transitions",
            "--captions",
            "burn",
            "--fast",
            "--anim-fps",
            "20",
        ],
    )
    assert res.exit_code == 0
    assert len(captured_options) == 1
    opts = captured_options[0]
    assert opts.no_animation is True
    assert opts.no_transitions is True
    assert opts.captions == "burn"
    assert opts.fast is True
    assert opts.anim_fps == 20
