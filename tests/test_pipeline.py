"""Tests for the pipeline orchestrator, preflight checks, and doctor inspections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from repo2video.config import Settings
from repo2video.errors import (
    LLMError,
    RenderError,
    Repo2VideoError,
    VoiceError,
)
from repo2video.models import RepoData, Scene, ScriptData
from repo2video.pipeline import (
    DoctorCheck,
    PipelineOptions,
    PipelineResult,
    _pair_scene_assets,
    _run_preflight,
    _sanitize_name,
    check_doctor,
    render_only,
    run,
)
from repo2video.render import SceneAssets, SceneVisual
from repo2video.voice import SceneAudio


@pytest.fixture
def sample_repo_data() -> RepoData:
    """Fixture providing a valid RepoData instance."""
    return RepoData(
        name="demo-repo",
        owner="test-owner",
        description="A great demo repo",
        stars=123,
        languages=["Python", "TypeScript"],
        readme="# Demo Repo\nThis is a test demo repo.",
        top_level=["src", "tests", "README.md"],
    )


@pytest.fixture
def sample_script_data() -> ScriptData:
    """Fixture providing a valid 5-scene ScriptData instance."""
    narr = "This is a valid narration passage designed to meet the minimum length requirements for testing purposes."
    return ScriptData(
        title="test-owner/demo-repo",
        scenes=[
            Scene(
                id=1,
                type="title",
                heading="Introduction",
                bullets=["Welcome to demo-repo"],
                narration=narr,
                code="",
            ),
            Scene(
                id=2,
                type="problem",
                heading="The Problem",
                bullets=["Manual video production takes forever"],
                narration=narr,
                code="",
            ),
            Scene(
                id=3,
                type="feature",
                heading="Key Features",
                bullets=["Automated slide rendering and narration"],
                narration=narr,
                code="",
            ),
            Scene(
                id=4,
                type="howto",
                heading="Quick Start",
                bullets=["Run pip install repo2video"],
                narration=narr,
                code="",
            ),
            Scene(
                id=5,
                type="outro",
                heading="Conclusion",
                bullets=["Try it out today"],
                narration=narr,
                code="",
            ),
        ],
    )


@pytest.fixture
def mock_pipeline_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_repo_data: RepoData,
    sample_script_data: ScriptData,
) -> dict[str, Any]:
    """Mock external pipeline stage functions with in-memory test doubles."""
    calls: list[str] = []

    # Settings pointing to tmp_path
    settings = Settings(
        output_dir=tmp_path / "out",
        piper_voice_path=tmp_path / "voices" / "en_US-lessac-medium.onnx",
        github_token="secret_token_12345",
        llm_api_key="secret_key_67890",
    )

    # Fake preflight checks
    class FakeLLMClient:
        def check_ready(self) -> None:
            calls.append("llm_check_ready")

        def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
            return "{}"

    class FakePiperEngine:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def check_ready(self) -> None:
            calls.append("piper_check_ready")

    monkeypatch.setattr("repo2video.pipeline.get_client", lambda s: FakeLLMClient())
    monkeypatch.setattr("repo2video.pipeline.PiperEngine", FakePiperEngine)
    monkeypatch.setattr(
        "repo2video.pipeline.check_ffmpeg", lambda: calls.append("check_ffmpeg") or "ffmpeg 6.1"
    )
    monkeypatch.setattr(
        "repo2video.pipeline.check_chromium",
        lambda: calls.append("check_chromium") or "/fake/chromium",
    )

    # Stage mocks
    def fake_analyze(url: str, settings: Settings) -> RepoData:
        calls.append("analyze")
        return sample_repo_data

    def fake_write_script(repo: RepoData, client: Any, settings: Settings) -> ScriptData:
        calls.append("script")
        return sample_script_data

    def fake_collect_media(
        repo: RepoData,
        work_dir: Path,
        settings: Settings,
        force: bool = False,
    ) -> list[Any]:
        calls.append("media")
        return []

    def fake_render_visuals(
        script_data: ScriptData,
        repo_data: RepoData,
        media_assets: list[Any],
        work_dir: Path,
        audios: list[SceneAudio],
        settings: Settings,
        options: PipelineOptions,
        *,
        render_options: Any,
        on_scene: Any = None,
    ) -> tuple[list[SceneVisual], list[dict[str, Any]], str | None]:
        from repo2video.render import SceneVisual

        calls.append("visuals")
        (work_dir / "slides").mkdir(parents=True, exist_ok=True)
        visuals = [
            SceneVisual(kind="image", image=work_dir / "slides" / f"scene_{s.id:02d}.png")
            for s in script_data.scenes
        ]
        return visuals, [], None

    def fake_render_scenes(
        script: ScriptData,
        out_dir: Path,
        repo_label: str,
        theme: str,
        accent: str,
        settings: Settings,
    ) -> list[Path]:
        calls.append("slides")
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for s in script.scenes:
            p = out_dir / f"scene_{s.id:02d}.png"
            p.write_bytes(b"PNG_DATA")
            paths.append(p)
        return paths

    def fake_synthesize_all(
        script: ScriptData, out_dir: Path, settings: Settings
    ) -> list[SceneAudio]:
        calls.append("voice")
        out_dir.mkdir(parents=True, exist_ok=True)
        audios: list[SceneAudio] = []
        for s in script.scenes:
            p = out_dir / f"scene_{s.id:02d}.wav"
            p.write_bytes(b"RIFF_WAV")
            audios.append(SceneAudio(scene_id=s.id, path=p, duration_seconds=3.0))
        return audios

    def fake_render_video(
        assets: list[SceneAssets],
        out_path: Path,
        settings: Settings,
        keep_temp: bool = False,
    ) -> None:
        calls.append("render")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"FAKE_MP4_CONTENT")
        out_path.with_suffix(".srt").write_text(
            "1\n00:00:00,000 --> 00:00:03,000\nSub", encoding="utf-8"
        )

    def fake_probe_video(path: Path) -> dict[str, Any]:
        return {
            "duration_seconds": 6.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "video_codec": "h264",
            "audio_codec": "aac",
            "size_bytes": 1024,
        }

    monkeypatch.setattr("repo2video.pipeline.analyze", fake_analyze)
    monkeypatch.setattr("repo2video.pipeline.collect_media", fake_collect_media)
    monkeypatch.setattr("repo2video.pipeline.write_script", fake_write_script)
    monkeypatch.setattr("repo2video.pipeline.render_scenes", fake_render_scenes)
    monkeypatch.setattr("repo2video.pipeline._render_visuals", fake_render_visuals)
    monkeypatch.setattr("repo2video.pipeline.synthesize_all", fake_synthesize_all)
    monkeypatch.setattr("repo2video.pipeline.render_video", fake_render_video)
    monkeypatch.setattr("repo2video.pipeline.probe_video", fake_probe_video)

    return {
        "calls": calls,
        "settings": settings,
        "sample_repo": sample_repo_data,
        "sample_script": sample_script_data,
    }


def test_sanitize_name() -> None:
    """Verify name sanitization replaces disallowed characters and lowercases."""
    assert _sanitize_name("psf") == "psf"
    assert _sanitize_name("Requests.Repo-1_2") == "requests.repo-1_2"
    assert _sanitize_name("Hello World!@#") == "hello_world___"
    assert _sanitize_name("Foo/Bar") == "foo_bar"


def test_pipeline_stage_ordering_and_progress_callback(
    mock_pipeline_environment: dict[str, Any],
) -> None:
    """Verify stages execute in correct order and notify the progress callback."""
    settings: Settings = mock_pipeline_environment["settings"]
    stage_events: list[tuple[str, int, int]] = []

    def on_stage(name: str, number: int, total: int) -> None:
        stage_events.append((name, number, total))

    options = PipelineOptions(on_stage=on_stage)
    result = run("https://github.com/test-owner/demo-repo", settings=settings, options=options)

    # Check progress callbacks
    assert stage_events == [
        ("analyze", 1, 6),
        ("media", 2, 6),
        ("script", 3, 6),
        ("voice", 4, 6),
        ("visuals", 5, 6),
        ("render", 6, 6),
    ]

    # Check execution sequence in calls
    calls: list[str] = mock_pipeline_environment["calls"]
    expected_calls = [
        "llm_check_ready",
        "piper_check_ready",
        "check_ffmpeg",
        "analyze",
        "media",
        "script",
        "voice",
        "visuals",
        "render",
    ]
    assert calls == expected_calls

    # Check returned PipelineResult
    assert isinstance(result, PipelineResult)
    assert result.work_dir.name == "test-owner__demo-repo"
    assert result.repo_json.is_file()
    assert result.scenes_json.is_file()
    assert result.video is not None and result.video.is_file()
    assert result.captions is not None and result.captions.is_file()
    assert result.duration_seconds == 6.0
    for stage in ["analyze", "media", "script", "voice", "visuals", "render"]:
        assert stage in result.timings
        assert result.timings[stage] >= 0.0


def test_pipeline_script_only(
    mock_pipeline_environment: dict[str, Any],
) -> None:
    """Verify script_only stops after scriptwriter without calling media stages."""
    settings: Settings = mock_pipeline_environment["settings"]
    stage_events: list[tuple[str, int, int]] = []

    options = PipelineOptions(
        script_only=True,
        on_stage=lambda name, num, tot: stage_events.append((name, num, tot)),
    )
    result = run("https://github.com/test-owner/demo-repo", settings=settings, options=options)

    assert stage_events == [
        ("analyze", 1, 3),
        ("media", 2, 3),
        ("script", 3, 3),
    ]

    calls: list[str] = mock_pipeline_environment["calls"]
    # Preflight in script_only only checks LLM readiness
    assert "llm_check_ready" in calls
    assert "piper_check_ready" not in calls
    assert "check_ffmpeg" not in calls
    assert "visuals" not in calls
    assert "voice" not in calls
    assert "render" not in calls

    assert result.video is None
    assert result.captions is None
    assert result.duration_seconds is None
    assert result.scenes_json.is_file()

    # run.json written with None for video and captions
    run_json = result.work_dir / "run.json"
    assert run_json.is_file()
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    assert meta["outputs"]["video"] is None
    assert meta["outputs"]["captions"] is None


def test_pipeline_reuse_skips_analyze_and_script(
    mock_pipeline_environment: dict[str, Any],
) -> None:
    """Verify reuse=True skips analyze and script when repo.json and scenes.json exist."""
    settings: Settings = mock_pipeline_environment["settings"]

    # First run without reuse to generate artifacts
    run("https://github.com/test-owner/demo-repo", settings=settings)
    mock_pipeline_environment["calls"].clear()

    # Second run with reuse=True
    options = PipelineOptions(reuse=True, force=False)
    result = run("https://github.com/test-owner/demo-repo", settings=settings, options=options)

    calls: list[str] = mock_pipeline_environment["calls"]
    assert "analyze" not in calls
    assert "script" not in calls
    assert result.timings["analyze"] == 0.0
    assert result.timings["script"] == 0.0


def test_pipeline_reuse_falls_back_on_corrupt_files(
    mock_pipeline_environment: dict[str, Any],
) -> None:
    """Verify reuse falls back to fresh generation when cached json is malformed."""
    settings: Settings = mock_pipeline_environment["settings"]
    work_dir = settings.output_dir / "test-owner__demo-repo"
    work_dir.mkdir(parents=True, exist_ok=True)
    repo_json = work_dir / "repo.json"
    repo_json.write_text("INVALID_JSON{", encoding="utf-8")

    options = PipelineOptions(reuse=True, force=False)
    run("https://github.com/test-owner/demo-repo", settings=settings, options=options)

    calls: list[str] = mock_pipeline_environment["calls"]
    assert "analyze" in calls


def test_pipeline_force_ignores_reuse_and_cleans_directories(
    mock_pipeline_environment: dict[str, Any],
) -> None:
    """Verify force=True ignores reuse, rebuilds stages, and clears media directories."""
    settings: Settings = mock_pipeline_environment["settings"]

    # First run
    run("https://github.com/test-owner/demo-repo", settings=settings)
    work_dir = settings.output_dir / "test-owner__demo-repo"
    stray_slide = work_dir / "slides" / "stray.png"
    stray_slide.write_bytes(b"STRAY")

    mock_pipeline_environment["calls"].clear()

    # Run with both reuse=True and force=True -> force wins
    options = PipelineOptions(reuse=True, force=True)
    run("https://github.com/test-owner/demo-repo", settings=settings, options=options)

    calls: list[str] = mock_pipeline_environment["calls"]
    assert "analyze" in calls
    assert "script" in calls
    assert not stray_slide.exists()


def test_preflight_reports_all_failures_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify preflight runs all checks and aggregates all failures in a single error."""
    settings = Settings(output_dir=tmp_path / "out")

    class BrokenLLMClient:
        def check_ready(self) -> None:
            raise LLMError(
                "Cannot reach Ollama at http://localhost:11434.", hint="Run: ollama serve"
            )

    class BrokenPiperEngine:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            raise VoiceError("Piper voice model not found.", hint="Download voice model")

    def broken_ffmpeg() -> str:
        raise RenderError("FFmpeg binary missing from PATH.", hint="Install FFmpeg")

    monkeypatch.setattr("repo2video.pipeline.get_client", lambda s: BrokenLLMClient())
    monkeypatch.setattr("repo2video.pipeline.PiperEngine", BrokenPiperEngine)
    monkeypatch.setattr("repo2video.pipeline.check_ffmpeg", broken_ffmpeg)

    with pytest.raises(Repo2VideoError) as exc_info:
        _run_preflight(settings, script_only=False)

    err = exc_info.value
    assert "Preflight checks failed with 3 error(s)" in err.message
    assert "Cannot reach Ollama" in err.message
    assert "Piper voice model not found" in err.message
    assert "FFmpeg binary missing" in err.message
    assert "ollama serve" in str(err)
    assert "Download voice model" in str(err)
    assert "Install FFmpeg" in str(err)


def test_run_json_contains_expected_fields_and_no_secrets(
    mock_pipeline_environment: dict[str, Any],
) -> None:
    """Verify run.json metadata contains required metrics and omits sensitive secrets."""
    settings: Settings = mock_pipeline_environment["settings"]
    result = run("https://github.com/test-owner/demo-repo", settings=settings)

    run_json = result.work_dir / "run.json"
    assert run_json.is_file()
    raw = run_json.read_text(encoding="utf-8")
    meta = json.loads(raw)

    expected_keys = {
        "tool_version",
        "timestamp",
        "url",
        "provider",
        "model",
        "voice_model",
        "theme",
        "scene_count",
        "total_narration_words",
        "timings",
        "outputs",
    }
    assert expected_keys.issubset(meta.keys())
    assert meta["url"] == "https://github.com/test-owner/demo-repo"
    assert meta["scene_count"] == 5
    assert meta["total_narration_words"] > 0

    # Ensure no secrets leak
    assert "secret_token_12345" not in raw
    assert "secret_key_67890" not in raw
    assert "github_token" not in raw
    assert "openai_api_key" not in raw


def test_render_only_accepts_file_and_folder(
    mock_pipeline_environment: dict[str, Any],
    sample_script_data: ScriptData,
    tmp_path: Path,
) -> None:
    """Verify render_only functions identically given a scenes.json file or its parent folder."""
    settings: Settings = mock_pipeline_environment["settings"]
    work_dir = tmp_path / "custom_work_dir"
    work_dir.mkdir(parents=True, exist_ok=True)
    scenes_path = work_dir / "scenes.json"
    sample_script_data.save(scenes_path)

    # 1. Test passing scenes.json directly
    res1 = render_only(
        scenes_json=scenes_path,
        repo_label="test-owner/demo-repo",
        settings=settings,
    )
    assert res1.video is not None and res1.video.is_file()
    assert "voice" in res1.timings
    assert "visuals" in res1.timings
    assert "render" in res1.timings

    # 2. Test passing the folder containing scenes.json
    res2 = render_only(
        scenes_json=work_dir,
        repo_label="test-owner/demo-repo",
        settings=settings,
    )
    assert res2.video is not None and res2.video.is_file()


def test_render_only_missing_file_raises(
    mock_pipeline_environment: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Verify render_only raises Repo2VideoError when scenes.json does not exist."""
    settings: Settings = mock_pipeline_environment["settings"]
    missing = tmp_path / "nonexistent" / "scenes.json"

    with pytest.raises(Repo2VideoError) as exc_info:
        render_only(scenes_json=missing, repo_label="owner/repo", settings=settings)

    assert "Scenes file not found" in exc_info.value.message


def test_pair_scene_assets_validation(
    sample_script_data: ScriptData,
    tmp_path: Path,
) -> None:
    """Verify asset pairing detects count mismatches and missing scene IDs."""
    slides: list[Path] = []
    audios: list[SceneAudio] = []
    for sc in sample_script_data.scenes:
        p = tmp_path / f"scene_{sc.id:02d}.png"
        p.write_bytes(b"PNG")
        slides.append(p)
        a = tmp_path / f"scene_{sc.id:02d}.wav"
        a.write_bytes(b"WAV")
        audios.append(SceneAudio(scene_id=sc.id, path=a, duration_seconds=3.0))

    # Valid pairing
    assets = _pair_scene_assets(sample_script_data, slides, audios)
    assert len(assets) == 5
    for i, asset in enumerate(assets, start=1):
        assert asset.scene_id == i

    # Count mismatch raises error
    with pytest.raises(Repo2VideoError) as exc_info:
        _pair_scene_assets(sample_script_data, slides[:-1], audios)
    assert "Mismatch in scene assets count" in exc_info.value.message

    # Scene ID mismatch raises error
    bad_audios = list(audios)
    bad_audios[1] = SceneAudio(scene_id=99, path=tmp_path / "scene_99.wav", duration_seconds=2.0)
    with pytest.raises(Repo2VideoError) as exc_info:
        _pair_scene_assets(sample_script_data, slides, bad_audios)
    assert "missing slide or audio for scene 2" in exc_info.value.message


def test_check_doctor_all_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify check_doctor returns all OK status when all prerequisites pass."""
    monkeypatch.setenv("GITHUB_TOKEN", "tok_123")
    settings = Settings(output_dir=tmp_path / "out", github_token="tok_123")

    class ReadyLLM:
        def check_ready(self) -> None:
            pass

    class ReadyPiper:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("repo2video.pipeline.get_client", lambda s: ReadyLLM())
    monkeypatch.setattr("repo2video.pipeline.PiperEngine", ReadyPiper)
    monkeypatch.setattr("repo2video.pipeline.check_ffmpeg", lambda: "ffmpeg 6.1")
    monkeypatch.setattr("repo2video.pipeline.check_chromium", lambda: "/path/to/chrome")

    checks: list[DoctorCheck] = check_doctor(settings)
    assert len(checks) == 10
    for c in checks:
        if c.name == "GITHUB_TOKEN":
            assert c.status == "set"
        else:
            assert c.ok is True
            assert c.status == "OK"


def test_check_doctor_mixed_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify check_doctor identifies specific failing tools with their hints."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    settings = Settings(output_dir=tmp_path / "out", github_token=None)

    class FailingLLM:
        def check_ready(self) -> None:
            raise LLMError("Ollama down.", hint="Start Ollama")

    class ReadyPiper:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("repo2video.pipeline.get_client", lambda s: FailingLLM())
    monkeypatch.setattr("repo2video.pipeline.PiperEngine", ReadyPiper)
    monkeypatch.setattr("repo2video.pipeline.check_ffmpeg", lambda: "ffmpeg 6.1")
    monkeypatch.setattr("repo2video.pipeline.check_chromium", lambda: "/path/to/chrome")

    checks = check_doctor(settings)
    llm_check = next(c for c in checks if "LLM" in c.name)
    assert llm_check.ok is False
    assert llm_check.status == "FAIL"
    assert llm_check.hint == "Start Ollama"

    token_check = next(c for c in checks if c.name == "GITHUB_TOKEN")
    assert token_check.status == "not set"
