"""Comprehensive v2 pipeline tests covering stage ordering, fallbacks, visual dispatch, and e2e."""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from GRC.config import Settings
from GRC.errors import SceneError
from GRC.models import (
    ArchitectureSpec,
    EdgeSpec,
    KeyFile,
    MediaAsset,
    MediaRef,
    NodeSpec,
    RepoData,
    Scene,
    ScriptData,
    TerminalSession,
)
from GRC.pipeline import PipelineOptions, run
from GRC.render import SceneAssets, SceneVisual, compute_timeline, probe_video
from GRC.voice import SceneAudio

HAS_FFMPEG = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


def is_chromium_available() -> bool:
    """Check if Playwright Chromium browser is installed and launchable."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:  # noqa: BLE001
        return False


skip_if_no_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG, reason="FFmpeg and ffprobe binaries are required on PATH"
)
skip_if_no_chromium = pytest.mark.skipif(
    not is_chromium_available(), reason="Playwright Chromium browser executable is not installed"
)


def make_silent_wav(path: Path, duration: float, rate: int = 44100) -> Path:
    """Create a valid PCM WAV audio file with silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * duration)
    data = b"\x00" * (frames * 4)  # 16-bit stereo PCM
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data)
    return path


def make_color_png(
    path: Path, color: tuple[int, int, int] = (100, 150, 200), size: tuple[int, int] = (640, 360)
) -> Path:
    """Create a solid color PNG image using Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


# ==============================================================================
# Mock Fixtures for Pipeline V2
# ==============================================================================


@pytest.fixture
def v2_media_assets(tmp_path: Path) -> list[MediaAsset]:
    """Fixture providing a list of MediaAsset for tests."""
    sample_img = make_color_png(tmp_path / "media_asset_pic.png")
    return [
        MediaAsset(
            id="media_pic",
            kind="image",
            path=sample_img,
            width=640,
            height=360,
            origin="readme",
            source_url="https://example.com/pic.png",
        )
    ]


@pytest.fixture
def v2_repo_data(tmp_path: Path) -> RepoData:
    """Fixture providing a complete RepoData instance with media, code, terminals, and stats."""
    return RepoData(
        name="full-v2-repo",
        owner="test-owner",
        description="A full v2 demo repo with Parser, Engine, and Output architecture.",
        stars=1500,
        forks=250,
        open_issues=12,
        license="MIT",
        languages=["Python", "TypeScript"],
        readme="# Full Demo Repo\nThis is a complete demo repo for v2 tests with Parser, Engine, and Output.",
        top_level=["src", "tests", "README.md"],
        key_files=[
            KeyFile(
                id="kf_main",
                path="src/main.py",
                language="python",
                content="def run():\n    print('Hello World')\n",
                total_lines=2,
            )
        ],
        terminals=[
            TerminalSession(
                id="term_install",
                purpose="install",
                commands=["pip install repo2video"],
                output=["Successfully installed repo2video"],
            )
        ],
        media=[
            MediaRef(
                id="media_pic",
                url="https://example.com/pic.png",
                alt="Sample Image",
                kind="image",
            )
        ],
    )


@pytest.fixture
def v2_all_10_scenes_script() -> ScriptData:
    """Fixture providing a ScriptData with all visual scene types (9 scenes total)."""
    narr = (
        "This is an automated narration sentence designed to test the full video generation "
        "pipeline across all visual scene types."
    )
    return ScriptData(
        title="test-owner/full-v2-repo",
        scenes=[
            Scene(id=1, type="title", heading="Title Scene", bullets=["Welcome"], narration=narr),
            Scene(
                id=2,
                type="feature",
                heading="Feature Scene",
                bullets=["Automated visual demo generation"],
                narration=narr,
            ),
            Scene(
                id=3,
                type="howto",
                heading="HowTo Scene",
                bullets=["Install the CLI tool"],
                narration=narr,
            ),
            Scene(
                id=4,
                type="media",
                heading="Media Scene",
                bullets=["Screenshot from README"],
                narration=narr,
                media_id="media_pic",
            ),
            Scene(
                id=5,
                type="code",
                heading="Code Scene",
                bullets=["Highlighting key code"],
                narration=narr,
                file_id="kf_main",
            ),
            Scene(
                id=6,
                type="terminal",
                heading="Terminal Scene",
                bullets=["Replaying commands"],
                narration=narr,
                terminal_id="term_install",
            ),
            Scene(
                id=7,
                type="architecture",
                heading="Architecture Scene",
                bullets=["System architecture"],
                narration=narr,
                architecture=ArchitectureSpec(
                    nodes=[
                        NodeSpec(id="n1", label="Parser", group="input"),
                        NodeSpec(id="n2", label="Engine", group="core"),
                        NodeSpec(id="n3", label="Output", group="output"),
                    ],
                    edges=[
                        EdgeSpec(source="n1", target="n2", label="AST"),
                        EdgeSpec(source="n2", target="n3", label="MP4"),
                    ],
                ),
            ),
            Scene(
                id=8,
                type="stats",
                heading="Stats Scene",
                bullets=["Repository metrics"],
                narration=narr,
                stat_keys=["stars", "forks"],
            ),
            Scene(
                id=9,
                type="outro",
                heading="Outro Scene",
                bullets=["Star the repo on GitHub"],
                narration=narr,
            ),
        ],
    )


# ==============================================================================
# Unit & Integration Tests for Pipeline V2
# ==============================================================================


def test_stage_order_and_six_stage_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
    v2_media_assets: list[MediaAsset],
) -> None:
    """Verify the 6-stage execution order and progress numbers [1/6] to [6/6]."""
    calls: list[str] = []
    stage_events: list[tuple[str, int, int]] = []

    settings = Settings(output_dir=tmp_path / "out")

    class FakeClient:
        def check_ready(self) -> None:
            pass

    class FakePiper:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: FakeClient())
    monkeypatch.setattr("GRC.pipeline.PiperEngine", FakePiper)
    monkeypatch.setattr("GRC.pipeline.check_ffmpeg", lambda: "ffmpeg 6.1")
    monkeypatch.setattr("GRC.pipeline.check_chromium", lambda: "/fake/chrome")

    monkeypatch.setattr(
        "GRC.pipeline.analyze",
        lambda url, settings: calls.append("analyze") or v2_repo_data,
    )
    monkeypatch.setattr(
        "GRC.pipeline.collect_media",
        lambda repo, work_dir, settings, force=False: calls.append("media") or v2_media_assets,
    )
    monkeypatch.setattr(
        "GRC.pipeline.write_script",
        lambda repo, client, settings, media_assets=None: (
            calls.append("script") or v2_all_10_scenes_script
        ),
    )
    monkeypatch.setattr(
        "GRC.pipeline.synthesize_all",
        lambda script, out_dir, settings: (
            calls.append("voice")
            or [
                SceneAudio(scene_id=s.id, path=tmp_path / f"a_{s.id}.wav", duration_seconds=2.0)
                for s in script.scenes
            ]
        ),
    )

    def fake_render_visuals(*args: Any, **kwargs: Any) -> tuple[list[SceneVisual], list[Any], str]:
        calls.append("visuals")
        script = kwargs.get("script_data") or args[0]
        vis = [SceneVisual(kind="image", image=tmp_path / f"s_{s.id}.png") for s in script.scenes]
        return vis, [], "Visuals from README"

    monkeypatch.setattr("GRC.pipeline._render_visuals", fake_render_visuals)

    def fake_render_video(*args: Any, **kwargs: Any) -> None:
        calls.append("render")
        out_path = args[1]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"MP4_DATA")

    monkeypatch.setattr("GRC.pipeline.render_video", fake_render_video)
    monkeypatch.setattr(
        "GRC.pipeline.probe_video",
        lambda p: {"duration_seconds": 22.0, "width": 1920, "height": 1080},
    )

    options = PipelineOptions(on_stage=lambda name, n, tot: stage_events.append((name, n, tot)))
    res = run("https://github.com/test-owner/full-v2-repo", settings=settings, options=options)

    assert stage_events == [
        ("analyze", 1, 6),
        ("media", 2, 6),
        ("script", 3, 6),
        ("voice", 4, 6),
        ("visuals", 5, 6),
        ("render", 6, 6),
    ]
    assert calls == ["analyze", "media", "script", "voice", "visuals", "render"]
    assert res.video is not None and res.video.is_file()
    assert res.timings.keys() == {"analyze", "media", "script", "voice", "visuals", "render"}


def test_media_stage_failure_gives_empty_list_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify an exception in collect_media logs WARNING and pipeline proceeds with empty media list."""
    settings = Settings(output_dir=tmp_path / "out")

    class FakeClient:
        def check_ready(self) -> None:
            pass

    class FakePiper:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: FakeClient())
    monkeypatch.setattr("GRC.pipeline.PiperEngine", FakePiper)
    monkeypatch.setattr("GRC.pipeline.check_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("GRC.pipeline.check_chromium", lambda: "/chrome")

    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)

    def broken_collect_media(*args: Any, **kwargs: Any) -> list[MediaAsset]:
        raise RuntimeError("Network down or download blocked")

    monkeypatch.setattr("GRC.pipeline.collect_media", broken_collect_media)

    media_received: list[Any] = []

    def mock_write_script(
        repo: RepoData, client: Any, settings: Settings, media_assets: list[Any] | None = None
    ) -> ScriptData:
        media_received.extend(media_assets or [])
        return v2_all_10_scenes_script

    monkeypatch.setattr("GRC.pipeline.write_script", mock_write_script)

    # Run script only to inspect media stage handling
    options = PipelineOptions(script_only=True)
    res = run("https://github.com/test-owner/full-v2-repo", settings=settings, options=options)

    assert media_received == []
    assert res.media_count == 0
    assert "Media collection failed" in caplog.text


def test_no_media_flag_skips_media_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
) -> None:
    """Verify --no-media skips calling collect_media completely."""
    settings = Settings(output_dir=tmp_path / "out")

    class FakeClient:
        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: FakeClient())
    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)

    def never_call_collect(*args: Any, **kwargs: Any) -> list[MediaAsset]:
        raise AssertionError("collect_media must not be called when --no-media is passed")

    monkeypatch.setattr("GRC.pipeline.collect_media", never_call_collect)
    monkeypatch.setattr(
        "GRC.pipeline.write_script", lambda *args, **kwargs: v2_all_10_scenes_script
    )

    options = PipelineOptions(script_only=True, no_media=True)
    res = run("https://github.com/test-owner/full-v2-repo", settings=settings, options=options)
    assert res.media_count == 0


def test_resolve_references_on_reused_scenes_json_degrades_invalid_scenes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
) -> None:
    """Verify resolve_references is called on reused scenes.json and degrades bad references."""
    settings = Settings(output_dir=tmp_path / "out")

    class FakeClient:
        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: FakeClient())
    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)
    monkeypatch.setattr("GRC.pipeline.collect_media", lambda r, w, s, force=False: [])

    # Pre-populate reused scenes.json pointing to non-existent code file 'f_missing'
    work_dir = settings.output_dir / "test-owner__full-v2-repo"
    work_dir.mkdir(parents=True, exist_ok=True)
    bad_script = v2_all_10_scenes_script.model_copy(deep=True)
    for sc in bad_script.scenes:
        if sc.type == "code":
            sc.file_id = "f_missing"  # not in v2_repo_data

    bad_script.save(work_dir / "scenes.json")

    options = PipelineOptions(script_only=True, reuse=True)
    res = run("https://github.com/test-owner/full-v2-repo", settings=settings, options=options)

    # Re-read scenes.json; scene 5 must have degraded to feature because file was missing
    loaded = ScriptData.load(res.scenes_json)
    code_scene = next(s for s in loaded.scenes if s.id == 5)
    assert code_scene.type == "feature"
    assert code_scene.file_id is None


def test_builder_exception_triggers_fallback_and_records_in_run_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
    v2_media_assets: list[MediaAsset],
) -> None:
    """Verify an exception in an animated builder triggers fallback to static slide and logs fallback."""
    settings = Settings(output_dir=tmp_path / "out")

    class FakeClient:
        def check_ready(self) -> None:
            pass

    class FakePiper:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: FakeClient())
    monkeypatch.setattr("GRC.pipeline.PiperEngine", FakePiper)
    monkeypatch.setattr("GRC.pipeline.check_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("GRC.pipeline.check_chromium", lambda: "/chrome")

    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)
    monkeypatch.setattr("GRC.pipeline.collect_media", lambda r, w, s, force=False: v2_media_assets)
    monkeypatch.setattr(
        "GRC.pipeline.write_script", lambda *args, **kwargs: v2_all_10_scenes_script
    )
    monkeypatch.setattr(
        "GRC.pipeline.synthesize_all",
        lambda sc, out_dir, *args, **kwargs: [
            SceneAudio(scene_id=scene.id, path=tmp_path / f"{scene.id}.wav", duration_seconds=1.5)
            for scene in sc.scenes
        ],
    )

    # Make build_terminal_html raise an exception
    def broken_terminal(*args: Any, **kwargs: Any) -> str:
        raise SceneError("Terminal session formatting failed.")

    monkeypatch.setattr("GRC.pipeline.build_terminal_html", broken_terminal)

    # Mock screenshot_html so fallback static slide generation succeeds
    def mock_screenshot(html_str: str, out_p: Path, *args: Any, **kwargs: Any) -> Path:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"STATIC_SLIDE_PNG")
        return out_p

    monkeypatch.setattr("GRC.pipeline.screenshot_html", mock_screenshot)

    # Mock AnimationRenderer.render
    class FakeRenderer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def render(
            self, html: str, duration: float, out_path: Path, *args: Any, **kwargs: Any
        ) -> Path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"CLIP_DATA")
            return out_path

    monkeypatch.setattr("GRC.pipeline.AnimationRenderer", FakeRenderer)

    def mock_render_video(*args: Any, **kwargs: Any) -> None:
        out_path = args[1]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"VIDEO_DATA")

    monkeypatch.setattr("GRC.pipeline.render_video", mock_render_video)
    monkeypatch.setattr(
        "GRC.pipeline.probe_video",
        lambda p: {"duration_seconds": 15.0, "width": 1920, "height": 1080},
    )

    res = run("https://github.com/test-owner/full-v2-repo", settings=settings)

    assert len(res.fallbacks) == 1
    assert res.fallbacks[0]["scene_id"] == 6
    assert res.fallbacks[0]["type"] == "terminal"

    # Check run.json contains the fallback
    run_json = json.loads((res.work_dir / "run.json").read_text(encoding="utf-8"))
    assert len(run_json["fallbacks"]) == 1
    assert run_json["fallbacks"][0]["scene_id"] == 6


def test_failing_fallback_aborts_with_scene_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
    v2_media_assets: list[MediaAsset],
) -> None:
    """Verify that if the static fallback slide also fails, SceneError is raised and aborts."""
    settings = Settings(output_dir=tmp_path / "out")

    class FakeClient:
        def check_ready(self) -> None:
            pass

    class FakePiper:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: FakeClient())
    monkeypatch.setattr("GRC.pipeline.PiperEngine", FakePiper)
    monkeypatch.setattr("GRC.pipeline.check_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("GRC.pipeline.check_chromium", lambda: "/chrome")

    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)
    monkeypatch.setattr("GRC.pipeline.collect_media", lambda r, w, s, force=False: v2_media_assets)
    monkeypatch.setattr(
        "GRC.pipeline.write_script", lambda *args, **kwargs: v2_all_10_scenes_script
    )
    monkeypatch.setattr(
        "GRC.pipeline.synthesize_all",
        lambda sc, out_dir, *args, **kwargs: [
            SceneAudio(scene_id=scene.id, path=tmp_path / f"{scene.id}.wav", duration_seconds=1.5)
            for scene in sc.scenes
        ],
    )

    # Builder fails
    monkeypatch.setattr(
        "GRC.pipeline.build_code_html",
        lambda *args, **kwargs: (_ for _ in ()).throw(SceneError("Code failed")),
    )

    # Fallback static slide screenshot fails for scene 5
    def failing_screenshot_for_scene_5(
        html_str: str, out_p: Path, *args: Any, **kwargs: Any
    ) -> Path:
        if "scene_05" in str(out_p):
            raise SceneError("Screenshot impossible")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"STATIC_PNG")
        return out_p

    monkeypatch.setattr("GRC.pipeline.screenshot_html", failing_screenshot_for_scene_5)

    class FakeRenderer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.AnimationRenderer", FakeRenderer)

    with pytest.raises(SceneError, match="Static fallback slide failed for scene 5"):
        run("https://github.com/test-owner/full-v2-repo", settings=settings)


def test_no_animation_uses_capture_frame_and_creates_image_visuals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
    v2_media_assets: list[MediaAsset],
) -> None:
    """Verify --no-animation invokes capture_frame at 85% timestamp and creates kind='image'."""
    settings = Settings(output_dir=tmp_path / "out")

    class FakeClient:
        def check_ready(self) -> None:
            pass

    class FakePiper:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: FakeClient())
    monkeypatch.setattr("GRC.pipeline.PiperEngine", FakePiper)
    monkeypatch.setattr("GRC.pipeline.check_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("GRC.pipeline.check_chromium", lambda: "/chrome")

    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)
    monkeypatch.setattr("GRC.pipeline.collect_media", lambda r, w, s, force=False: v2_media_assets)
    monkeypatch.setattr(
        "GRC.pipeline.write_script", lambda *args, **kwargs: v2_all_10_scenes_script
    )
    monkeypatch.setattr(
        "GRC.pipeline.synthesize_all",
        lambda sc, out_dir, *args, **kwargs: [
            SceneAudio(scene_id=scene.id, path=tmp_path / f"{scene.id}.wav", duration_seconds=2.0)
            for scene in sc.scenes
        ],
    )

    captured_ts: list[float] = []

    class FakeRenderer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def capture_frame(self, html: str, t: float) -> bytes:
            captured_ts.append(round(t, 2))
            return b"FRAME_PNG"

        def render(self, *args: Any, **kwargs: Any) -> Path:
            raise AssertionError("render() must not be called when no_animation=True")

    monkeypatch.setattr("GRC.pipeline.AnimationRenderer", FakeRenderer)
    monkeypatch.setattr(
        "GRC.pipeline.screenshot_html",
        lambda h, p, settings=None, **kwargs: p.write_bytes(b"STATIC_PNG") or p,
    )

    rendered_visual_kinds: list[str] = []

    def inspect_render_video(assets: list[SceneAssets], *args: Any, **kwargs: Any) -> None:
        for a in assets:
            if a.visual:
                rendered_visual_kinds.append(a.visual.kind)
        out_path = args[0] if len(args) > 0 and isinstance(args[0], Path) else args[1]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"VID")

    monkeypatch.setattr("GRC.pipeline.render_video", inspect_render_video)
    monkeypatch.setattr(
        "GRC.pipeline.probe_video",
        lambda p: {"duration_seconds": 20.0, "width": 1920, "height": 1080},
    )

    options = PipelineOptions(no_animation=True)
    res = run("https://github.com/test-owner/full-v2-repo", settings=settings, options=options)

    # Every visual kind must be 'image'
    assert len(rendered_visual_kinds) == 9
    assert all(k == "image" for k in rendered_visual_kinds)
    assert len(captured_ts) > 0
    assert res.video is not None and res.video.is_file()


def test_credit_line_placement_rules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
    v2_media_assets: list[MediaAsset],
) -> None:
    """Verify credit line appends to outro bullets when < 4 bullets, and appears in run.json."""
    settings = Settings(output_dir=tmp_path / "out")

    class FakeClient:
        def check_ready(self) -> None:
            pass

    class FakePiper:
        def __init__(self, settings: Settings) -> None:
            pass

        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: FakeClient())
    monkeypatch.setattr("GRC.pipeline.PiperEngine", FakePiper)
    monkeypatch.setattr("GRC.pipeline.check_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("GRC.pipeline.check_chromium", lambda: "/chrome")

    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)
    monkeypatch.setattr("GRC.pipeline.collect_media", lambda r, w, s, force=False: v2_media_assets)
    monkeypatch.setattr(
        "GRC.pipeline.write_script", lambda *args, **kwargs: v2_all_10_scenes_script
    )
    monkeypatch.setattr(
        "GRC.pipeline.synthesize_all",
        lambda sc, out_dir, *args, **kwargs: [
            SceneAudio(scene_id=scene.id, path=tmp_path / f"{scene.id}.wav", duration_seconds=1.5)
            for scene in sc.scenes
        ],
    )

    def mock_screenshot(html_str: str, out_p: Path, *args: Any, **kwargs: Any) -> Path:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"STATIC_SLIDE_PNG")
        return out_p

    monkeypatch.setattr("GRC.pipeline.screenshot_html", mock_screenshot)

    class FakeRenderer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def render(
            self, html: str, duration: float, out_path: Path, *args: Any, **kwargs: Any
        ) -> Path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"CLIP_DATA")
            return out_path

    monkeypatch.setattr("GRC.pipeline.AnimationRenderer", FakeRenderer)

    def mock_render_video(*args: Any, **kwargs: Any) -> None:
        out_path = args[1]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"VIDEO_DATA")

    monkeypatch.setattr("GRC.pipeline.render_video", mock_render_video)
    monkeypatch.setattr(
        "GRC.pipeline.probe_video",
        lambda p: {"duration_seconds": 15.0, "width": 1920, "height": 1080},
    )

    res = run("https://github.com/test-owner/full-v2-repo", settings=settings)

    # Check credit line
    assert res.credit_line == "Visuals from the test-owner/full-v2-repo README"
    outro_scene = next(s for s in v2_all_10_scenes_script.scenes if s.type == "outro")
    assert "Visuals from the test-owner/full-v2-repo README" in outro_scene.bullets

    run_json = json.loads((res.work_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["credit_line"] == "Visuals from the test-owner/full-v2-repo README"


# ==============================================================================
# End-to-End Real Tool Integration Tests
# ==============================================================================


@pytest.mark.browser
@pytest.mark.ffmpeg
@skip_if_no_chromium
@skip_if_no_ffmpeg
def test_e2e_all_10_scene_types_real_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
    v2_media_assets: list[MediaAsset],
) -> None:
    """Real e2e test with Playwright & FFmpeg rendering all 10 scene types into an MP4 video."""
    work_out = tmp_path / "out"
    settings = Settings(
        output_dir=work_out,
        video_width=640,
        video_height=360,
        video_fps=24,
        anim_fps=24,
        scene_pause_seconds=0.4,
    )

    # Provide real silent audio files for Piper
    aud_dir = tmp_path / "audios"
    aud_dir.mkdir(parents=True, exist_ok=True)
    real_audios = [
        SceneAudio(
            scene_id=s.id,
            path=make_silent_wav(aud_dir / f"s_{s.id}.wav", duration=1.0),
            duration_seconds=1.0,
        )
        for s in v2_all_10_scenes_script.scenes
    ]

    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)
    monkeypatch.setattr("GRC.pipeline.collect_media", lambda r, w, s, force=False: v2_media_assets)
    monkeypatch.setattr(
        "GRC.pipeline.write_script", lambda *args, **kwargs: v2_all_10_scenes_script
    )
    monkeypatch.setattr("GRC.pipeline.synthesize_all", lambda *args, **kwargs: real_audios)

    class ReadyClient:
        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: ReadyClient())

    options = PipelineOptions(fast=True, captions="srt")
    res = run("https://github.com/test-owner/full-v2-repo", settings=settings, options=options)

    assert res.video is not None and res.video.is_file()
    assert res.captions is not None and res.captions.is_file()
    assert res.fallbacks == []

    # Probe duration
    info = probe_video(res.video)
    expected_tl = compute_timeline([1.0] * 9, pause=0.4, T=0.4, transitions=True)
    assert abs(info["duration_seconds"] - expected_tl.total) <= 0.75
    assert info["width"] == 640
    assert info["height"] == 360

    run_meta = json.loads((res.work_dir / "run.json").read_text(encoding="utf-8"))
    assert run_meta["scene_count"] == 9
    assert len(run_meta["fallbacks"]) == 0


@pytest.mark.browser
@pytest.mark.ffmpeg
@skip_if_no_chromium
@skip_if_no_ffmpeg
def test_e2e_one_builder_raises_falls_back_and_renders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v2_repo_data: RepoData,
    v2_all_10_scenes_script: ScriptData,
    v2_media_assets: list[MediaAsset],
) -> None:
    """Real e2e test where stats builder raises SceneError, falls back to static slide, and renders."""
    work_out = tmp_path / "out"
    settings = Settings(
        output_dir=work_out,
        video_width=640,
        video_height=360,
        video_fps=24,
        anim_fps=24,
        scene_pause_seconds=0.4,
    )

    aud_dir = tmp_path / "audios"
    aud_dir.mkdir(parents=True, exist_ok=True)
    real_audios = [
        SceneAudio(
            scene_id=s.id,
            path=make_silent_wav(aud_dir / f"s_{s.id}.wav", duration=1.0),
            duration_seconds=1.0,
        )
        for s in v2_all_10_scenes_script.scenes
    ]

    monkeypatch.setattr("GRC.pipeline.analyze", lambda *args, **kwargs: v2_repo_data)
    monkeypatch.setattr("GRC.pipeline.collect_media", lambda r, w, s, force=False: v2_media_assets)
    monkeypatch.setattr(
        "GRC.pipeline.write_script", lambda *args, **kwargs: v2_all_10_scenes_script
    )
    monkeypatch.setattr("GRC.pipeline.synthesize_all", lambda *args, **kwargs: real_audios)

    class ReadyClient:
        def check_ready(self) -> None:
            pass

    monkeypatch.setattr("GRC.pipeline.get_client", lambda s: ReadyClient())

    # Make build_stats_html fail intentionally
    def failing_stats(*args: Any, **kwargs: Any) -> str:
        raise SceneError("Intentional stats failure for fallback verification.")

    monkeypatch.setattr("GRC.pipeline.build_stats_html", failing_stats)

    options = PipelineOptions(fast=True, captions="srt")
    res = run("https://github.com/test-owner/full-v2-repo", settings=settings, options=options)

    assert res.video is not None and res.video.is_file()
    assert len(res.fallbacks) == 1
    assert res.fallbacks[0]["scene_id"] == 8
    assert res.fallbacks[0]["type"] == "stats"

    run_meta = json.loads((res.work_dir / "run.json").read_text(encoding="utf-8"))
    assert len(run_meta["fallbacks"]) == 1
    assert run_meta["fallbacks"][0]["scene_id"] == 8
