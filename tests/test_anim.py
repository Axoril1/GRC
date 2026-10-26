"""Tests for repo2video/anim.py animation engine."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from GRC.anim import (
    BASE_CSS,
    RUNTIME_JS,
    AnimationRenderer,
    build_ffmpeg_cmd,
    compute_cache_key,
    get_frame_times,
    make_selftest_html,
    wrap_page,
)
from GRC.config import Settings
from GRC.errors import RenderError, SceneError


def is_chromium_available() -> bool:
    """Check if the Playwright Chromium browser is installed and launchable."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except (PlaywrightError, OSError, RuntimeError):
        return False


skip_if_no_chromium = pytest.mark.skipif(
    not is_chromium_available(),
    reason="Playwright Chromium browser executable is not installed",
)


# ==============================================================================
# Non-Browser Unit Tests
# ==============================================================================


class TestAnimNonBrowser:
    """Fast offline unit tests that do not require Playwright or Chromium."""

    def test_wrap_page_structure(self) -> None:
        """Verify wrap_page generates valid HTML structure with all required parts."""
        body = "<div id='content'>Content Here</div>"
        scene_js = "window.__setTime = function(t) { console.log(t); };"
        css = ".my-class { color: red; }"
        theme_css = ":root { --accent: #22c55e; }"

        html = wrap_page(body, scene_js, css=css, theme_css=theme_css)

        assert "<!DOCTYPE html>" in html
        assert '<html lang="en">' in html
        assert BASE_CSS in html
        assert "animation: none !important;" in html
        assert "transition: none !important;" in html
        assert theme_css in html
        assert css in html
        assert body in html
        assert RUNTIME_JS in html
        assert scene_js in html

    def test_runtime_js_contains_all_helpers(self) -> None:
        """Verify RUNTIME_JS contains every specified helper and no real-time APIs."""
        required_helpers = [
            "clamp",
            "lerp",
            "progress",
            "ease",
            "inOutCubic",
            "outCubic",
            "outQuart",
            "inOutSine",
            "typed",
            "blink",
            "rand",
            "fmtInt",
        ]
        for helper in required_helpers:
            assert helper in RUNTIME_JS, f"Helper {helper} missing from RUNTIME_JS"

        # Disallowed real-time APIs
        assert "Date.now" not in RUNTIME_JS
        assert "performance.now" not in RUNTIME_JS
        assert "setInterval" not in RUNTIME_JS
        assert "setTimeout" not in RUNTIME_JS

    def test_cache_key_varies_with_each_input(self) -> None:
        """Verify sha256 cache key changes whenever any render parameter changes."""
        baseline = compute_cache_key("<html>1</html>", 2.0, 24, 1920, 1080, "normal", 1)

        # Vary each parameter
        assert baseline != compute_cache_key("<html>2</html>", 2.0, 24, 1920, 1080, "normal", 1)
        assert baseline != compute_cache_key("<html>1</html>", 2.5, 24, 1920, 1080, "normal", 1)
        assert baseline != compute_cache_key("<html>1</html>", 2.0, 15, 1920, 1080, "normal", 1)
        assert baseline != compute_cache_key("<html>1</html>", 2.0, 24, 1280, 1080, "normal", 1)
        assert baseline != compute_cache_key("<html>1</html>", 2.0, 24, 1920, 720, "normal", 1)
        assert baseline != compute_cache_key("<html>1</html>", 2.0, 24, 1920, 1080, "fast", 1)
        assert baseline != compute_cache_key("<html>1</html>", 2.0, 24, 1920, 1080, "normal", 2)

    def test_ffmpeg_command_construction(self) -> None:
        """Verify FFmpeg command construction for normal and fast quality presets."""
        out_path = Path("/tmp/out.mkv")

        cmd_normal = build_ffmpeg_cmd(
            out_path,
            fps=24,
            width=1920,
            height=1080,
            crf=20,
            video_fps=30,
        )
        assert cmd_normal[0] == "ffmpeg"
        assert "-framerate" in cmd_normal
        assert cmd_normal[cmd_normal.index("-framerate") + 1] == "24"
        assert "-crf" in cmd_normal
        assert cmd_normal[cmd_normal.index("-crf") + 1] == "20"
        assert "-r" in cmd_normal
        assert cmd_normal[cmd_normal.index("-r") + 1] == "30"
        assert "-vf" in cmd_normal
        assert "scale=1920:1080:flags=lanczos,setsar=1" in cmd_normal[cmd_normal.index("-vf") + 1]

        cmd_fast = build_ffmpeg_cmd(
            out_path,
            fps=15,
            width=1280,
            height=720,
            crf=24,
            video_fps=25,
        )
        assert cmd_fast[cmd_fast.index("-framerate") + 1] == "15"
        assert cmd_fast[cmd_fast.index("-crf") + 1] == "24"
        assert cmd_fast[cmd_fast.index("-r") + 1] == "25"
        assert "scale=1280:720:flags=lanczos,setsar=1" in cmd_fast[cmd_fast.index("-vf") + 1]

    def test_frame_count_and_t_values(self) -> None:
        """Verify frame count and t values for 2.0 second, 24 fps render (first 0.0, last 2.0)."""
        times = get_frame_times(2.0, 24)

        assert len(times) == 48
        assert times[0] == 0.0
        assert times[-1] == 2.0
        assert all(t <= 2.0 for t in times)
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_error_mapping_with_fake_ffmpeg_process(self, tmp_path: Path) -> None:
        """Verify RenderError is raised with FFmpeg error output when FFmpeg fails."""
        settings = Settings(output_dir=tmp_path)
        renderer = AnimationRenderer(settings)

        # Mock page to avoid requiring Playwright
        mock_page = MagicMock()
        mock_page.evaluate.return_value = True
        mock_page.screenshot.return_value = b"\xff\xd8\xff\xe0fakejpeg"
        renderer._page = mock_page

        # Mock FFmpeg process with returncode 1
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.poll.return_value = 1
        mock_proc.stderr = [b"Invalid filter specification: error in scale filter"]

        out_video = tmp_path / "failed_scene.mkv"

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            pytest.raises(RenderError) as exc_info,
        ):
            renderer.render(
                "<html><body></body></html>",
                duration=1.0,
                out_path=out_video,
                fps=24,
            )

        assert "FFmpeg failed" in exc_info.value.message
        assert "error in scale filter" in exc_info.value.message
        assert exc_info.value.hint is not None

    def test_keyboard_interrupt_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        """Verify KeyboardInterrupt cleans up and leaves no temporary output file."""
        settings = Settings(output_dir=tmp_path)
        renderer = AnimationRenderer(settings)

        # Mock page screenshot to simulate user pressing Ctrl+C
        mock_page = MagicMock()
        mock_page.evaluate.return_value = True
        mock_page.screenshot.side_effect = KeyboardInterrupt
        renderer._page = mock_page

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr = []

        out_video = tmp_path / "interrupted.mkv"
        temp_video = tmp_path / "interrupted.tmp.mkv"

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            pytest.raises(KeyboardInterrupt),
        ):
            renderer.render(
                "<html><body></body></html>",
                duration=2.0,
                out_path=out_video,
                fps=24,
            )

        assert not temp_video.exists()
        assert not out_video.exists()


# ==============================================================================
# Playwright Browser Integration Tests
# ==============================================================================


@pytest.mark.browser
@skip_if_no_chromium
class TestAnimBrowser:
    """Browser-marked tests verifying frame capture, determinism, and safety."""

    def test_selftest_box_positions_increasing(self) -> None:
        """Verify selftest page moves box horizontally to increasing positions at t=0, 1, 2."""
        renderer = AnimationRenderer()
        html = make_selftest_html()

        bboxes = []
        for t in [0.0, 1.0, 2.0]:
            frame_png = renderer.capture_frame(html, t)
            img = Image.open(io.BytesIO(frame_png))
            # Box is bright blue on dark slate background
            mask = Image.eval(img.convert("L"), lambda p: 255 if p > 50 else 0)
            bbox = mask.getbbox()
            assert bbox is not None, f"Box not found at t={t}"
            bboxes.append(bbox)

        renderer.close()

        # Bounding box left coordinate must strictly increase
        assert bboxes[0][0] < bboxes[1][0] < bboxes[2][0]

    def test_two_captures_at_same_t_are_byte_identical(self) -> None:
        """Verify determinism: two captures at the exact same t yield byte-identical PNGs."""
        renderer = AnimationRenderer()
        html = make_selftest_html()

        capture1 = renderer.capture_frame(html, 1.5)
        capture2 = renderer.capture_frame(html, 1.5)

        renderer.close()

        assert len(capture1) > 0
        assert capture1 == capture2

    def test_remote_network_requests_are_blocked(self) -> None:
        """Verify page scripts cannot reach external network endpoints."""
        renderer = AnimationRenderer()
        html = wrap_page(
            "<div>Test Page</div>",
            """
            window.__setTime = function(t) {};
            window.__BLOCKED = false;
            fetch("https://example.com/api/test")
              .catch(() => { window.__BLOCKED = true; });
            """,
        )

        renderer.capture_frame(html, 0.0)
        assert renderer._page is not None
        # Allow any microtask to settle
        renderer._page.wait_for_timeout(200)
        is_blocked = renderer._page.evaluate("() => window.__BLOCKED")

        renderer.close()
        assert is_blocked is True

    def test_missing_set_time_raises_scene_error(self) -> None:
        """Verify page without window.__setTime raises SceneError."""
        renderer = AnimationRenderer()
        html = "<html><body><h1>No setTime defined</h1></body></html>"

        with pytest.raises(SceneError) as exc_info:
            renderer.capture_frame(html, 0.0)

        renderer.close()
        assert "window.__setTime" in exc_info.value.message
        assert exc_info.value.hint is not None


# ==============================================================================
# FFmpeg and Browser Integration Tests
# ==============================================================================


@pytest.mark.browser
@pytest.mark.ffmpeg
@skip_if_no_chromium
class TestAnimRenderIntegration:
    """Full end-to-end rendering test with Playwright and FFmpeg."""

    def test_render_2s_clip_duration_streams_and_cache(self, tmp_path: Path) -> None:
        """Render a 2-second clip and verify duration, resolution, no audio, and cache reuse."""
        settings = Settings(
            output_dir=tmp_path,
            video_width=1280,
            video_height=720,
            video_fps=30,
            anim_fps=24,
            anim_quality="normal",
        )
        renderer = AnimationRenderer(settings)
        html = make_selftest_html()
        out_video = tmp_path / "test_2s_clip.mkv"

        # 1. First render
        rendered_path = renderer.render(html, duration=2.0, out_path=out_video)
        assert rendered_path == out_video
        assert out_video.exists()
        assert (tmp_path / "test_2s_clip.mkv.key").exists()

        # Probe output with ffprobe
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(out_video),
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)

        # Check duration within 0.1s of requested 2.0s
        dur = float(data.get("format", {}).get("duration", "0"))
        assert abs(dur - 2.0) < 0.1

        # Check canvas dimensions
        streams = data.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        assert len(video_streams) == 1
        assert video_streams[0]["width"] == 1280
        assert video_streams[0]["height"] == 720

        # Check silent: no audio streams
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        assert len(audio_streams) == 0

        # 2. Second call must reuse cache
        # Spy on screenshot to ensure no new frames are captured
        with patch.object(renderer._page, "screenshot", side_effect=RuntimeError("Cache bypassed")):
            cached_path = renderer.render(html, duration=2.0, out_path=out_video)
            assert cached_path == out_video

        renderer.close()
