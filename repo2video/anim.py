"""Deterministic animation rendering engine using Playwright and FFmpeg.

THE PAGE CONTRACT:
An animated page is a complete HTML string that defines `window.__setTime(t)`,
a function taking seconds as a number and setting the entire visual state for
that moment. Calling it twice with the same `t` must give an identical picture.
Before the page's scripts run, the engine injects `window.__DURATION` (seconds,
number) and `window.__FPS` (number) using Playwright's `add_init_script`.

Rules for page authors (enforced where possible by the engine):
1. No setTimeout, setInterval, or requestAnimationFrame loops.
2. No CSS animations or transitions (the engine injects a style rule that sets
   `animation: none !important; transition: none !important;` on every element).
3. No Math.random (use the seeded generator `rand(seed)` from the runtime).
4. No Date.now or performance.now.
5. All visual motion must be computed as a pure function of `t`.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

from repo2video.config import Settings, get_settings
from repo2video.errors import RenderError, SceneError

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_CSS",
    "ENGINE_VERSION",
    "RUNTIME_JS",
    "AnimationRenderer",
    "build_ffmpeg_cmd",
    "compute_cache_key",
    "get_frame_times",
    "make_selftest_html",
    "wrap_page",
]

ENGINE_VERSION: int = 1

BASE_CSS: str = """*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  animation: none !important;
  transition: none !important;
}
html, body {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  margin: 0;
  padding: 0;
  background: transparent;
  -webkit-font-smoothing: antialiased;
}
"""

RUNTIME_JS: str = """function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

function lerp(a, b, k) {
  return a + (b - a) * k;
}

function progress(t, start, end) {
  if (start === end) return t >= end ? 1 : 0;
  return clamp((t - start) / (end - start), 0, 1);
}

const ease = {
  inOutCubic: function(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  },
  outCubic: function(t) {
    return 1 - Math.pow(1 - t, 3);
  },
  outQuart: function(t) {
    return 1 - Math.pow(1 - t, 4);
  },
  inOutSine: function(t) {
    return -(Math.cos(Math.PI * t) - 1) / 2;
  }
};

function typed(text, t, start, cps) {
  const elapsed = Math.max(0, t - start);
  const chars = Math.floor(elapsed * (cps || 20));
  const visible_chars = clamp(chars, 0, text.length);
  return {
    text: text.slice(0, visible_chars),
    done: visible_chars >= text.length,
    visible_chars: visible_chars
  };
}

function blink(t, hz) {
  const rate = hz || 1;
  return Math.floor(Math.max(0, t) * rate * 2) % 2 === 0;
}

function rand(seed) {
  let s = (typeof seed === "number" ? seed : 12345) >>> 0;
  return function() {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fmtInt(n) {
  return Math.round(n).toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, ",");
}
"""


def wrap_page(body_html: str, scene_js: str, css: str = "", theme_css: str = "") -> str:
    """Build a complete HTML document for an animated scene.

    Includes BASE_CSS, theme_css, scene-specific css, body_html, RUNTIME_JS,
    and the scene script. Canvas size is fixed and overflow is hidden.

    Args:
        body_html: Markup to place in the <body> tag.
        scene_js: JavaScript code defining window.__setTime(t) and scene logic.
        css: Scene-specific CSS rules.
        theme_css: Global theme CSS rules.

    Returns:
        Complete HTML document string.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
{BASE_CSS}
{theme_css}
{css}
</style>
</head>
<body>
{body_html}
<script>
{RUNTIME_JS}
</script>
<script>
{scene_js}
</script>
</body>
</html>"""


def compute_cache_key(
    html: str,
    duration: float,
    fps: int,
    width: int,
    height: int,
    quality: str,
    version: int = ENGINE_VERSION,
) -> str:
    """Compute sha256 cache key for animation render parameters.

    Args:
        html: Complete HTML string of the page.
        duration: Duration in seconds rounded to 3 decimal places.
        fps: Animation frame rate.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        quality: Quality preset ('normal' or 'fast').
        version: Animation engine version.

    Returns:
        Hex-encoded sha256 hash string.
    """
    payload = f"{version}|{html}|{duration:.3f}|{fps}|{width}x{height}|{quality}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_frame_times(duration: float, fps: int) -> list[float]:
    """Calculate the deterministic timestamp t (in seconds) for each frame.

    Frame count is max(2, round(duration * fps)).
    Frame i uses t = min(i / fps, duration), with the final frame reaching exact duration.

    Args:
        duration: Scene duration in seconds.
        fps: Animation frame rate.

    Returns:
        List of timestamp floats.
    """
    count = max(2, round(duration * fps))
    times: list[float] = []
    for i in range(count):
        if i == count - 1:
            times.append(float(duration))
        else:
            times.append(min(i / fps, duration))
    return times


def build_ffmpeg_cmd(
    out_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
    crf: int,
    video_fps: int,
) -> list[str]:
    """Construct the FFmpeg command argument list for encoding image2pipe stream.

    Args:
        out_path: Target video file path.
        fps: Input animation frame rate.
        width: Output video width in pixels.
        height: Output video height in pixels.
        crf: Constant Rate Factor value for libx264.
        video_fps: Target video output frame rate.

    Returns:
        List of command line argument strings.
    """
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-vcodec",
        "mjpeg",
        "-i",
        "-",
        "-an",
        "-vf",
        f"scale={width}:{height}:flags=lanczos,setsar=1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(video_fps),
        str(out_path),
    ]


def make_selftest_html() -> str:
    """Generate the built-in 3-second demo HTML document for selftest."""
    body_html = (
        '<div id="container" style="position: relative; width: 100%; height: 100%; '
        'background: #0f172a; display: flex; align-items: center;">'
        '<div id="box" style="position: absolute; left: 100px; width: 280px; height: 120px; '
        "background: #3b82f6; border-radius: 20px; display: flex; align-items: center; "
        "justify-content: center; color: #ffffff; font-family: sans-serif; font-size: 36px; "
        'font-weight: bold; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">'
        "0.00s"
        "</div>"
        "</div>"
    )
    scene_js = (
        "window.__setTime = function(t) {\n"
        '  const box = document.getElementById("box");\n'
        "  const p = progress(t, 0, 3);\n"
        "  const x = lerp(100, 1500, p);\n"
        '  box.style.left = x + "px";\n'
        '  box.innerText = t.toFixed(2) + "s";\n'
        "};\n"
    )
    return wrap_page(body_html, scene_js)


class AnimationRenderer:
    """Deterministic animation renderer using Playwright Chromium and FFmpeg."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the renderer with settings.

        Args:
            settings: Application settings, or loaded defaults if omitted.
        """
        self.settings = settings if settings is not None else get_settings()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def __enter__(self) -> Self:
        """Context manager entry point: launch browser and prepare page."""
        self._ensure_browser()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit point: close browser and Playwright."""
        del exc_type, exc_val, exc_tb
        self.close()

    def _ensure_browser(self) -> None:
        """Launch Chromium and configure a clean page with network blocked."""
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            viewport={
                "width": self.settings.video_width,
                "height": self.settings.video_height,
            },
            device_scale_factor=1,
            accept_downloads=False,
            permissions=[],
            service_workers="block",
        )
        self._page = self._context.new_page()

        # Block every network request except data: and about:
        def _block_network(route):
            url = route.request.url.lower()
            if url.startswith(("data:", "about:")):
                route.continue_()
            else:
                route.abort()

        self._page.route("**/*", _block_network)

    def close(self) -> None:
        """Close browser context, browser instance, and Playwright driver."""
        if self._context is not None:
            try:
                self._context.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Browser context close error: %s", exc)
            self._context = None
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Browser close error: %s", exc)
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Playwright stop error: %s", exc)
            self._playwright = None
        self._page = None

    def capture_frame(self, html: str, t: float) -> bytes:
        """Render a single frame at time t (in seconds) and return PNG bytes.

        Useful for unit tests, static snapshots, and scene previews.

        Args:
            html: Complete HTML string of the page.
            t: Timestamp in seconds.

        Returns:
            PNG image bytes.

        Raises:
            SceneError: If window.__setTime is not defined on the page.
        """
        self._ensure_browser()
        assert self._page is not None
        page = self._page

        page.add_init_script("window.__DURATION = 10.0; window.__FPS = 24;")
        page.set_content(html, wait_until="load")
        page.evaluate("() => document.fonts.ready")

        has_set_time = page.evaluate("() => typeof window.__setTime === 'function'")
        if not has_set_time:
            raise SceneError(
                "The animated page does not define window.__setTime.",
                hint="Define window.__setTime(t) in the page scripts to control visual state.",
            )

        page.evaluate(
            """async (t) => {
                window.__setTime(t);
                await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
            }""",
            t,
        )
        return page.screenshot(type="png")

    @staticmethod
    def _probe_duration(path: Path) -> float:
        """Extract video duration in seconds using ffprobe.

        Args:
            path: Path to the video file to inspect.

        Returns:
            Duration in seconds as a float, or 0.0 on failure.
        """
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10, check=False)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
        return 0.0

    def render(
        self,
        html: str,
        duration: float,
        out_path: Path,
        *,
        fps: int | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> Path:
        """Render an animated HTML page into a video file over exact duration seconds.

        Captures one screenshot per frame with Playwright and pipes to FFmpeg.
        Returns the output path on success.

        Args:
            html: Complete HTML string defining window.__setTime(t).
            duration: Duration in seconds.
            out_path: Target video file path.
            fps: Optional override for animation frame rate.
            on_progress: Optional callback invoked with progress fraction (0.0 to 1.0).

        Returns:
            The output Path on success.

        Raises:
            SceneError: If window.__setTime is not defined on the page.
            RenderError: If FFmpeg encoding fails or output duration is inaccurate.
        """
        quality = self.settings.anim_quality
        effective_fps = (
            fps if fps is not None else (15 if quality == "fast" else self.settings.anim_fps)
        )
        w = self.settings.video_width
        h = self.settings.video_height

        cache_key = compute_cache_key(html, duration, effective_fps, w, h, quality)
        key_path = out_path.with_name(f"{out_path.name}.key")

        # Caching check
        if out_path.is_file() and key_path.is_file():
            try:
                stored_key = key_path.read_text(encoding="utf-8").strip()
                if stored_key == cache_key:
                    logger.info(
                        "Animation cache hit for %s (key %s)", out_path.name, cache_key[:12]
                    )
                    return out_path
            except OSError:
                pass

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_browser()
        assert self._page is not None
        page = self._page

        # Inject duration and fps before page scripts run
        page.add_init_script(f"window.__DURATION = {duration}; window.__FPS = {effective_fps};")
        page.set_content(html, wait_until="load")
        page.evaluate("() => document.fonts.ready")

        has_set_time = page.evaluate("() => typeof window.__setTime === 'function'")
        if not has_set_time:
            raise SceneError(
                "The animated page does not define window.__setTime.",
                hint="Define window.__setTime(t) in the page scripts to control visual state.",
            )

        # Call window.__setTime(0) once without taking a screenshot so lazy layout is complete
        page.evaluate("() => window.__setTime(0)")

        times = get_frame_times(duration, effective_fps)
        frame_count = len(times)
        jpeg_quality = 85 if quality == "fast" else 92
        crf = 24 if quality == "fast" else 20
        video_fps = self.settings.video_fps

        temp_out = out_path.parent / f"{out_path.stem}.tmp{out_path.suffix}"
        temp_out.unlink(missing_ok=True)

        cmd = build_ffmpeg_cmd(
            temp_out,
            fps=effective_fps,
            width=w,
            height=h,
            crf=crf,
            video_fps=video_fps,
        )

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

        stderr_lines: list[str] = []

        def drain_stderr():
            if proc.stderr:
                for line in proc.stderr:
                    stderr_lines.append(line.decode("utf-8", errors="replace"))

        drain_thread = threading.Thread(target=drain_stderr, daemon=True)
        drain_thread.start()

        # Time cost on a normal laptop: roughly 20 to 60 milliseconds per frame,
        # so 10 seconds at 24 fps takes about 5 to 15 seconds.
        render_start = time.monotonic()
        try:
            for i, t in enumerate(times):
                page.evaluate(
                    """async (t) => {
                        window.__setTime(t);
                        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
                    }""",
                    t,
                )
                frame_bytes = page.screenshot(type="jpeg", quality=jpeg_quality)

                try:
                    if proc.stdin:
                        proc.stdin.write(frame_bytes)
                except BrokenPipeError:
                    break

                pct = (i + 1) / frame_count
                if i == frame_count // 4:
                    logger.debug("Animation %s progress: 25%%", out_path.name)
                elif i == frame_count // 2:
                    logger.debug("Animation %s progress: 50%%", out_path.name)
                elif i == (3 * frame_count) // 4:
                    logger.debug("Animation %s progress: 75%%", out_path.name)

                if on_progress and ((i + 1) % 10 == 0 or i == frame_count - 1):
                    on_progress(pct)

            if proc.stdin:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

            proc.wait(timeout=60)
            drain_thread.join(timeout=5)

            stderr_text = "".join(stderr_lines)
            if proc.returncode != 0:
                temp_out.unlink(missing_ok=True)
                last_err = (
                    stderr_text[-600:]
                    if stderr_text
                    else f"FFmpeg exited with code {proc.returncode}"
                )
                raise RenderError(
                    f"FFmpeg failed while rendering animation {out_path.name}: {last_err}",
                    hint="Check FFmpeg installation and video encoding parameters.",
                )

            # Check duration with ffprobe
            actual_duration = self._probe_duration(temp_out)
            tolerance = max(1.0 / effective_fps, 1.0 / video_fps) + 0.05
            if abs(actual_duration - duration) > tolerance:
                temp_out.unlink(missing_ok=True)
                last_err = stderr_text[-600:]
                raise RenderError(
                    f"Rendered animation duration {actual_duration:.2f}s differs from requested "
                    f"{duration:.2f}s by more than 1 frame (tolerance {tolerance:.2f}s). {last_err}",
                    hint="Check frame rate calculations and FFmpeg parameters.",
                )

            # Atomic rename & cache key storage
            temp_out.replace(out_path)
            key_path.write_text(cache_key, encoding="utf-8")

            total_time = time.monotonic() - render_start
            fps_achieved = frame_count / total_time if total_time > 0 else 0.0
            file_size = out_path.stat().st_size
            logger.info(
                "Animation rendered: %s (%d frames in %.2fs, %.1f fps, %d bytes)",
                out_path.name,
                frame_count,
                total_time,
                fps_achieved,
                file_size,
            )
            return out_path

        except (KeyboardInterrupt, Exception):
            if proc.poll() is None:
                if proc.stdin:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            drain_thread.join(timeout=1)
            temp_out.unlink(missing_ok=True)
            raise


def main() -> None:
    """CLI entry point for animation selftest."""
    if "--selftest" in sys.argv:
        settings = get_settings()
        out_path = settings.output_dir / "anim_selftest.mp4"
        html = make_selftest_html()

        print(f"Running animation selftest (3.0s demo) to {out_path}...")
        t0 = time.monotonic()
        with AnimationRenderer(settings) as renderer:
            renderer.render(html, duration=3.0, out_path=out_path)
        elapsed = time.monotonic() - t0
        print(f"Selftest completed: {out_path} rendered in {elapsed:.2f}s")
    else:
        print("Usage: python -m repo2video.anim --selftest")
        sys.exit(1)


if __name__ == "__main__":
    main()
