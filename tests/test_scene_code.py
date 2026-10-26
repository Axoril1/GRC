"""Tests for repo2video/scene_code.py code scene rendering."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from repo2video.anim import AnimationRenderer
from repo2video.models import FocusRange, KeyFile, Scene
from repo2video.scene_code import (
    build_code_html,
    format_file_title,
    highlight_code_to_lines,
    step_windows,
)


def is_chromium_available() -> bool:
    """Check if Playwright Chromium browser is installed and launchable."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            b.close()
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


def test_build_code_html_escaped_heading() -> None:
    """Verify scene heading with special characters is HTML escaped."""
    scene = Scene(
        id=1,
        type="code",
        file_id="f1",
        heading='Danger <script>alert("xss")</script> & "quotes"',
        narration="Here is a detailed explanation of the code snippet demonstrating the key implementation details of our project.",
    )
    kf = KeyFile(
        id="f1",
        path="src/main.py",
        language="python",
        content="print('hello')",
        total_lines=1,
    )
    html_out = build_code_html(scene, kf, duration=4.0)

    assert "<script>alert" not in html_out
    assert (
        "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt; &amp; &quot;quotes&quot;" in html_out
    )


def test_code_snippet_escaping() -> None:
    """Verify code snippet containing HTML tags is escaped and never executable."""
    scene = Scene(
        id=1,
        type="code",
        file_id="f1",
        heading="Code with tags",
        narration="Here is a detailed explanation of the code snippet demonstrating the key implementation details of our project.",
    )
    dangerous_code = "<div>test</div>\nif a < b and c > d:\n    return '&' + '\"'"
    kf = KeyFile(
        id="f1",
        path="src/template.html",
        language="html",
        content=dangerous_code,
        total_lines=3,
    )
    html_out = build_code_html(scene, kf, duration=5.0)

    assert "<div>test</div>" not in html_out
    assert "&lt;" in html_out
    assert "&gt;" in html_out
    assert "&amp;" in html_out


def test_multiline_token_spans_do_not_cross_lines_python() -> None:
    """Verify multi-line tokens in Python never produce spans crossing line boundaries."""
    py_code = '"""Line 1\nLine 2\nLine 3"""\ndef hello():\n    # Multi-line comment\n    # second line\n    return 42'
    lines, _css = highlight_code_to_lines(py_code, "python", "test.py", "dark")

    assert len(lines) == 7
    for i, line_html in enumerate(lines):
        # Count opening and closing spans on each individual line
        open_count = line_html.count("<span")
        close_count = line_html.count("</span>")
        assert open_count == close_count, f"Line {i} has unbalanced spans: {line_html}"


def test_multiline_token_spans_do_not_cross_lines_javascript() -> None:
    """Verify multi-line comments and template strings in JS do not cross line boundaries."""
    js_code = "/* Multi-line\n   comment\n   block */\nconst msg = `hello\nworld`;"
    lines, _css = highlight_code_to_lines(js_code, "javascript", "app.js", "dark")

    assert len(lines) == 5
    for i, line_html in enumerate(lines):
        open_count = line_html.count("<span")
        close_count = line_html.count("</span>")
        assert open_count == close_count, f"Line {i} has unbalanced spans: {line_html}"


def test_unknown_language_fallback() -> None:
    """Verify unknown language falls back to plain text lexer without error."""
    content = "unknown syntax line 1\nunknown syntax line 2"
    lines, _css = highlight_code_to_lines(content, "non_existent_lang_12345", "nofile", "light")

    assert len(lines) == 2
    assert "unknown syntax line 1" in lines[0]
    assert "unknown syntax line 2" in lines[1]


def test_format_file_title() -> None:
    """Verify long file paths are truncated from the left with leading ellipsis."""
    short_path = "repo/utils.py"
    assert format_file_title(short_path) == "repo/utils.py"

    long_path = "a" * 70 + "/file.py"
    formatted = format_file_title(long_path)
    assert len(formatted) == 60
    assert formatted.startswith("…")
    assert formatted.endswith("/file.py")


def test_step_windows_boundaries_and_compression() -> None:
    """Verify step_windows computes correct boundaries, clips to lines, and compresses."""
    # 2 steps over 10.0 seconds with total 15 lines
    steps = [FocusRange(start=2, end=5), FocusRange(start=8, end=14)]
    windows = step_windows(10.0, steps, snippet_total_lines=12)

    assert len(windows) == 2
    # Step 0: 12% to 52% = 1.2s to 5.2s
    assert windows[0]["t_start"] == 1.2
    assert windows[0]["t_end"] == 5.2
    assert windows[0]["start"] == 2
    assert windows[0]["end"] == 5

    # Step 1: 52% to 92% = 5.2s to 9.2s, end clipped to 12
    assert windows[1]["t_start"] == 5.2
    assert windows[1]["t_end"] == 9.2
    assert windows[1]["start"] == 8
    assert windows[1]["end"] == 12

    # Short duration drops trailing steps when time is insufficient
    windows_short = step_windows(
        1.2,
        [
            FocusRange(start=1, end=2),
            FocusRange(start=3, end=4),
            FocusRange(start=5, end=6),
        ],
    )
    assert len(windows_short) < 3


# ==============================================================================
# Browser-Marked Integration Tests
# ==============================================================================


@pytest.mark.browser
@skip_if_no_chromium
def test_code_scene_dims_non_active_lines() -> None:
    """Verify active focus range lines have full brightness while inactive lines are dimmed."""
    lines_content = "\n".join(f"line_{i} = calculate_value({i})" for i in range(1, 11))
    kf = KeyFile(
        id="f1",
        path="src/calc.py",
        language="python",
        content=lines_content,
        total_lines=10,
    )
    # Focus lines 2 to 3
    scene = Scene(
        id=1,
        type="code",
        file_id="f1",
        heading="Calculating Values",
        narration="Here is a detailed explanation of the code snippet demonstrating the key implementation details of our project.",
        focus=[FocusRange(start=2, end=3)],
    )

    html = build_code_html(scene, kf, duration=6.0, theme="dark", accent="#3b82f6")

    renderer = AnimationRenderer()
    # At t=3.0, step 1 is fully active (started at ~0.72s)
    frame_bytes = renderer.capture_frame(html, 3.0)
    renderer.close()

    img = Image.open(io.BytesIO(frame_bytes)).convert("L")

    # In visual area (x=120 to 1800, y=200 to 960)
    # Top bar is 60px, padding is 24px -> Line 1 top is around y=284
    # Line height is 44px
    # Active lines (2-3) are roughly y=328 to y=416
    # Inactive lines (8-9) are roughly y=592 to y=680
    # Measure average luminance of active line vs inactive line
    active_crop = img.crop((250, 335, 800, 365))
    inactive_crop = img.crop((250, 600, 800, 630))

    active_data = active_crop.tobytes()
    inactive_data = inactive_crop.tobytes()
    active_mean = sum(active_data) / len(active_data)
    inactive_mean = sum(inactive_data) / len(inactive_data)

    # Active row has 100% text opacity + accent background, inactive has 30% text opacity
    assert active_mean > inactive_mean
