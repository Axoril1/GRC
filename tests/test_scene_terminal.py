"""Tests for repo2video/scene_terminal.py terminal scene rendering."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from GRC.anim import AnimationRenderer
from GRC.errors import SceneError
from GRC.models import Scene, TerminalSession
from GRC.scene_terminal import build_terminal_html, terminal_schedule


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


def test_build_terminal_html_escaped_heading_and_commands() -> None:
    """Verify heading, commands, and output with HTML entities are escaped."""
    scene = Scene(
        id=1,
        type="terminal",
        terminal_id="t1",
        heading='Install & <script>alert("hack")</script>',
        narration="Here is a step by step guide to installing and running our project in your local environment.",
    )
    session = TerminalSession(
        id="t1",
        purpose="install",
        commands=['curl -s https://example.com | bash < "arg" & test'],
        output=["Installing <b>package</b> & dependencies... > done"],
    )
    html_out = build_terminal_html(scene, session, duration=5.0)

    assert "<script>alert" not in html_out
    assert "&lt;script&gt;alert(&quot;hack&quot;)&lt;/script&gt;" in html_out
    assert "&lt;b&gt;package&lt;/b&gt;" in html_out
    assert "<b>package</b>" not in html_out


def test_terminal_schedule_normal_duration() -> None:
    """Verify terminal_schedule computes normal speed (32 cps) with standard pauses."""
    cmds = ["git clone https://github.com/repo/sample.git", "cd sample"]
    out = ["Cloning into 'sample'...", "Done."]
    # Ample duration: 20 seconds
    plan = terminal_schedule(cmds, out, duration=20.0)

    assert plan["cps"] == 32.0
    assert len(plan["commands"]) == 2
    assert len(plan["output"]) == 2

    # Command 0 starts at entry (0.5s)
    c0 = plan["commands"][0]
    assert c0["type_start"] == 0.5
    typing_dur_0 = len(cmds[0]) / 32.0
    assert abs(c0["type_end"] - (0.5 + typing_dur_0)) < 0.05
    # Pause of 0.25s after command
    assert abs(c0["done_at"] - (c0["type_end"] + 0.25)) < 0.05


def test_terminal_schedule_compress_pauses_before_dropping_commands() -> None:
    """Verify compression increases cps and compresses pauses down to 50% before dropping commands."""
    cmds = ["npm install", "npm run build"]
    out = ["Built in 1.2s"]

    # Short duration where natural time > avail_time, but fits within 70 cps & 50% pauses
    plan = terminal_schedule(cmds, out, duration=2.5)

    # Must retain both commands without dropping any
    assert len(plan["commands"]) == 2
    assert plan["cps"] > 32.0


def test_terminal_schedule_drops_commands_only_after_compression() -> None:
    """Verify commands are dropped from the end only when even compressed timings cannot fit."""
    cmds = [
        "pip install -r requirements.txt",
        "python setup.py build_ext --inplace",
        "pytest tests/",
        "python -m GRC.cli --help",
    ]
    out = ["All tests passed successfully."]

    # Very short duration (2.0s)
    plan = terminal_schedule(cmds, out, duration=2.0)

    # Should drop trailing commands but retain at least 1
    assert len(plan["commands"]) < 4
    assert len(plan["commands"]) >= 1
    assert plan["cps"] == 70.0


def test_terminal_empty_session_raises_scene_error() -> None:
    """Verify empty terminal session raises SceneError."""
    scene = Scene(
        id=1,
        type="terminal",
        terminal_id="t1",
        heading="Terminal",
        narration="Here is a step by step guide to installing and running our project in your local environment.",
    )
    session = TerminalSession.model_construct(id="t1", purpose="install", commands=[], output=[])

    with pytest.raises(SceneError):
        build_terminal_html(scene, session, duration=5.0)


# ==============================================================================
# Browser-Marked Integration Tests
# ==============================================================================


@pytest.mark.browser
@skip_if_no_chromium
def test_terminal_shows_typed_commands_progressively() -> None:
    """Verify terminal renders partially typed command early and fully typed command later."""
    cmd = "git clone https://github.com/fastapi/fastapi.git"
    session = TerminalSession(
        id="t1",
        purpose="install",
        commands=[cmd],
        output=["Cloning into 'fastapi'..."],
    )
    scene = Scene(
        id=1,
        type="terminal",
        terminal_id="t1",
        heading="Clone Repository",
        narration="Here is a step by step guide to installing and running our project in your local environment.",
    )

    html = build_terminal_html(scene, session, duration=6.0, theme="dark", accent="#3b82f6")

    renderer = AnimationRenderer()
    # At t=0.8s: command starts typing at 0.5s; at ~32 cps, ~10 characters typed
    frame_early = renderer.capture_frame(html, 0.8)
    # At t=2.5s: command has finished typing (len ~45 chars / 32 cps = ~1.4s + 0.5s = 1.9s)
    frame_late = renderer.capture_frame(html, 2.5)
    renderer.close()

    img_early = Image.open(io.BytesIO(frame_early)).convert("L")
    img_late = Image.open(io.BytesIO(frame_late)).convert("L")

    # Crop the first command line text area (y roughly 300 to 350, x from 220 to 1200)
    crop_early = img_early.crop((220, 300, 1200, 350))
    crop_late = img_late.crop((220, 300, 1200, 350))

    # Dark theme background is near-black (< 30)
    # Count foreground/text pixels (luminance > 40)
    early_text_pixels = sum(1 for p in crop_early.tobytes() if p > 40)
    late_text_pixels = sum(1 for p in crop_late.tobytes() if p > 40)

    # Late frame must have more typed characters visible than early frame
    assert late_text_pixels > early_text_pixels
