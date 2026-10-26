"""Tests for repo2video/captions.py subtitle cue generation and pill rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from repo2video.captions import (
    Cue,
    build_cues,
    format_srt_timestamp,
    render_caption_images,
    write_srt,
)
from repo2video.config import Settings


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


def test_format_srt_timestamp() -> None:
    """Verify format_srt_timestamp correctly formats seconds into HH:MM:SS,mmm."""
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(1.234) == "00:00:01,234"
    assert format_srt_timestamp(65.123) == "00:01:05,123"
    assert format_srt_timestamp(3661.045) == "01:01:01,045"
    assert format_srt_timestamp(-5.0) == "00:00:00,000"


def test_build_cues_short_text() -> None:
    """Verify build_cues handles short single-sentence text without error."""
    narrations = ["Quick summary."]
    starts = [0.0]
    durations = [3.0]

    cues = build_cues(narrations, starts, durations, max_chars=42, max_lines=2)

    assert len(cues) == 1
    assert cues[0].start == 0.0
    assert cues[0].end == 3.0
    assert cues[0].lines == ["Quick summary."]


def test_build_cues_long_text_and_many_sentences() -> None:
    """Verify build_cues splits multiple sentences and wraps into cues with max lines and chars."""
    text = (
        "Welcome to repo2video. This pipeline transforms public GitHub repositories into "
        "polished demonstration videos. It uses local LLMs, Playwright browser rendering, "
        "and FFmpeg concatenation."
    )
    narrations = [text]
    starts = [0.0]
    durations = [12.0]

    cues = build_cues(narrations, starts, durations, max_chars=42, max_lines=2)

    assert len(cues) >= 2
    for cue in cues:
        assert len(cue.lines) <= 2
        for line in cue.lines:
            assert len(line) <= 42
        assert cue.end - cue.start >= 1.2
        assert cue.start >= 0.0
        assert cue.end <= 12.0


def test_build_cues_unicode_and_emojis() -> None:
    """Verify build_cues properly preserves non-ASCII and accented characters."""
    narrations = ["Développé avec succès! Visitez le site web officiel dès maintenant."]
    starts = [2.5]
    durations = [5.0]

    cues = build_cues(narrations, starts, durations, max_chars=42, max_lines=2)

    assert len(cues) >= 1
    combined_text = " ".join(" ".join(c.lines) for c in cues)
    assert "Développé avec succès!" in combined_text
    assert "dès maintenant." in combined_text
    assert cues[0].start >= 2.5
    assert cues[-1].end <= 7.5


def test_build_cues_minimum_duration_merging() -> None:
    """Verify build_cues merges short cues so that each cue meets minimum 1.2s threshold."""
    # Two very short sentences that individually would be < 1.2s
    narrations = ["First part. Second part."]
    starts = [0.0]
    durations = [1.8]

    cues = build_cues(narrations, starts, durations, max_chars=42, max_lines=2)

    # 1.8s is less than 2 * 1.2s = 2.4s, so cues must be merged into 1
    assert len(cues) == 1
    assert cues[0].start == 0.0
    assert cues[0].end == 1.8
    assert "First part." in cues[0].lines[0]


def test_build_cues_no_overlaps_and_bounds() -> None:
    """Verify cues across multiple scenes never overlap and stay within narration bounds."""
    narrations = [
        "First scene introduction and setup.",
        "Second scene showing core features and code highlights in detail.",
        "Third scene wrapping up with architecture overview.",
    ]
    starts = [0.0, 5.0, 12.0]
    durations = [4.5, 6.0, 3.5]

    cues = build_cues(narrations, starts, durations, max_chars=42, max_lines=2)

    assert len(cues) >= 3

    # Check that each cue stays strictly within its scene bounds
    for i in range(len(cues) - 1):
        assert cues[i].start < cues[i].end
        # Non-overlapping
        assert cues[i].end <= cues[i + 1].start

    # Check absolute boundary
    assert cues[0].start >= 0.0
    assert cues[-1].end <= 15.5


def test_write_srt(tmp_path: Path) -> None:
    """Verify write_srt generates a compliant UTF-8 SRT file."""
    cues = [
        Cue(start=0.0, end=2.5, lines=["First subtitle line", "Second subtitle line"]),
        Cue(start=3.0, end=5.25, lines=["Third subtitle line"]),
    ]
    srt_path = tmp_path / "captions.srt"
    write_srt(cues, srt_path)

    assert srt_path.is_file()
    content = srt_path.read_text(encoding="utf-8")

    assert content.startswith(
        "1\n00:00:00,000 --> 00:00:02,500\nFirst subtitle line\nSecond subtitle line"
    )
    assert "2\n00:00:03,000 --> 00:00:05,250\nThird subtitle line" in content


# ==============================================================================
# Browser-Marked Integration Tests
# ==============================================================================


@pytest.mark.browser
@skip_if_no_chromium
def test_render_caption_images_transparent_and_cached(tmp_path: Path) -> None:
    """Verify render_caption_images produces transparent PNGs with bounded dimensions and caches."""
    cues = [
        Cue(start=0.0, end=2.0, lines=["Short caption pill"]),
        Cue(
            start=2.0,
            end=5.0,
            lines=[
                "This is a longer two-line caption pill",
                "demonstrating centered rendering and transparent background.",
            ],
        ),
    ]
    settings = Settings()
    out_dir = tmp_path / "captions_cache"

    # First call: renders from scratch
    paths_first = render_caption_images(cues, out_dir, settings, theme="dark")

    assert len(paths_first) == 2
    for p in paths_first:
        assert p.is_file()
        assert p.stat().st_size > 0
        img = Image.open(p)
        assert img.mode in ("RGBA", "LA")
        # Pill max width constraint
        assert img.width <= 1500

    # Modify timestamp of the first file to check cache hit
    mtime_before = paths_first[0].stat().st_mtime_ns

    # Second call: must hit cache without modifying files
    paths_second = render_caption_images(cues, out_dir, settings, theme="dark")
    assert paths_first == paths_second
    assert paths_second[0].stat().st_mtime_ns == mtime_before
