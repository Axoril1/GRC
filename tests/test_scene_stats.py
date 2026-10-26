"""Tests for repo2video/scene_stats.py statistics scene rendering."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image

from GRC.anim import AnimationRenderer
from GRC.errors import SceneError
from GRC.models import RepoData, Scene
from GRC.scene_stats import (
    build_stats_html,
    format_count,
    relative_time,
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


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0"),
        (42, "42"),
        (999, "999"),
        (1000, "1.0k"),
        (1234, "1.2k"),
        (9999, "10.0k"),
        (15000, "15k"),
        (999999, "1000k"),
        (1500000, "1.5M"),
        (2345678, "2.3M"),
        (-500, "-500"),
    ],
)
def test_format_count(n: int, expected: str) -> None:
    """Verify format_count handles small numbers, thousands, ten-thousands, and millions."""
    assert format_count(n) == expected


def test_relative_time_table() -> None:
    """Verify relative_time formats timestamps relative to a fixed reference time."""
    now = datetime(2026, 10, 26, 12, 0, 0, tzinfo=UTC)

    # Same day
    ts_today = (now - timedelta(hours=3)).isoformat()
    assert relative_time(ts_today, now=now) == "today"

    # Yesterday
    ts_yesterday = (now - timedelta(days=1)).isoformat()
    assert relative_time(ts_yesterday, now=now) == "yesterday"

    # 5 days ago
    ts_5d = (now - timedelta(days=5)).isoformat()
    assert relative_time(ts_5d, now=now) == "5 days ago"

    # 20 days ago
    ts_20d = (now - timedelta(days=20)).isoformat()
    assert relative_time(ts_20d, now=now) == "20 days ago"

    # ~2 months ago (60 days)
    ts_60d = (now - timedelta(days=60)).isoformat()
    assert relative_time(ts_60d, now=now) == "2 months ago"

    # ~1 year ago (400 days)
    ts_400d = (now - timedelta(days=400)).isoformat()
    assert relative_time(ts_400d, now=now) == "1 year ago"

    # ~2 years ago (800 days)
    ts_800d = (now - timedelta(days=800)).isoformat()
    assert relative_time(ts_800d, now=now) == "2 years ago"

    # Invalid and empty timestamps
    assert relative_time("", now=now) == ""
    assert relative_time("not-a-timestamp", now=now) == ""


def test_empty_stat_blocks_skipped() -> None:
    """Verify empty fields (topics, languages, license, updated) are omitted from blocks."""
    repo = RepoData(
        owner="pallets",
        name="flask",
        stars=65000,
        forks=16000,
        open_issues=12,
        license="",  # empty
        pushed_at="",  # empty
        language_percentages={},  # empty
        topics=[],  # empty
    )
    # Ask for license, topics, languages, and stars
    scene = Scene(
        id=1,
        type="stats",
        heading="Flask Metrics",
        narration="Here is a detailed breakdown of the repository community activity and adoption metrics over time.",
        stat_keys=["stars", "license", "topics", "languages"],
    )

    # Only 'stars' is present in repo; but len(valid_blocks) will be 1, raising SceneError!
    with pytest.raises(SceneError, match="[Ff]ewer than 2 valid stat blocks"):
        build_stats_html(scene, repo, duration=5.0)

    # When we add forks, we get 2 valid blocks (stars, forks)
    scene_valid = Scene(
        id=1,
        type="stats",
        heading="Flask Metrics",
        narration="Here is a detailed breakdown of the repository community activity and adoption metrics over time.",
        stat_keys=["stars", "forks", "license", "topics"],
    )
    html_out = build_stats_html(scene_valid, repo, duration=5.0)

    # Stars and forks are present in card headers
    assert "Stars" in html_out
    assert "Forks" in html_out
    # License and topics cards/blocks are omitted from DOM
    assert '<div id="topics-block"' not in html_out
    assert '<div id="languages-block"' not in html_out
    assert "License" not in html_out


def test_scene_error_on_too_few_blocks() -> None:
    """Verify SceneError is raised when fewer than 2 valid stat blocks can be built."""
    repo = RepoData(
        owner="owner",
        name="repo",
        stars=10,
        forks=0,
        license="",
    )
    # Requested keys have license which is empty in repo -> only 1 valid block ('stars')
    scene = Scene(
        id=1,
        type="stats",
        heading="Repo Stats",
        narration="Here is a comprehensive overview of repository metrics and usage statistics across the codebase of our system.",
        stat_keys=["stars", "license"],
    )

    with pytest.raises(SceneError, match="[Ff]ewer than 2 valid stat blocks"):
        build_stats_html(scene, repo, duration=4.0)
        build_stats_html(scene, repo, duration=4.0)


def test_build_stats_html_escapes_content() -> None:
    """Verify build_stats_html escapes HTML entities in heading, license, and topics."""
    repo = RepoData(
        owner="secure",
        name="app",
        stars=1200,
        forks=340,
        license='MIT & <script>alert("hack")</script>',
        topics=["python", "ai & ml", "<span>tag</span>"],
    )
    scene = Scene(
        id=1,
        type="stats",
        heading='Danger <script>alert("xss")</script> & Heading',
        narration="Here is an in-depth look at the repository stars, license information, and primary topic tags.",
        stat_keys=["stars", "license", "topics"],
    )

    html_out = build_stats_html(scene, repo, duration=5.0)

    # Danger tags must never be present raw
    assert "<script>alert" not in html_out
    assert "<span>tag</span>" not in html_out
    # Escaped versions must appear
    assert "&lt;script&gt;alert(&quot;hack&quot;)&lt;/script&gt;" in html_out
    assert "&lt;span&gt;tag&lt;/span&gt;" in html_out
    assert "ai &amp; ml" in html_out
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html_out


def test_build_stats_html_languages_bar() -> None:
    """Verify language percentages block renders stacked bar with legend."""
    repo = RepoData(
        owner="fastapi",
        name="fastapi",
        stars=75000,
        language_percentages={"Python": 95.5, "Shell": 3.2, "HTML": 1.3},
    )
    scene = Scene(
        id=1,
        type="stats",
        heading="Language Distribution",
        narration="A comprehensive breakdown of the primary programming languages comprising the project codebase and repository in detail.",
        stat_keys=["stars", "languages"],
    )

    html_out = build_stats_html(scene, repo, duration=5.0)

    assert "lang-bar" in html_out
    assert "Python" in html_out
    assert "95.5%" in html_out
    assert "Shell" in html_out
    assert "3.2%" in html_out


# ==============================================================================
# Browser-Marked Integration Tests
# ==============================================================================


@pytest.mark.browser
@skip_if_no_chromium
def test_stats_scene_renders_and_counts_up() -> None:
    """Verify Playwright renders stats scene and counter values animate upwards."""
    repo = RepoData(
        owner="psf",
        name="requests",
        stars=50000,
        forks=9200,
        open_issues=180,
        license="Apache-2.0",
        pushed_at="2026-10-01T00:00:00Z",
    )
    scene = Scene(
        id=1,
        type="stats",
        heading="Requests By The Numbers",
        narration="Here is an overview of requests community metrics, stars, and open issues across GitHub repositories.",
        stat_keys=["stars", "forks", "issues", "license"],
    )

    duration = 5.0
    html_page = build_stats_html(scene, repo, duration=duration, theme="dark", accent="#3b82f6")

    renderer = AnimationRenderer()
    # At t=0.1: card and counter are at initial state (0)
    frame_early = renderer.capture_frame(html_page, 0.1)
    # At t=4.5: counter has animated to full formatted string (50k)
    frame_late = renderer.capture_frame(html_page, 4.5)
    renderer.close()

    img_early = Image.open(io.BytesIO(frame_early)).convert("L")
    img_late = Image.open(io.BytesIO(frame_late)).convert("L")

    assert img_early.size == (1920, 1080)
    assert img_late.size == (1920, 1080)

    # In visual area (x: 120 to 1800, y: 200 to 940)
    crop_early = img_early.crop((120, 200, 1800, 940))
    crop_late = img_late.crop((120, 200, 1800, 940))

    # Early frame card is entering or has "0"
    # Late frame has fully animated counters, icons, and labels
    early_pixels = sum(1 for p in crop_early.tobytes() if p > 50)
    late_pixels = sum(1 for p in crop_late.tobytes() if p > 50)

    assert late_pixels > early_pixels
