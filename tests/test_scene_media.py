"""Tests for repo2video/scene_media.py media scene rendering."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from GRC.anim import AnimationRenderer
from GRC.config import Settings
from GRC.media import MediaAsset
from GRC.models import Scene
from GRC.scene_media import (
    Rect,
    build_clip_frame_html,
    build_media_html,
    media_frame_rect,
    seed_from_id,
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


def test_media_frame_rect_scaling() -> None:
    """Verify media_frame_rect scales appropriately for 1920x1080 and 1280x720."""
    # Full HD (1920x1080)
    settings_fhd = Settings(video_width=1920, video_height=1080)
    rect_fhd = media_frame_rect(settings_fhd)
    assert rect_fhd == Rect(x=160, y=200, w=1600, h=760)

    # Default settings (omitted)
    assert media_frame_rect(None) == Rect(x=160, y=200, w=1600, h=760)

    # 720p (1280x720)
    settings_hd = Settings(video_width=1280, video_height=720)
    rect_hd = media_frame_rect(settings_hd)
    assert rect_hd == Rect(x=107, y=133, w=1067, h=507)


def test_seed_from_id_deterministic() -> None:
    """Verify seed_from_id is deterministic and returns an integer."""
    s1 = seed_from_id("m1")
    s2 = seed_from_id("m1")
    s3 = seed_from_id("m2")

    assert s1 == s2
    assert s1 != s3
    assert isinstance(s1, int)


def test_build_media_html_escaping_and_caption(tmp_path: Path) -> None:
    """Verify heading, alt text, and address bar are escaped properly."""
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (800, 380), color=(100, 150, 200))
    img.save(img_path)

    asset = MediaAsset(
        id="m1",
        kind="image",
        path=img_path,
        width=800,
        height=380,
        duration_seconds=0.0,
        alt='Screenshot with <script>alert("hack")</script> & "quotes"',
        origin="readme",
        source_url="https://example.com/assets/demo.png",
    )
    scene = Scene(
        id=1,
        type="media",
        media_id="m1",
        heading='Architecture & <script>alert("title")</script>',
        narration="Here is an overview of the architecture and component structure of our open source software project.",
    )

    html_out = build_media_html(scene, asset, duration=5.0)

    # Heading escaped
    assert "<script>alert" not in html_out
    assert "&lt;script&gt;alert(&quot;title&quot;)&lt;/script&gt;" in html_out

    # Alt text in caption chip escaped
    assert "&lt;script&gt;alert(&quot;hack&quot;)&lt;/script&gt;" in html_out
    assert "From the README" in html_out
    assert "README" in html_out
    assert "data:image/png;base64," in html_out


def test_build_media_html_zoom_decision(tmp_path: Path) -> None:
    """Verify slow zoom is enabled if aspect ratio is within 25% of frame aspect (1600/760=2.105)."""
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), color="blue").save(img_path)

    scene = Scene(
        id=1,
        type="media",
        media_id="m1",
        heading="Demo Media",
        narration="Here is an overview of the architecture and component structure of our open source software project.",
    )

    # 1. Matching aspect ratio: 2.105 (width 1600, height 760) -> within 25%
    asset_matching = MediaAsset(
        id="m1",
        kind="image",
        path=img_path,
        width=1600,
        height=760,
        duration_seconds=0.0,
        alt="diagram",
        origin="readme",
        source_url="https://example.com/demo.png",
    )
    html_zoom = build_media_html(scene, asset_matching, duration=5.0)
    assert "const SHOULD_ZOOM = true;" in html_zoom

    # 2. Square image: 1.0 (diff from 2.105 is ~52% > 25%) -> no zoom
    asset_square = MediaAsset(
        id="m2",
        kind="image",
        path=img_path,
        width=500,
        height=500,
        duration_seconds=0.0,
        alt="square logo",
        origin="readme",
        source_url="https://example.com/logo.png",
    )
    html_no_zoom = build_media_html(scene, asset_square, duration=5.0)
    assert "const SHOULD_ZOOM = false;" in html_no_zoom


def test_build_clip_frame_html_static_page(tmp_path: Path) -> None:
    """Verify build_clip_frame_html returns a static page with a black content area."""
    asset = MediaAsset(
        id="m1",
        kind="clip",
        path=tmp_path / "dummy.mp4",
        width=1280,
        height=720,
        duration_seconds=4.0,
        alt="demo video",
        origin="homepage",
        source_url="https://docs.myproject.org/tour",
    )
    scene = Scene(
        id=1,
        type="media",
        media_id="m1",
        heading="Video Tour",
        narration="Here is an overview of the architecture and component structure of our open source software project.",
    )

    clip_html = build_clip_frame_html(scene, asset)

    assert "docs.myproject.org" in clip_html
    assert "background: #000000;" in clip_html
    assert "__setTime" not in clip_html
    assert "media_frame_rect" in (build_clip_frame_html.__doc__ or "")


# ==============================================================================
# Browser-Marked Integration Tests
# ==============================================================================


@pytest.mark.browser
@skip_if_no_chromium
def test_media_still_zooms_between_start_and_end(tmp_path: Path) -> None:
    """Verify media still slightly scales between t=0 and t=duration while remaining within frame."""
    # Create an image with high-contrast corner markings to clearly detect zoom
    img_path = tmp_path / "zoom_target.png"
    im = Image.new("RGB", (1600, 760), color=(30, 30, 40))
    # Draw contrast blocks
    for x in range(100, 1500, 200):
        for y in range(100, 700, 150):
            im.paste((240, 240, 240), (x, y, x + 50, y + 50))
    im.save(img_path)

    asset = MediaAsset(
        id="m1",
        kind="image",
        path=img_path,
        width=1600,
        height=760,
        duration_seconds=0.0,
        alt="Interactive Demo",
        origin="readme",
        source_url="https://example.com/demo.png",
    )
    scene = Scene(
        id=1,
        type="media",
        media_id="m1",
        heading="Interface Zoom",
        narration="Here is an overview of the architecture and component structure of our open source software project.",
    )

    html = build_media_html(scene, asset, duration=5.0)

    renderer = AnimationRenderer()
    frame_t0 = renderer.capture_frame(html, 0.5)  # Window entrance finished
    frame_tend = renderer.capture_frame(html, 5.0)  # End of zoom
    renderer.close()

    img_t0 = Image.open(io.BytesIO(frame_t0))
    img_tend = Image.open(io.BytesIO(frame_tend))

    # Inspect inside the media frame: (x=160, y=200, w=1600, h=760)
    crop_t0 = img_t0.crop((300, 300, 1400, 800))
    crop_tend = img_tend.crop((300, 300, 1400, 800))

    # Because of zoom from 1.00 to 1.07, pixels must differ slightly
    diff_pixels = sum(
        1
        for p0, p1 in zip(crop_t0.tobytes(), crop_tend.tobytes(), strict=True)
        if abs(p0 - p1) > 10
    )
    assert diff_pixels > 500
