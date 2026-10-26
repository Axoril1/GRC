"""Animated media scene builder for README images, homepage screenshots, and clip frames."""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import mimetypes
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from GRC.anim import wrap_page
from GRC.errors import SceneError
from GRC.media import MediaAsset
from GRC.models import Scene
from GRC.scenes import DEFAULT_ACCENT, FOOTER_CSS, footer_html, theme_css

if TYPE_CHECKING:
    from GRC.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "Rect",
    "build_clip_frame_html",
    "build_media_html",
    "media_frame_rect",
    "seed_from_id",
]


@dataclass(frozen=True)
class Rect:
    """Bounding box rectangle defining frame positioning and dimensions.

    Attributes:
        x: Horizontal coordinate from left in pixels.
        y: Vertical coordinate from top in pixels.
        w: Width of the bounding box in pixels.
        h: Height of the bounding box in pixels.
    """

    x: int
    y: int
    w: int
    h: int


def media_frame_rect(settings: Settings | None = None) -> Rect:
    """Calculate the content area of the browser-style media frame.

    Scaled from a 1600 by 760 area at (160, 200) on a standard 1920 by 1080 canvas.

    Args:
        settings: Application settings containing canvas dimensions.

    Returns:
        Frozen Rect instance with scaled x, y, w, and h.
    """
    w_canvas = settings.video_width if settings is not None else 1920
    h_canvas = settings.video_height if settings is not None else 1080
    scale_x = w_canvas / 1920.0
    scale_y = h_canvas / 1080.0
    return Rect(
        x=round(160 * scale_x),
        y=round(200 * scale_y),
        w=round(1600 * scale_x),
        h=round(760 * scale_y),
    )


def seed_from_id(asset_id: str) -> int:
    """Derive a deterministic 32-bit unsigned integer seed from an asset ID.

    Args:
        asset_id: Identifier string (e.g. 'm1', 'h1').

    Returns:
        32-bit unsigned integer seed.
    """
    return int(hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:8], 16)


def _encode_data_uri(asset: MediaAsset) -> str:
    """Read an asset image file and convert it to a base64 data URI.

    Args:
        asset: MediaAsset whose path points to an image file.

    Returns:
        Data URI string.

    Raises:
        SceneError: If asset file is missing or cannot be read.
    """
    if not asset.path.is_file():
        raise SceneError(
            f"Media asset image not found: {asset.path}",
            hint="Ensure media assets are collected before building media scenes.",
        )
    try:
        raw_bytes = asset.path.read_bytes()
    except OSError as exc:
        raise SceneError(
            f"Failed to read media asset file: {asset.path} ({exc})",
            hint="Check file system permissions.",
        ) from exc

    mime, _ = mimetypes.guess_type(str(asset.path))
    if not mime:
        suffix = asset.path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        elif suffix == ".gif":
            mime = "image/gif"
        else:
            mime = "image/png"

    b64_str = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime};base64,{b64_str}"


def build_media_html(
    scene: Scene,
    asset: MediaAsset,
    *,
    theme: str = "dark",
    accent: str = DEFAULT_ACCENT,
    repo_label: str = "",
    index: int = 1,
    total: int = 1,
    duration: float = 5.0,
    settings: Settings | None = None,
) -> str:
    """Build an animated HTML page displaying a still image in a browser frame.

    If the image's aspect ratio is within 25 percent of the frame's aspect ratio,
    applies a slow zoom from 1.00 to 1.07 drifting toward a deterministic focus
    point in the middle 60 percent.

    Args:
        scene: Scene definition containing heading.
        asset: MediaAsset containing image metadata and local file path.
        theme: Color theme, either 'dark' or 'light'.
        accent: Hex accent color for accents.
        repo_label: Repository name or label for the footer.
        index: Current scene index (1-based).
        total: Total number of scenes.
        duration: Scene duration in seconds.
        settings: Application settings.

    Returns:
        Complete HTML document string.

    Raises:
        SceneError: If asset path does not exist.
    """
    rect = media_frame_rect(settings)
    data_uri = _encode_data_uri(asset)

    escaped_heading = html.escape(scene.heading)
    footer_markup = footer_html(repo_label, index, total)

    # Address bar content
    if asset.origin == "homepage":
        parsed = urlparse(asset.source_url)
        address_text = parsed.hostname or asset.source_url or "homepage"
    else:
        address_text = "README"
    escaped_address = html.escape(address_text)

    # Caption chip
    credit = "From the README" if asset.origin == "readme" else "Homepage"
    if asset.alt:
        caption_text = f"{credit} • {html.escape(asset.alt)}"
    else:
        caption_text = credit

    # Aspect ratio check: within 25% of frame aspect
    frame_aspect = rect.w / rect.h
    img_aspect = (asset.width / asset.height) if asset.height > 0 else 1.0
    aspect_diff = abs(img_aspect - frame_aspect) / frame_aspect
    should_zoom = aspect_diff <= 0.25

    seed = seed_from_id(asset.id)

    body_html = f"""
<div class="scene-container">
  <div class="scene-heading">{escaped_heading}</div>
  <div class="media-frame" style="left: {rect.x}px; top: {rect.y}px; width: {rect.w}px; height: {rect.h}px;">
    <div id="browser-window" class="browser-window">
      <div class="browser-bar">
        <div class="browser-dots">
          <span class="dot red"></span>
          <span class="dot yellow"></span>
          <span class="dot green"></span>
        </div>
        <div class="address-pill">
          <span class="lock-icon">🔒</span>
          <span class="address-text">{escaped_address}</span>
        </div>
      </div>
      <div class="browser-viewport">
        <img id="media-img" class="media-img" src="{data_uri}" alt="" />
        <div class="caption-chip">{caption_text}</div>
      </div>
    </div>
  </div>
  {footer_markup}
</div>
"""

    css = f"""
{FOOTER_CSS}

body {{
  width: 1920px;
  height: 1080px;
  overflow: hidden;
  background-color: var(--bg);
  background-image: radial-gradient(circle at 90% 10%, var(--accent-soft) 0%, transparent 65%);
  color: var(--text);
  font-family: Inter, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.35;
}}

.scene-container {{
  width: 1920px;
  height: 1080px;
  position: relative;
  overflow: hidden;
}}

.scene-heading {{
  position: absolute;
  top: 120px;
  left: 120px;
  right: 120px;
  height: 68px;
  font-size: 56px;
  font-weight: 700;
  line-height: 1.2;
  color: var(--text);
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}}

.media-frame {{
  position: absolute;
}}

.browser-window {{
  width: 100%;
  height: 100%;
  border-radius: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  box-shadow: 0 20px 48px rgba(0, 0, 0, 0.35);
  opacity: 0;
  transform: translateY(24px);
}}

.browser-bar {{
  height: 60px;
  min-height: 60px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  background: rgba(0, 0, 0, 0.15);
  border-bottom: 1px solid var(--border);
  position: relative;
}}

.browser-dots {{
  display: flex;
  align-items: center;
  gap: 10px;
}}

.dot {{
  width: 16px;
  height: 16px;
  border-radius: 50%;
}}
.dot.red {{ background: #ff5f56; }}
.dot.yellow {{ background: #ffbd2e; }}
.dot.green {{ background: #27c93f; }}

.address-pill {{
  margin-left: 24px;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid var(--border);
  border-radius: 9999px;
  padding: 6px 20px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 20px;
  color: var(--muted);
  font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
}}

.lock-icon {{
  font-size: 16px;
}}

.browser-viewport {{
  flex: 1;
  position: relative;
  overflow: hidden;
  background: var(--surface);
  display: flex;
  align-items: center;
  justify-content: center;
}}

.media-img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}}

.caption-chip {{
  position: absolute;
  bottom: 24px;
  left: 24px;
  padding: 8px 24px;
  border-radius: 9999px;
  background: rgba(0, 0, 0, 0.75);
  color: #ffffff;
  font-size: 20px;
  font-weight: 500;
  backdrop-filter: blur(8px);
  z-index: 10;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  pointer-events: none;
}}

.slide-footer {{
  position: absolute;
  bottom: 60px;
  left: 120px;
  right: 120px;
}}
"""

    scene_js = f"""
const SHOULD_ZOOM = {str(should_zoom).lower()};
const SEED = {seed};

// Deterministic focus point in middle 60% (range 20% to 80%)
const rng = rand(SEED);
const focusX = lerp(20, 80, rng());
const focusY = lerp(20, 80, rng());

window.__setTime = function(t) {{
  const D = window.__DURATION || {duration};
  const card = document.getElementById("browser-window");
  const img = document.getElementById("media-img");

  // Entrance animation (0.0 to 0.5s)
  const pEntry = progress(t, 0, 0.5);
  const eEntry = ease.outCubic(pEntry);
  if (card) {{
    card.style.opacity = eEntry;
    card.style.transform = "translateY(" + lerp(24, 0, eEntry) + "px)";
  }}

  // Slow zoom animation across duration
  if (img) {{
    if (SHOULD_ZOOM) {{
      const pZoom = progress(t, 0, D);
      const zoomScale = lerp(1.00, 1.07, pZoom);
      img.style.transformOrigin = focusX + "% " + focusY + "%";
      img.style.transform = "scale(" + zoomScale + ")";
    }} else {{
      img.style.transform = "none";
    }}
  }}
}};
"""

    return wrap_page(body_html, scene_js, css=css, theme_css=theme_css(theme, accent))


def build_clip_frame_html(
    scene: Scene,
    asset: MediaAsset,
    *,
    theme: str = "dark",
    accent: str = DEFAULT_ACCENT,
    repo_label: str = "",
    index: int = 1,
    total: int = 1,
    duration: float = 5.0,
    settings: Settings | None = None,
) -> str:
    """Build a static HTML page containing the browser frame with a black viewport.

    The content rectangle returned by media_frame_rect is (x=160, y=200, w=1600, h=760)
    on a 1920x1080 canvas. The video rendering pipeline will overlay the external video
    clip precisely onto this black content area.

    Args:
        scene: Scene definition containing heading.
        asset: MediaAsset defining origin and source URL.
        theme: Color theme, either 'dark' or 'light'.
        accent: Hex accent color for accents.
        repo_label: Repository name or label for the footer.
        index: Current scene index (1-based).
        total: Total number of scenes.
        duration: Scene duration in seconds.
        settings: Application settings.

    Returns:
        Complete static HTML document string.
    """
    del duration
    rect = media_frame_rect(settings)
    escaped_heading = html.escape(scene.heading)
    footer_markup = footer_html(repo_label, index, total)

    if asset.origin == "homepage":
        parsed = urlparse(asset.source_url)
        address_text = parsed.hostname or asset.source_url or "homepage"
    else:
        address_text = "README"
    escaped_address = html.escape(address_text)

    credit = "From the README" if asset.origin == "readme" else "Homepage"
    if asset.alt:
        caption_text = f"{credit} • {html.escape(asset.alt)}"
    else:
        caption_text = credit

    body_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
{theme_css(theme, accent)}
{FOOTER_CSS}

*, *::before, *::after {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

body {{
  width: 1920px;
  height: 1080px;
  overflow: hidden;
  background-color: var(--bg);
  background-image: radial-gradient(circle at 90% 10%, var(--accent-soft) 0%, transparent 65%);
  color: var(--text);
  font-family: Inter, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.35;
}}

.scene-container {{
  width: 1920px;
  height: 1080px;
  position: relative;
  overflow: hidden;
}}

.scene-heading {{
  position: absolute;
  top: 120px;
  left: 120px;
  right: 120px;
  height: 68px;
  font-size: 56px;
  font-weight: 700;
  line-height: 1.2;
  color: var(--text);
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}}

.media-frame {{
  position: absolute;
}}

.browser-window {{
  width: 100%;
  height: 100%;
  border-radius: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  box-shadow: 0 20px 48px rgba(0, 0, 0, 0.35);
}}

.browser-bar {{
  height: 60px;
  min-height: 60px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  background: rgba(0, 0, 0, 0.15);
  border-bottom: 1px solid var(--border);
}}

.browser-dots {{
  display: flex;
  align-items: center;
  gap: 10px;
}}

.dot {{
  width: 16px;
  height: 16px;
  border-radius: 50%;
}}
.dot.red {{ background: #ff5f56; }}
.dot.yellow {{ background: #ffbd2e; }}
.dot.green {{ background: #27c93f; }}

.address-pill {{
  margin-left: 24px;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid var(--border);
  border-radius: 9999px;
  padding: 6px 20px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 20px;
  color: var(--muted);
  font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
}}

.lock-icon {{
  font-size: 16px;
}}

.browser-viewport {{
  flex: 1;
  position: relative;
  overflow: hidden;
  background: #000000;
  display: flex;
  align-items: center;
  justify-content: center;
}}

.caption-chip {{
  position: absolute;
  bottom: 24px;
  left: 24px;
  padding: 8px 24px;
  border-radius: 9999px;
  background: rgba(0, 0, 0, 0.75);
  color: #ffffff;
  font-size: 20px;
  font-weight: 500;
  backdrop-filter: blur(8px);
  z-index: 10;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  pointer-events: none;
}}

.slide-footer {{
  position: absolute;
  bottom: 60px;
  left: 120px;
  right: 120px;
}}
</style>
</head>
<body>
<div class="scene-container">
  <div class="scene-heading">{escaped_heading}</div>
  <div class="media-frame" style="left: {rect.x}px; top: {rect.y}px; width: {rect.w}px; height: {rect.h}px;">
    <div id="browser-window" class="browser-window">
      <div class="browser-bar">
        <div class="browser-dots">
          <span class="dot red"></span>
          <span class="dot yellow"></span>
          <span class="dot green"></span>
        </div>
        <div class="address-pill">
          <span class="lock-icon">🔒</span>
          <span class="address-text">{escaped_address}</span>
        </div>
      </div>
      <div class="browser-viewport">
        <div class="caption-chip">{caption_text}</div>
      </div>
    </div>
  </div>
  {footer_markup}
</div>
</body>
</html>
"""
    return body_html
