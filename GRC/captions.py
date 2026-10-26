"""Subtitle caption cues parsing, SRT generation, and transparent pill rendering."""

from __future__ import annotations

import hashlib
import html
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from GRC.errors import RenderError

if TYPE_CHECKING:
    from GRC.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "Cue",
    "build_cues",
    "format_srt_timestamp",
    "render_caption_images",
    "write_srt",
]


@dataclass(frozen=True)
class Cue:
    """A single subtitle caption cue.

    Attributes:
        start: Start timestamp in seconds on the global video timeline.
        end: End timestamp in seconds on the global video timeline.
        lines: Text lines to display in the subtitle box.
    """

    start: float
    end: float
    lines: list[str]


def format_srt_timestamp(total_seconds: float) -> str:
    """Format floating-point seconds into an SRT timestamp HH:MM:SS,mmm.

    Args:
        total_seconds: Time in seconds (non-negative).

    Returns:
        Formatted timestamp string 'HH:MM:SS,mmm'.
    """
    if total_seconds < 0:
        total_seconds = 0.0
    millis = round(total_seconds * 1000)
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_cues(
    narrations: list[str],
    starts: list[float],
    durations: list[float],
    max_chars: int = 42,
    max_lines: int = 2,
) -> list[Cue]:
    """Split scene narrations into timed, wrapped subtitle cues.

    Splits each scene narration into sentences first, then wraps sentences
    into cues with at most max_lines lines and max_chars characters per line.
    The scene narration duration is divided proportionally among its cues,
    merging adjacent short cues so that each cue has a minimum duration of 1.2
    seconds (or a single merged cue if the scene duration is shorter).

    Args:
        narrations: List of narration texts for each scene.
        starts: Narration start timestamps for each scene on the timeline.
        durations: Real audio narration durations for each scene.
        max_chars: Maximum character width per subtitle line (default: 42).
        max_lines: Maximum lines per subtitle cue (default: 2).

    Returns:
        List of non-overlapping Cue objects aligned to the timeline.
    """
    cues: list[Cue] = []

    for text, scene_start, duration in zip(narrations, starts, durations):
        clean_text = " ".join(text.split())
        if not clean_text or duration <= 0:
            continue

        # Split into sentences preserving punctuation boundaries
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
        if not raw_sentences:
            raw_sentences = [clean_text]

        # Break each sentence into lines and chunks of max_lines
        raw_cues: list[list[str]] = []
        for s in raw_sentences:
            lines = textwrap.wrap(
                s,
                width=max_chars,
                break_long_words=True,
                break_on_hyphens=False,
            )
            if not lines:
                continue
            num_cues = (len(lines) + max_lines - 1) // max_lines
            base, extra = divmod(len(lines), num_cues)
            idx = 0
            for i in range(num_cues):
                # Distribute extra lines towards later chunks so earlier lines don't leave short tails
                chunk_len = base + (1 if (num_cues - 1 - i) < extra else 0)
                raw_cues.append(lines[idx : idx + chunk_len])
                idx += chunk_len

        if not raw_cues:
            continue

        # Iteratively merge short cues (< 1.2s) with adjacent cues
        while len(raw_cues) > 1:
            total_chars = max(1, sum(sum(len(line) for line in c) for c in raw_cues))
            durs = [duration * (sum(len(line) for line in c) / total_chars) for c in raw_cues]
            min_dur = min(durs)
            if min_dur >= 1.2:
                break

            min_idx = durs.index(min_dur)
            merged = False

            # Check neighbors where combined lines <= max_lines
            neighbors = ([min_idx - 1] if min_idx > 0 else []) + (
                [min_idx + 1] if min_idx < len(raw_cues) - 1 else []
            )
            for n_idx in neighbors:
                combined_text = " ".join(raw_cues[min_idx] + raw_cues[n_idx])
                wrapped = textwrap.wrap(
                    combined_text,
                    width=max_chars,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                if len(wrapped) <= max_lines:
                    first, second = sorted([min_idx, n_idx])
                    raw_cues[first] = wrapped
                    del raw_cues[second]
                    merged = True
                    break

            if not merged:
                # If cannot merge within max_lines, merge anyway if scene duration is short (< 2.4s)
                if duration < 2.4:
                    if min_idx == 0:
                        comb = textwrap.wrap(
                            " ".join(raw_cues[0] + raw_cues[1]),
                            width=max_chars,
                            break_long_words=True,
                            break_on_hyphens=False,
                        )
                        raw_cues[0] = comb
                        del raw_cues[1]
                    else:
                        comb = textwrap.wrap(
                            " ".join(raw_cues[min_idx - 1] + raw_cues[min_idx]),
                            width=max_chars,
                            break_long_words=True,
                            break_on_hyphens=False,
                        )
                        raw_cues[min_idx - 1] = comb
                        del raw_cues[min_idx]
                else:
                    break

        # Calculate proportional time ranges
        total_chars = max(1, sum(sum(len(line) for line in c) for c in raw_cues))
        cum_chars = 0
        scene_end = scene_start + duration

        for c in raw_cues:
            c_len = sum(len(line) for line in c)
            start_off = duration * (cum_chars / total_chars)
            cum_chars += c_len
            end_off = duration * (cum_chars / total_chars)

            t_start = round(scene_start + start_off, 3)
            t_end = round(scene_start + end_off, 3)

            # Ensure bounds within scene narration window
            t_start = max(scene_start, min(t_start, scene_end))
            t_end = max(t_start, min(t_end, scene_end))

            cues.append(Cue(start=t_start, end=t_end, lines=c))

    return cues


def write_srt(cues: list[Cue], path: Path) -> Path:
    """Write subtitle cues to a standard UTF-8 SRT file.

    Args:
        cues: Ordered list of subtitle cues.
        path: Target file path for the .srt file.

    Returns:
        Path to the generated .srt file.

    Raises:
        RenderError: If writing the file fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    for idx, cue in enumerate(cues, 1):
        start_ts = format_srt_timestamp(cue.start)
        end_ts = format_srt_timestamp(cue.end)
        text = "\n".join(cue.lines)
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}")

    content = "\n\n".join(blocks) + ("\n" if blocks else "")
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RenderError(
            f"Failed to write SRT subtitle file at {path}: {exc}",
            hint="Check write permissions and directory existence.",
        ) from exc

    return path


def render_caption_images(
    cues: list[Cue],
    out_dir: Path,
    settings: Settings | None = None,
    theme: str = "dark",
) -> list[Path]:
    """Render transparent PNG pill images for each subtitle cue using Playwright.

    Images are cropped tightly to the caption pill bounding box and cached
    by content hash so identical cues are only rendered once.

    Args:
        cues: List of subtitle cues to render.
        out_dir: Directory where caption PNGs will be saved.
        settings: Application settings.
        theme: Color theme, either 'dark' or 'light'.

    Returns:
        Ordered list of Paths to the rendered PNG images for each cue.

    Raises:
        RenderError: If Playwright browser launch or screenshot capture fails.
    """
    del settings
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cues:
        return []

    # Map each cue to its cache path
    cue_paths: list[Path] = []
    needed_renders: dict[str, tuple[Path, list[str]]] = {}

    for cue in cues:
        cue_text = "\n".join(cue.lines)
        h = hashlib.sha256(f"{theme}:{cue_text}".encode()).hexdigest()[:16]
        png_path = out_dir / f"caption_{h}.png"
        cue_paths.append(png_path)
        if not png_path.is_file():
            needed_renders[h] = (png_path, cue.lines)

    if not needed_renders:
        return cue_paths

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RenderError(
            "Playwright is required to render burned caption images.",
            hint="Run `pip install playwright && playwright install chromium`.",
        ) from exc

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Disallow all external network connections
            page.route("**/*", lambda route: route.abort())

            # Render each missing caption PNG
            for png_path, lines in needed_renders.values():
                escaped_lines = "<br>".join(html.escape(line) for line in lines)
                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    background: transparent;
    display: inline-block;
    padding: 30px;
    overflow: hidden;
  }}
  .caption-pill {{
    display: inline-block;
    max-width: 1500px;
    padding: 18px 34px;
    border-radius: 28px;
    background: rgba(15, 23, 42, 0.85);
    color: #f8fafc;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 44px;
    font-weight: 600;
    line-height: 1.35;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    word-break: break-word;
  }}
  .caption-pill.light {{
    background: rgba(255, 255, 255, 0.90);
    color: #0f172a;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
  }}
</style>
</head>
<body>
  <div id="pill" class="caption-pill {theme}">{escaped_lines}</div>
</body>
</html>"""
                page.set_content(html_content)
                pill = page.locator("#pill")
                pill.screenshot(path=str(png_path), omit_background=True)

            browser.close()
    except PlaywrightError as exc:
        raise RenderError(
            f"Failed rendering caption pill image with Playwright: {exc}",
            hint="Ensure Playwright Chromium is properly installed.",
        ) from exc

    return cue_paths
