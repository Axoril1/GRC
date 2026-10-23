"""Scene builder stage that converts scene definitions into PNG slides using Playwright."""

from __future__ import annotations

import argparse
import hashlib
import html
import logging
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from playwright.sync_api import Route, sync_playwright

from repo2video.config import get_settings
from repo2video.errors import Repo2VideoError, SceneError
from repo2video.models import Scene, ScriptData

if TYPE_CHECKING:
    from repo2video.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_ACCENT = "#6366f1"
HEX_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _validate_accent(accent: str) -> str:
    """Validate that accent is a 3- or 6-digit hex color, returning fallback if invalid.

    Args:
        accent: The hex color string to validate.

    Returns:
        The validated hex color string or default accent.
    """
    clean = accent.strip()
    if HEX_COLOR_PATTERN.match(clean):
        return clean
    logger.warning("Invalid accent color '%s'; falling back to %s.", accent, DEFAULT_ACCENT)
    return DEFAULT_ACCENT


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 3- or 6-digit hex color string to rgba CSS format.

    Args:
        hex_color: Hex color string (e.g. '#6366f1' or '#fff').
        alpha: Alpha opacity value between 0.0 and 1.0.

    Returns:
        CSS rgba string (e.g. 'rgba(99, 102, 241, 0.12)').
    """
    clean = hex_color.lstrip("#")
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    try:
        r = int(clean[0:2], 16)
        g = int(clean[2:4], 16)
        b = int(clean[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha:.2f})"
    except (ValueError, IndexError):
        return f"rgba(99, 102, 241, {alpha:.2f})"


def build_html(
    scene: Scene,
    index: int,
    total: int,
    *,
    repo_label: str,
    theme: str = "dark",
    accent: str = DEFAULT_ACCENT,
) -> str:
    """Build a standalone HTML document for a single scene slide.

    Generates a deterministic, self-contained HTML page using embedded CSS variables,
    client-side font fitting, and strict HTML escaping.

    Args:
        scene: Scene data model defining the slide content and layout.
        index: Current 1-based scene index.
        total: Total number of scenes in the script.
        repo_label: Human-readable repository identifier (e.g. 'owner/repo').
        theme: Color theme, either 'dark' or 'light'.
        accent: Hex accent color for highlights, borders, and gradients.

    Returns:
        Complete HTML markup string.
    """
    valid_accent = _validate_accent(accent)
    accent_soft = _hex_to_rgba(valid_accent, 0.12)

    is_light = theme.lower().strip() == "light"
    if is_light:
        bg_color = "#f7f8fc"
        surface_color = "#ffffff"
        text_color = "#121528"
        muted_color = "#5b6382"
        border_color = "rgba(0, 0, 0, 0.08)"
    else:
        bg_color = "#0b1020"
        surface_color = "#151b30"
        text_color = "#f4f6fb"
        muted_color = "#9aa3bd"
        border_color = "rgba(255, 255, 255, 0.08)"

    # Escaped content
    escaped_heading = html.escape(scene.heading)
    escaped_bullets = [html.escape(b) for b in scene.bullets]
    escaped_code = html.escape(scene.code)
    escaped_label = html.escape(repo_label)

    progress_pct = max(0.0, min(100.0, (index / total * 100.0) if total > 0 else 100.0))

    # Body layout depending on scene type
    scene_type = scene.type.lower()
    content_html = ""

    if scene_type == "title":
        subtitle_html = ""
        if escaped_bullets:
            subtitle_html = f'<div class="title-subtitle">{escaped_bullets[0]}</div>'
        content_html = f"""
        <div class="layout-title">
          <div class="heading title-heading">{escaped_heading}</div>
          {subtitle_html}
        </div>
        """

    elif scene_type == "problem":
        bullet_cards = "".join(f'<div class="problem-card">{b}</div>' for b in escaped_bullets)
        content_html = f"""
        <div class="layout-standard">
          <div class="heading standard-heading">{escaped_heading}</div>
          <div class="bullets-container problem-bullets">{bullet_cards}</div>
        </div>
        """

    elif scene_type == "feature":
        bullet_items = "".join(
            f'<div class="feature-item"><span class="feature-badge">{i}</span><span class="feature-text">{b}</span></div>'
            for i, b in enumerate(escaped_bullets, start=1)
        )
        if escaped_code:
            code_panel = f"""
            <div class="code-column">
              <div class="code-container standard-code-box">
                <pre class="code-pre"><code>{escaped_code}</code></pre>
              </div>
            </div>
            """
            content_html = f"""
            <div class="layout-standard">
              <div class="heading standard-heading">{escaped_heading}</div>
              <div class="two-column-container">
                <div class="bullets-column">
                  <div class="bullets-container feature-bullets">{bullet_items}</div>
                </div>
                {code_panel}
              </div>
            </div>
            """
        else:
            content_html = f"""
            <div class="layout-standard">
              <div class="heading standard-heading">{escaped_heading}</div>
              <div class="bullets-container feature-bullets">{bullet_items}</div>
            </div>
            """

    elif scene_type == "howto":
        steps_items = "".join(
            f'<div class="howto-step"><span class="step-num">{i}</span><span class="step-text">{b}</span></div>'
            for i, b in enumerate(escaped_bullets, start=1)
        )
        terminal_html = ""
        if escaped_code:
            terminal_html = f"""
            <div class="code-column">
              <div class="code-container terminal-window">
                <div class="terminal-bar">
                  <span class="term-dot red"></span>
                  <span class="term-dot yellow"></span>
                  <span class="term-dot green"></span>
                  <span class="term-title">terminal</span>
                </div>
                <div class="terminal-body">
                  <pre class="code-pre"><code>{escaped_code}</code></pre>
                </div>
              </div>
            </div>
            """

        if escaped_code and escaped_bullets:
            content_html = f"""
            <div class="layout-standard">
              <div class="heading standard-heading">{escaped_heading}</div>
              <div class="two-column-container">
                <div class="bullets-column">
                  <div class="bullets-container howto-steps">{steps_items}</div>
                </div>
                {terminal_html}
              </div>
            </div>
            """
        elif escaped_code:
            content_html = f"""
            <div class="layout-standard">
              <div class="heading standard-heading">{escaped_heading}</div>
              {terminal_html}
            </div>
            """
        else:
            content_html = f"""
            <div class="layout-standard">
              <div class="heading standard-heading">{escaped_heading}</div>
              <div class="bullets-container howto-steps">{steps_items}</div>
            </div>
            """

    elif scene_type == "outro":
        outro_bullets = "".join(f'<div class="outro-bullet">{b}</div>' for b in escaped_bullets)
        content_html = f"""
        <div class="layout-outro">
          <div class="heading outro-heading">{escaped_heading}</div>
          <div class="bullets-container outro-bullets">{outro_bullets}</div>
          <div class="outro-pill">
            <span class="star-icon">★</span>
            <span class="pill-label">{escaped_label}</span>
            <span class="pill-divider">•</span>
            <span class="pill-cta">Star it on GitHub</span>
          </div>
        </div>
        """
    else:
        # Fallback for unexpected scene types
        bullet_items = "".join(f'<div class="generic-bullet">{b}</div>' for b in escaped_bullets)
        content_html = f"""
        <div class="layout-standard">
          <div class="heading standard-heading">{escaped_heading}</div>
          <div class="bullets-container">{bullet_items}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
:root {{
  --bg: {bg_color};
  --surface: {surface_color};
  --text: {text_color};
  --muted: {muted_color};
  --accent: {valid_accent};
  --accent-soft: {accent_soft};
  --border: {border_color};
}}

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

html, body {{
  width: 1920px;
  height: 1080px;
  overflow: hidden;
  background-color: var(--bg);
  background-image: radial-gradient(circle at 90% 10%, var(--accent-soft) 0%, transparent 65%);
  color: var(--text);
  font-family: Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  line-height: 1.35;
  -webkit-font-smoothing: antialiased;
}}

.slide-canvas {{
  width: 1920px;
  height: 1080px;
  padding: 120px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  overflow: hidden;
  position: relative;
}}

.content-box {{
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  overflow: hidden;
}}

.heading {{
  font-weight: 700;
  line-height: 1.2;
  color: var(--text);
  overflow-wrap: anywhere;
  word-break: break-word;
}}

.standard-heading {{
  font-size: 96px;
  margin-bottom: 48px;
  max-height: 240px;
  overflow: hidden;
}}

.title-heading {{
  font-size: 120px;
  font-weight: 800;
  text-align: center;
  max-height: 480px;
  overflow: hidden;
}}

.outro-heading {{
  font-size: 96px;
  font-weight: 800;
  text-align: center;
  margin-bottom: 32px;
  max-height: 240px;
  overflow: hidden;
}}

.bullets-container {{
  font-size: 52px;
  line-height: 1.35;
  color: var(--text);
  overflow-wrap: anywhere;
  word-break: break-word;
  max-height: 520px;
  overflow: hidden;
}}

.code-container {{
  font-size: 38px;
  line-height: 1.35;
  font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, Menlo, monospace;
  overflow-wrap: anywhere;
  word-break: break-word;
  max-height: 520px;
  overflow: hidden;
}}

.code-pre {{
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  white-space: pre;
  margin: 0;
}}

.layout-title {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  width: 100%;
}}

.title-subtitle {{
  font-size: 52px;
  color: var(--muted);
  margin-top: 36px;
  max-width: 1400px;
}}

.layout-standard {{
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  justify-content: flex-start;
}}

.two-column-container {{
  display: flex;
  flex-direction: row;
  gap: 56px;
  align-items: flex-start;
  width: 100%;
}}

.bullets-column {{
  flex: 1.1;
  min-width: 0;
}}

.code-column {{
  flex: 0.9;
  min-width: 0;
  width: 100%;
}}

/* Problem cards */
.problem-bullets {{
  display: flex;
  flex-direction: column;
  gap: 24px;
}}

.problem-card {{
  background: var(--surface);
  border-left: 14px solid var(--accent);
  border-radius: 24px;
  padding: 24px 36px;
  color: var(--text);
}}

/* Feature items */
.feature-bullets {{
  display: flex;
  flex-direction: column;
  gap: 24px;
}}

.feature-item {{
  display: flex;
  align-items: center;
  gap: 24px;
}}

.feature-badge {{
  width: 64px;
  height: 64px;
  min-width: 64px;
  border-radius: 50%;
  background: var(--accent);
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 34px;
}}

/* Howto steps & Terminal */
.howto-steps {{
  display: flex;
  flex-direction: column;
  gap: 24px;
}}

.howto-step {{
  display: flex;
  align-items: center;
  gap: 24px;
}}

.step-num {{
  width: 64px;
  height: 64px;
  min-width: 64px;
  border-radius: 20px;
  background: var(--surface);
  border: 2px solid var(--accent);
  color: var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 34px;
}}

.standard-code-box {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 32px;
}}

.terminal-window {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 24px;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(0,0,0,0.25);
}}

.terminal-bar {{
  background: rgba(0, 0, 0, 0.2);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border);
}}

.term-dot {{
  width: 18px;
  height: 18px;
  border-radius: 50%;
}}
.term-dot.red {{ background: #ff5f56; }}
.term-dot.yellow {{ background: #ffbd2e; }}
.term-dot.green {{ background: #27c93f; }}

.term-title {{
  color: var(--muted);
  font-size: 24px;
  margin-left: 12px;
}}

.terminal-body {{
  padding: 32px;
  overflow: hidden;
}}

/* Outro */
.layout-outro {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  width: 100%;
}}

.outro-bullets {{
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 48px;
}}

.outro-pill {{
  background: var(--surface);
  border: 3px solid var(--accent);
  border-radius: 9999px;
  padding: 20px 48px;
  font-size: 40px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 20px;
  color: var(--text);
  box-shadow: 0 12px 32px var(--accent-soft);
}}

.star-icon {{
  color: #eab308;
  font-size: 44px;
}}

.pill-divider {{
  color: var(--muted);
}}

.pill-cta {{
  color: var(--accent);
}}

/* Footer */
.slide-footer {{
  width: 100%;
  height: 48px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 28px;
  color: var(--muted);
}}

.footer-label {{
  font-weight: 600;
  letter-spacing: -0.01em;
}}

.footer-right {{
  display: flex;
  align-items: center;
  gap: 24px;
}}

.progress-track {{
  width: 220px;
  height: 8px;
  background: var(--border);
  border-radius: 4px;
  overflow: hidden;
}}

.progress-fill {{
  height: 100%;
  background: var(--accent);
  border-radius: 4px;
}}

.page-counter {{
  font-variant-numeric: tabular-nums;
  font-weight: 500;
}}
</style>
</head>
<body>
<div class="slide-canvas">
  <div class="content-box">
    {content_html}
  </div>
  <div class="slide-footer">
    <div class="footer-label">{escaped_label}</div>
    <div class="footer-right">
      <div class="progress-track">
        <div class="progress-fill" style="width: {progress_pct}%;"></div>
      </div>
      <span class="page-counter">{index} / {total}</span>
    </div>
  </div>
</div>

<script>
(function() {{
  function fitElement(el, floorRatio) {{
    if (!el) return;
    const style = window.getComputedStyle(el);
    let fontSize = parseFloat(style.fontSize);
    if (!fontSize || fontSize <= 0) return;
    const minSize = fontSize * floorRatio;

    while (
      (el.scrollHeight > el.clientHeight || el.scrollWidth > el.clientWidth) &&
      fontSize > minSize
    ) {{
      fontSize *= 0.96;
      if (fontSize < minSize) {{
        fontSize = minSize;
      }}
      el.style.fontSize = fontSize + 'px';
      if (fontSize <= minSize) {{
        break;
      }}
    }}
  }}

  function runAutoFit() {{
    const heading = document.querySelector('.heading');
    const bullets = document.querySelector('.bullets-container');
    const code = document.querySelector('.code-container');

    fitElement(heading, 0.6);
    fitElement(bullets, 0.6);
    fitElement(code, 0.6);

    window.__fitted = true;
  }}

  if (document.readyState === 'complete') {{
    runAutoFit();
  }} else {{
    window.addEventListener('load', runAutoFit);
  }}
}})();
</script>
</body>
</html>"""


def render_scenes(
    script: ScriptData,
    out_dir: Path,
    *,
    repo_label: str,
    theme: str = "dark",
    accent: str = DEFAULT_ACCENT,
    settings: Settings | None = None,
) -> list[Path]:
    """Render scenes to high-resolution PNG slides using headless Playwright Chromium.

    Blocks all external network traffic, auto-fits content, computes SHA-256 sidecar
    hashes for caching, and outputs PNG files named scene_01.png, scene_02.png, etc.

    Args:
        script: ScriptData instance containing scenes to render.
        out_dir: Directory where generated slides and cache sidecars are saved.
        repo_label: Human-readable project label for slide headers/footers.
        theme: Color theme ('dark' or 'light').
        accent: Hex accent color.
        settings: Application settings. If None, loaded via get_settings().

    Returns:
        List of paths to generated PNG files in scene order.

    Raises:
        SceneError: If Playwright Chromium is not installed or slide rendering fails.
    """
    cfg = settings if settings is not None else get_settings()
    width = cfg.video_width
    height = cfg.video_height

    out_dir.mkdir(parents=True, exist_ok=True)
    png_paths: list[Path] = []
    total = len(script.scenes)

    try:
        pw = sync_playwright().start()
    except Exception as exc:
        raise SceneError(f"Failed to start Playwright: {exc}") from exc

    browser = None
    try:
        try:
            browser = pw.chromium.launch(headless=True)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "executable doesn't exist" in err_msg or "executable does not exist" in err_msg:
                raise SceneError(
                    "The Chromium browser for Playwright is not installed.",
                    hint="Run: playwright install chromium",
                ) from exc
            raise SceneError(
                f"Failed to launch Chromium: {exc}",
                hint="Run: playwright install chromium",
            ) from exc

        page = browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )

        # Block all network traffic except internal data: and about: schemes
        def _filter_route(route: Route) -> None:
            url = route.request.url.lower()
            if url.startswith(("data:", "about:")):
                route.continue_()
            else:
                route.abort()

        page.route("**/*", _filter_route)

        for index, scene in enumerate(script.scenes, start=1):
            png_name = f"scene_{scene.id:02d}.png"
            png_path = out_dir / png_name
            sidecar_path = out_dir / f"{png_name}.sha256"

            try:
                html_doc = build_html(
                    scene=scene,
                    index=index,
                    total=total,
                    repo_label=repo_label,
                    theme=theme,
                    accent=accent,
                )
                html_hash = hashlib.sha256(html_doc.encode("utf-8")).hexdigest()

                # Check caching sidecar
                if png_path.is_file() and sidecar_path.is_file():
                    try:
                        cached_hash = sidecar_path.read_text(encoding="utf-8").strip()
                        if cached_hash == html_hash:
                            logger.info(
                                "Scene %02d cached, skipping render: %s",
                                scene.id,
                                png_path.name,
                            )
                            png_paths.append(png_path)
                            continue
                    except OSError:
                        pass

                start_time = time.monotonic()
                page.set_content(html_doc, wait_until="load")
                page.evaluate("() => document.fonts.ready")
                page.wait_for_function("() => window.__fitted === true")

                page.screenshot(path=str(png_path), type="png", full_page=False)
                sidecar_path.write_text(html_hash, encoding="utf-8")

                elapsed = time.monotonic() - start_time
                logger.info(
                    "Rendered scene %02d (%s) to %s in %.2fs",
                    scene.id,
                    scene.type,
                    png_path.name,
                    elapsed,
                )
                png_paths.append(png_path)

            except Exception as exc:
                if isinstance(exc, SceneError):
                    raise
                raise SceneError(
                    f"Failed to render scene {scene.id} ({scene.type}): {exc}",
                    hint="Check scene content for formatting errors or inspect Playwright logs.",
                ) from exc

    finally:
        if browser is not None:
            browser.close()
        pw.stop()

    return png_paths


def main() -> None:
    """CLI entrypoint for rendering scenes.json to PNG slide images."""
    parser = argparse.ArgumentParser(
        description="Render scenes.json to PNG slides using Playwright."
    )
    parser.add_argument("scenes_file", type=Path, help="Path to scenes.json")
    parser.add_argument(
        "--theme", default="dark", choices=["dark", "light"], help="Color theme (dark or light)"
    )
    parser.add_argument(
        "--accent", default=DEFAULT_ACCENT, help="Hex accent color (e.g. '#6366f1')"
    )
    parser.add_argument("--label", default=None, help="Repository label (e.g. owner/name)")

    args = parser.parse_args()

    try:
        if not args.scenes_file.is_file():
            raise SceneError(
                f"File not found: {args.scenes_file}",
                hint=f"Ensure that {args.scenes_file} exists.",
            )

        script = ScriptData.load(args.scenes_file)
        label = args.label if args.label is not None else script.title
        out_dir = args.scenes_file.parent / "slides"

        rendered_paths = render_scenes(
            script=script,
            out_dir=out_dir,
            repo_label=label,
            theme=args.theme,
            accent=args.accent,
        )

        for p in rendered_paths:
            print(str(p))

    except Repo2VideoError as err:
        sys.stderr.write(f"{err}\n")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Unexpected error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
