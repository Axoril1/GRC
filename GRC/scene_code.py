"""Animated code scene builder with syntax highlighting and focus range transitions."""

from __future__ import annotations

import html
import logging
from typing import TYPE_CHECKING, Any

import pygments
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer_for_filename
from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound

from GRC.anim import wrap_page
from GRC.errors import SceneError
from GRC.models import FocusRange, KeyFile, Scene
from GRC.scenes import (
    DEFAULT_ACCENT,
    FOOTER_CSS,
    _hex_to_rgba,
    _validate_accent,
    footer_html,
    theme_css,
)

if TYPE_CHECKING:
    from GRC.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "build_code_html",
    "format_file_title",
    "highlight_code_to_lines",
    "step_windows",
]


def format_file_title(path: str) -> str:
    """Format file path for window header, cutting from left if over 60 characters.

    Args:
        path: File path string.

    Returns:
        Formatted title with leading ellipsis if truncated.
    """
    if len(path) > 60:
        return f"…{path[-59:]}"
    return path


def highlight_code_to_lines(
    content: str,
    language: str,
    path: str,
    theme: str,
) -> tuple[list[str], str]:
    """Tokenize and syntax highlight code into per-line HTML without cross-line spans.

    Iterates the Pygments token stream, splits each token value on newlines,
    and wraps each segment in its own span with a CSS class derived from the token type.
    No span ever crosses a line boundary.

    Args:
        content: Source code text.
        language: Programming language identifier or empty.
        path: File path used as fallback for lexer guessing.
        theme: Color theme, either 'dark' or 'light'.

    Returns:
        Tuple of (list of HTML strings for each line, generated CSS rules string).
    """
    lexer = None
    if language:
        try:
            lexer = get_lexer_by_name(language.strip().lower())
        except (ClassNotFound, ValueError):
            logger.debug("Lexer for language %r not found; attempting filename guess.", language)

    if lexer is None and path:
        try:
            lexer = guess_lexer_for_filename(path, content)
        except (ClassNotFound, ValueError):
            logger.debug("Lexer guess for %r failed; falling back to TextLexer.", path)

    if lexer is None:
        lexer = TextLexer()

    style_name = "monokai" if theme.lower().strip() != "light" else "default"
    style = get_style_by_name(style_name)

    lines: list[list[tuple[str, str]]] = [[]]
    token_classes: dict[Any, str] = {}
    css_rules: list[str] = []

    for ttype, val in pygments.lex(content, lexer):
        if ttype not in token_classes:
            cls_name = f"tok-{len(token_classes)}"
            token_classes[ttype] = cls_name
            st = style.style_for_token(ttype)
            props: list[str] = []
            if st.get("color"):
                props.append(f"color: #{st['color']}")
            if st.get("bold"):
                props.append("font-weight: bold")
            if st.get("italic"):
                props.append("font-style: italic")
            if props:
                css_rules.append(f".{cls_name} {{ {'; '.join(props)}; }}")
        else:
            cls_name = token_classes[ttype]

        parts = val.split("\n")
        for idx, part in enumerate(parts):
            if idx > 0:
                lines.append([])
            if part:
                lines[-1].append((cls_name, part))

    # Pygments always appends a trailing newline to the token stream, so pop the resulting empty line
    if lines and not lines[-1]:
        lines.pop()

    line_htmls: list[str] = []
    for line_parts in lines:
        if not line_parts:
            line_htmls.append("&nbsp;")
        else:
            spans = [f'<span class="{cls}">{html.escape(text)}</span>' for cls, text in line_parts]
            line_htmls.append("".join(spans))

    return line_htmls, "\n".join(css_rules)


def step_windows(
    duration: float,
    steps: list[FocusRange] | list[tuple[int, int]],
    snippet_total_lines: int | None = None,
) -> list[dict[str, Any]]:
    """Calculate deterministic timeline boundaries for focus range steps.

    The first 12 percent of duration is overview; then S steps (1 to 4) occupy
    equal shares of the remaining 80 percent. If duration is compressed, steps
    are compressed down to 50 percent of natural duration; beyond that, trailing
    steps are dropped.

    Args:
        duration: Scene duration in seconds.
        steps: List of focus ranges or (start, end) tuples.
        snippet_total_lines: Optional upper bound to clamp line numbers.

    Returns:
        List of step metadata dictionaries with start, end, t_start, t_end.
    """
    if not steps:
        return []

    # Limit to maximum 4 steps
    raw_steps = steps[:4]
    parsed_steps: list[tuple[int, int]] = []
    for s in raw_steps:
        if isinstance(s, FocusRange):
            st, en = s.start, s.end
        else:
            st, en = s[0], s[1]
        if snippet_total_lines is not None and snippet_total_lines > 0:
            st = max(1, min(st, snippet_total_lines))
            en = max(st, min(en, snippet_total_lines))
        else:
            st = max(1, st)
            en = max(st, en)
        parsed_steps.append((st, en))

    # Compression check: minimum natural step duration = 1.2s; 50% = 0.6s
    min_step_duration = 0.6
    entry_and_overview = 0.5 + 0.12 * duration
    available_for_steps = max(0.5, duration - entry_and_overview)

    active_steps = parsed_steps
    while len(active_steps) > 1 and (available_for_steps / len(active_steps)) < min_step_duration:
        active_steps = active_steps[:-1]

    s_count = len(active_steps)
    share_pct = 80.0 / s_count
    windows: list[dict[str, Any]] = []

    for k, (start_line, end_line) in enumerate(active_steps):
        start_pct = 12.0 + k * share_pct
        end_pct = 12.0 + (k + 1) * share_pct
        t_start = (start_pct / 100.0) * duration
        t_end = (end_pct / 100.0) * duration
        windows.append(
            {
                "step": k,
                "start": start_line,
                "end": end_line,
                "t_start": round(t_start, 4),
                "t_end": round(t_end, 4),
            }
        )

    return windows


def build_code_html(
    scene: Scene,
    key_file: KeyFile,
    *,
    theme: str = "dark",
    accent: str = DEFAULT_ACCENT,
    repo_label: str = "",
    index: int = 1,
    total: int = 1,
    duration: float = 5.0,
    settings: Settings | None = None,
) -> str:
    """Build a complete animated HTML page for a code snippet scene.

    Renders a code window with Pygments syntax highlighting, line numbers,
    and deterministic scrolling/focus animations as a pure function of time.

    Args:
        scene: Scene definition containing heading and focus ranges.
        key_file: Code file model containing path, language, and content.
        theme: Color theme, either 'dark' or 'light'.
        accent: Hex accent color for highlights and borders.
        repo_label: Repository name or label for the footer.
        index: Current scene index (1-based).
        total: Total number of scenes.
        duration: Scene duration in seconds.
        settings: Application settings.

    Returns:
        Complete HTML document string.

    Raises:
        SceneError: If key_file has no content or line count is invalid.
    """
    del settings
    if not key_file.content and not scene.code:
        raise SceneError(
            f"Code scene {scene.id} has no code content.",
            hint="Ensure KeyFile or scene.code provides non-empty source text.",
        )

    code_content = key_file.content or scene.code
    line_htmls, pygments_css = highlight_code_to_lines(
        code_content,
        key_file.language,
        key_file.path,
        theme,
    )
    total_snippet_lines = len(line_htmls)

    valid_accent = _validate_accent(accent)
    accent_focus = _hex_to_rgba(valid_accent, 0.18)

    # Focus ranges
    focus_list = scene.focus
    windows = step_windows(duration, focus_list, total_snippet_lines)
    has_focus = len(windows) > 0

    escaped_heading = html.escape(scene.heading)
    file_title = format_file_title(key_file.path or "code")
    escaped_title = html.escape(file_title)
    footer_markup = footer_html(repo_label, index, total)

    # Assemble line elements
    lines_divs: list[str] = []
    for line_num, line_inner in enumerate(line_htmls, start=1):
        lines_divs.append(
            f'<div class="code-line" id="line-{line_num}" data-line="{line_num}">'
            f'<div class="line-gutter">'
            f'<span class="accent-bar" id="bar-{line_num}"></span>'
            f'<span class="line-num">{line_num}</span>'
            f"</div>"
            f'<div class="line-content">{line_inner}</div>'
            f"</div>"
        )
    rendered_lines = "\n".join(lines_divs)

    # Chip element for active focus step
    chip_html = (
        '<div id="focus-chip" class="focus-chip" style="opacity: 0;">Overview</div>'
        if has_focus
        else ""
    )

    body_html = f"""
<div class="scene-container">
  <div class="scene-heading">{escaped_heading}</div>
  <div class="visual-area">
    <div id="code-window" class="code-window">
      <div class="window-bar">
        <div class="window-dots">
          <span class="dot red"></span>
          <span class="dot yellow"></span>
          <span class="dot green"></span>
        </div>
        <div class="window-title">{escaped_title}</div>
      </div>
      <div class="code-viewport" id="code-viewport">
        <div class="code-scroll-track" id="code-track">
          {rendered_lines}
        </div>
      </div>
      {chip_html}
    </div>
  </div>
  {footer_markup}
</div>
"""

    css = f"""
{FOOTER_CSS}

:root {{
  --accent-focus: {accent_focus};
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

.visual-area {{
  position: absolute;
  top: 200px;
  left: 120px;
  width: 1680px;
  height: 760px;
}}

.code-window {{
  width: 100%;
  height: 100%;
  border-radius: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  box-shadow: 0 20px 48px rgba(0, 0, 0, 0.25);
  opacity: 0;
  transform: translateY(24px);
}}

.window-bar {{
  height: 60px;
  min-height: 60px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  background: rgba(0, 0, 0, 0.15);
  border-bottom: 1px solid var(--border);
}}

.window-dots {{
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

.window-title {{
  font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
  font-size: 24px;
  color: var(--muted);
  margin-left: 20px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.code-viewport {{
  flex: 1;
  position: relative;
  overflow: hidden;
  padding: 24px 0;
}}

.code-scroll-track {{
  position: absolute;
  top: 24px;
  left: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
}}

.code-line {{
  height: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
  font-size: 30px;
  line-height: 44px;
  padding-right: 32px;
}}

.line-gutter {{
  width: 100px;
  min-width: 100px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  position: relative;
  user-select: none;
}}

.accent-bar {{
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  background: var(--accent);
  opacity: 0;
}}

.line-num {{
  color: var(--muted);
  padding-right: 24px;
  font-size: 26px;
}}

.line-content {{
  flex: 1;
  white-space: pre;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.focus-chip {{
  position: absolute;
  bottom: 24px;
  left: 32px;
  height: 40px;
  padding: 0 20px;
  border-radius: 9999px;
  background: var(--surface);
  border: 2px solid var(--accent);
  color: var(--accent);
  font-family: 'JetBrains Mono', monospace;
  font-size: 20px;
  font-weight: 700;
  display: flex;
  align-items: center;
  box-shadow: 0 4px 16px var(--accent-soft);
  z-index: 10;
}}

.slide-footer {{
  position: absolute;
  bottom: 60px;
  left: 120px;
  right: 120px;
}}

{pygments_css}
"""

    # Scene JavaScript animation logic
    import json

    windows_json = json.dumps(windows)
    scene_js = f"""
const STEPS = {windows_json};
const TOTAL_LINES = {total_snippet_lines};
const HAS_FOCUS = {str(has_focus).lower()};
const LINE_HEIGHT = 44;
const VIEWPORT_HEIGHT = 650;

window.__setTime = function(t) {{
  const D = window.__DURATION || {duration};
  const card = document.getElementById("code-window");
  const track = document.getElementById("code-track");
  const chip = document.getElementById("focus-chip");

  // 1. Window entrance (0.0 to 0.5s)
  const pEntry = progress(t, 0, 0.5);
  const eEntry = ease.outCubic(pEntry);
  if (card) {{
    card.style.opacity = eEntry;
    card.style.transform = "translateY(" + lerp(24, 0, eEntry) + "px)";
  }}

  const maxScroll = Math.max(0, (TOTAL_LINES * LINE_HEIGHT) - VIEWPORT_HEIGHT + 48);

  if (!HAS_FOCUS) {{
    // Without focus ranges: lines appear one by one across 10% to 50% of duration
    const tStart = 0.10 * D;
    const tEnd = 0.50 * D;
    for (let i = 1; i <= TOTAL_LINES; i++) {{
      const el = document.getElementById("line-" + i);
      if (!el) continue;
      const lStart = lerp(tStart, Math.max(tStart, tEnd - 0.25), (i - 1) / Math.max(1, TOTAL_LINES - 1));
      const pLine = progress(t, lStart, lStart + 0.25);
      el.style.opacity = ease.outCubic(pLine);
    }}

    // Scroll smoothly from top to bottom across 50% to 95% of duration
    if (maxScroll > 0) {{
      const pScroll = progress(t, 0.50 * D, 0.95 * D);
      const scrollY = lerp(0, maxScroll, ease.inOutCubic(pScroll));
      if (track) track.style.transform = "translateY(-" + scrollY + "px)";
    }}
    return;
  }}

  // With focus ranges
  // Identify active step at time t
  let activeStep = null;
  let activeIndex = -1;
  for (let k = 0; k < STEPS.length; k++) {{
    if (t >= STEPS[k].t_start) {{
      activeStep = STEPS[k];
      activeIndex = k;
    }}
  }}

  if (!activeStep) {{
    // Initial 12% overview: whole snippet undimmed, no scroll, no chip
    if (chip) chip.style.opacity = "0";
    if (track) track.style.transform = "translateY(0px)";
    for (let i = 1; i <= TOTAL_LINES; i++) {{
      const lineEl = document.getElementById("line-" + i);
      const barEl = document.getElementById("bar-" + i);
      if (lineEl) {{
        lineEl.style.opacity = "1";
        lineEl.style.background = "transparent";
      }}
      if (barEl) barEl.style.opacity = "0";
    }}
    return;
  }}

  // Active step transition (eased over 0.5s for lines, 0.6s for scroll)
  const pStep = progress(t, activeStep.t_start, activeStep.t_start + 0.5);
  const eStep = ease.inOutCubic(pStep);

  const pScroll = progress(t, activeStep.t_start, activeStep.t_start + 0.6);
  const eScroll = ease.inOutCubic(pScroll);

  // Target scroll: line activeStep.start sits ~2 lines below top
  const targetScroll = clamp((activeStep.start - 3) * LINE_HEIGHT, 0, maxScroll);
  let prevScroll = 0;
  if (activeIndex > 0) {{
    prevScroll = clamp((STEPS[activeIndex - 1].start - 3) * LINE_HEIGHT, 0, maxScroll);
  }}
  const currentScroll = lerp(prevScroll, targetScroll, eScroll);
  if (track) track.style.transform = "translateY(-" + currentScroll + "px)";

  // Update lines opacity, accent background, and gutter bar
  for (let i = 1; i <= TOTAL_LINES; i++) {{
    const lineEl = document.getElementById("line-" + i);
    const barEl = document.getElementById("bar-" + i);
    const inRange = i >= activeStep.start && i <= activeStep.end;

    // Check if in previous step range for smooth crossfade
    let inPrev = false;
    if (activeIndex > 0) {{
      const prev = STEPS[activeIndex - 1];
      inPrev = i >= prev.start && i <= prev.end;
    }}

    const startOpacity = (activeIndex === 0) ? 1.0 : (inPrev ? 1.0 : 0.3);
    const targetOpacity = inRange ? 1.0 : 0.3;
    const lineOpacity = lerp(startOpacity, targetOpacity, eStep);

    if (lineEl) {{
      lineEl.style.opacity = lineOpacity;
      lineEl.style.backgroundColor = inRange ? "var(--accent-focus)" : "transparent";
    }}
    if (barEl) {{
      barEl.style.opacity = inRange ? eStep : 0;
    }}
  }}

  // Update chip text and opacity
  if (chip) {{
    chip.textContent = "Lines " + activeStep.start + " to " + activeStep.end;
    chip.style.opacity = String(progress(t, activeStep.t_start, activeStep.t_start + 0.3));
  }}
}};
"""

    return wrap_page(body_html, scene_js, css=css, theme_css=theme_css(theme, accent))
