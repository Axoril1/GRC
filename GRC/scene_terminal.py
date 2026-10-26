"""Animated terminal replay scene builder showing commands and outputs safely as text."""

from __future__ import annotations

import html
import json
import logging
from typing import TYPE_CHECKING, Any

from GRC.anim import wrap_page
from GRC.errors import SceneError
from GRC.models import Scene, TerminalSession
from GRC.scenes import DEFAULT_ACCENT, FOOTER_CSS, footer_html, theme_css

if TYPE_CHECKING:
    from GRC.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "build_terminal_html",
    "terminal_schedule",
]


def terminal_schedule(
    commands: list[str],
    output: list[str],
    duration: float,
) -> dict[str, Any]:
    """Calculate the deterministic typing and animation schedule for terminal replay.

    Timing rules:
    - Window entry: 0.0 to 0.5s.
    - End hold: at least 0.8s.
    - Commands type at 32 cps, followed by a 0.25s pause.
    - Output lines appear every 0.10s (fade and slide 6px) after the last command.
    - Next command pause: 0.50s.
    - If total natural time exceeds available time (duration - 0.5s - 0.8s),
      first raise cps up to 70, then compress pauses proportionally down to 50%,
      and only then drop the last commands.

    Args:
        commands: List of shell command strings.
        output: List of output line strings.
        duration: Total scene duration in seconds.

    Returns:
        Schedule dictionary containing active commands, output, cps, and timestamps.
    """
    t_entry = 0.5
    t_hold = 0.8
    avail_time = max(0.5, duration - t_entry - t_hold)

    def calc_time(
        cmds: list[str],
        out_lines: list[str],
        cps_val: float,
        p_cmd: float,
        p_line: float,
        p_next: float,
    ) -> float:
        total = 0.0
        for idx, cmd in enumerate(cmds):
            t_type = max(0.1, len(cmd) / cps_val)
            total += t_type + p_cmd
            if idx == len(cmds) - 1:
                total += len(out_lines) * p_line
            if idx < len(cmds) - 1:
                total += p_next
        return total

    active_cmds = list(commands) if commands else [""]
    active_output = list(output)

    # Iteratively compress or drop commands if necessary
    while True:
        # Step 1: Base parameters (32 cps, 100% pauses)
        t_base = calc_time(active_cmds, active_output, 32.0, 0.25, 0.10, 0.50)
        if t_base <= avail_time:
            cps = 32.0
            p_cmd, p_line, p_next = 0.25, 0.10, 0.50
            break

        # Step 2: Raise cps up to 70
        t_at_70 = calc_time(active_cmds, active_output, 70.0, 0.25, 0.10, 0.50)
        if t_at_70 <= avail_time:
            # Binary search / solve for cps between 32 and 70
            low, high = 32.0, 70.0
            for _ in range(12):
                mid = (low + high) / 2.0
                if calc_time(active_cmds, active_output, mid, 0.25, 0.10, 0.50) <= avail_time:
                    high = mid
                else:
                    low = mid
            cps = round(high, 1)
            p_cmd, p_line, p_next = 0.25, 0.10, 0.50
            break

        # Step 3: With cps = 70, compress pauses down to 50%
        t_at_half_pauses = calc_time(
            active_cmds,
            active_output,
            70.0,
            0.25 * 0.5,
            0.10 * 0.5,
            0.50 * 0.5,
        )
        if t_at_half_pauses <= avail_time:
            # Scale pauses between 0.5 and 1.0
            low, high = 0.5, 1.0
            for _ in range(12):
                mid = (low + high) / 2.0
                if (
                    calc_time(active_cmds, active_output, 70.0, 0.25 * mid, 0.10 * mid, 0.50 * mid)
                    <= avail_time
                ):
                    low = mid
                else:
                    high = mid
            scale = round(low, 2)
            cps = 70.0
            p_cmd, p_line, p_next = 0.25 * scale, 0.10 * scale, 0.50 * scale
            break

        # Step 4: Drop the last command if we have more than 1
        if len(active_cmds) > 1:
            active_cmds.pop()
        else:
            # Down to 1 command, compress as much as possible
            cps = 70.0
            scale = max(
                0.2,
                (avail_time - (len(active_cmds[0]) / cps))
                / max(0.1, (0.25 + len(active_output) * 0.10)),
            )
            p_cmd = 0.25 * scale
            p_line = 0.10 * scale
            p_next = 0.50 * scale
            break

    # Build chronological timeline
    current_t = t_entry
    cmd_events: list[dict[str, Any]] = []
    output_events: list[dict[str, Any]] = []

    for i, cmd in enumerate(active_cmds):
        typing_dur = max(0.1, len(cmd) / cps)
        type_start = round(current_t, 4)
        type_end = round(type_start + typing_dur, 4)
        done_at = round(type_end + p_cmd, 4)
        cmd_events.append(
            {
                "index": i,
                "text": cmd,
                "type_start": type_start,
                "type_end": type_end,
                "done_at": done_at,
            }
        )
        current_t = done_at

        # Output lines belong to the last command
        if i == len(active_cmds) - 1:
            for j, line in enumerate(active_output):
                appear_at = round(current_t, 4)
                output_events.append(
                    {
                        "index": j,
                        "text": line,
                        "appear_at": appear_at,
                    }
                )
                current_t = round(current_t + p_line, 4)

        if i < len(active_cmds) - 1:
            current_t = round(current_t + p_next, 4)

    return {
        "cps": cps,
        "duration": duration,
        "commands": cmd_events,
        "output": output_events,
    }


def build_terminal_html(
    scene: Scene,
    session: TerminalSession,
    *,
    theme: str = "dark",
    accent: str = DEFAULT_ACCENT,
    repo_label: str = "",
    index: int = 1,
    total: int = 1,
    duration: float = 5.0,
    settings: Settings | None = None,
) -> str:
    """Build a complete animated HTML page for a terminal replay scene.

    Simulates a shell session displaying commands and outputs without ever
    executing repository code.

    Args:
        scene: Scene definition containing heading.
        session: Terminal session containing commands and outputs.
        theme: Color theme, either 'dark' or 'light'.
        accent: Hex accent color for the prompt and accents.
        repo_label: Repository name or label for the footer.
        index: Current scene index (1-based).
        total: Total number of scenes.
        duration: Scene duration in seconds.
        settings: Application settings.

    Returns:
        Complete HTML document string.

    Raises:
        SceneError: If session has neither commands nor outputs.
    """
    del settings
    if not session.commands and not session.output:
        raise SceneError(
            f"Terminal scene {scene.id} has no commands or outputs.",
            hint="Ensure TerminalSession has at least one command or output string.",
        )

    escaped_commands = [html.escape(c) for c in session.commands]
    escaped_output = [html.escape(o) for o in session.output]
    plan = terminal_schedule(escaped_commands, escaped_output, duration)

    escaped_heading = html.escape(scene.heading)
    footer_markup = footer_html(repo_label, index, total)

    # Render DOM items
    items_html: list[str] = []
    for cmd_info in plan["commands"]:
        idx = cmd_info["index"]
        items_html.append(
            f'<div class="terminal-line" id="line-cmd-{idx}">'
            f'<span class="term-prompt">$ </span>'
            f'<span class="term-cmd" id="cmd-text-{idx}"></span>'
            f'<span class="term-cursor" id="cursor-{idx}">█</span>'
            f"</div>"
        )
    for out_info in plan["output"]:
        j = out_info["index"]
        items_html.append(f'<div class="term-out" id="out-line-{j}">{out_info["text"]}</div>')

    rendered_items = "\n".join(items_html)

    body_html = f"""
<div class="scene-container">
  <div class="scene-heading">{escaped_heading}</div>
  <div class="visual-area">
    <div id="term-window" class="term-window">
      <div class="window-bar">
        <div class="window-dots">
          <span class="dot red"></span>
          <span class="dot yellow"></span>
          <span class="dot green"></span>
        </div>
        <div class="window-title">Terminal</div>
      </div>
      <div class="term-viewport" id="term-viewport">
        <div class="term-scroll-track" id="term-track">
          {rendered_items}
        </div>
      </div>
    </div>
  </div>
  {footer_markup}
</div>
"""

    is_light = theme.lower().strip() == "light"
    term_bg = "#eef0f6" if is_light else "#0c0e14"

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

.visual-area {{
  position: absolute;
  top: 200px;
  left: 120px;
  width: 1680px;
  height: 760px;
}}

.term-window {{
  width: 100%;
  height: 100%;
  border-radius: 24px;
  background: {term_bg};
  border: 1px solid var(--border);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  box-shadow: 0 20px 48px rgba(0, 0, 0, 0.35);
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
  font-weight: 600;
}}

.term-viewport {{
  flex: 1;
  position: relative;
  overflow: hidden;
  padding: 32px 40px;
}}

.term-scroll-track {{
  position: absolute;
  top: 32px;
  left: 40px;
  right: 40px;
  display: flex;
  flex-direction: column;
}}

.terminal-line {{
  height: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
  font-size: 30px;
  line-height: 44px;
  color: var(--text);
  white-space: pre;
}}

.term-prompt {{
  color: var(--accent);
  font-weight: 700;
  margin-right: 8px;
}}

.term-cmd {{
  color: var(--text);
  font-weight: 500;
}}

.term-cursor {{
  display: inline-block;
  color: var(--text);
  margin-left: 2px;
  font-size: 28px;
}}

.term-out {{
  height: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
  font-size: 30px;
  line-height: 44px;
  color: var(--muted);
  white-space: pre;
  overflow: hidden;
  text-overflow: ellipsis;
  opacity: 0;
  transform: translateY(6px);
}}

.slide-footer {{
  position: absolute;
  bottom: 60px;
  left: 120px;
  right: 120px;
}}
"""

    plan_json = json.dumps(plan)
    scene_js = f"""
const PLAN = {plan_json};
const LINE_HEIGHT = 44;
const VIEWPORT_HEIGHT = 630;

window.__setTime = function(t) {{
  const card = document.getElementById("term-window");
  const track = document.getElementById("term-track");

  // 1. Entrance animation (0.0 to 0.5s)
  const pEntry = progress(t, 0, 0.5);
  const eEntry = ease.outCubic(pEntry);
  if (card) {{
    card.style.opacity = eEntry;
    card.style.transform = "translateY(" + lerp(24, 0, eEntry) + "px)";
  }}

  let activeLinesCount = 0;
  const cursorBlink = blink(t, 2);

  // 2. Animate commands
  for (let i = 0; i < PLAN.commands.length; i++) {{
    const cmd = PLAN.commands[i];
    const lineEl = document.getElementById("line-cmd-" + i);
    const textEl = document.getElementById("cmd-text-" + i);
    const cursorEl = document.getElementById("cursor-" + i);

    if (t < cmd.type_start) {{
      if (lineEl) lineEl.style.opacity = "0";
      if (textEl) textEl.textContent = "";
      if (cursorEl) cursorEl.style.opacity = "0";
    }} else {{
      if (lineEl) lineEl.style.opacity = "1";
      activeLinesCount++;

      if (t >= cmd.type_end) {{
        if (textEl) textEl.innerHTML = cmd.text;
      }} else {{
        const res = typed(cmd.text, t, cmd.type_start, PLAN.cps);
        if (textEl) textEl.innerHTML = res.text;
      }}

      // Cursor visibility
      const isLastCmd = (i === PLAN.commands.length - 1);
      const isNextStarted = (i + 1 < PLAN.commands.length) && (t >= PLAN.commands[i + 1].type_start);

      if (isNextStarted) {{
        if (cursorEl) cursorEl.style.opacity = "0";
      }} else if (t < cmd.type_end) {{
        // Typing: cursor stays solid or blinks at end of typed text
        if (cursorEl) cursorEl.style.opacity = cursorBlink ? "1" : "0";
      }} else if (t < cmd.done_at) {{
        // Pause after typing
        if (cursorEl) cursorEl.style.opacity = cursorBlink ? "1" : "0";
      }} else if (isLastCmd && PLAN.output.length > 0 && t < PLAN.output[0].appear_at) {{
        // Brief pause before output starts
        if (cursorEl) cursorEl.style.opacity = cursorBlink ? "1" : "0";
      }} else if (isLastCmd && PLAN.output.length === 0) {{
        // Last command without output: hold blinking cursor
        if (cursorEl) cursorEl.style.opacity = cursorBlink ? "1" : "0";
      }} else {{
        if (cursorEl) cursorEl.style.opacity = "0";
      }}
    }}
  }}

  // 3. Animate outputs
  for (let j = 0; j < PLAN.output.length; j++) {{
    const out = PLAN.output[j];
    const outEl = document.getElementById("out-line-" + j);
    if (!outEl) continue;

    if (t < out.appear_at) {{
      outEl.style.opacity = "0";
      outEl.style.transform = "translateY(6px)";
    }} else {{
      activeLinesCount++;
      const pOut = progress(t, out.appear_at, out.appear_at + 0.10);
      const eOut = ease.outCubic(pOut);
      outEl.style.opacity = eOut;
      outEl.style.transform = "translateY(" + lerp(6, 0, eOut) + "px)";
    }}
  }}

  // 4. Smooth auto-scrolling
  const totalContentHeight = activeLinesCount * LINE_HEIGHT;
  if (totalContentHeight > VIEWPORT_HEIGHT) {{
    const scrollTarget = totalContentHeight - VIEWPORT_HEIGHT;
    if (track) track.style.transform = "translateY(-" + scrollTarget + "px)";
  }} else {{
    if (track) track.style.transform = "translateY(0px)";
  }}
}};
"""

    return wrap_page(body_html, scene_js, css=css, theme_css=theme_css(theme, accent))
