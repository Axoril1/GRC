"""Animated statistics scene builder highlighting repository metrics and languages."""

from __future__ import annotations

import html
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from repo2video.anim import wrap_page
from repo2video.errors import SceneError
from repo2video.models import RepoData, Scene
from repo2video.scenes import DEFAULT_ACCENT, FOOTER_CSS, footer_html, theme_css

if TYPE_CHECKING:
    from repo2video.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "build_stats_html",
    "format_count",
    "relative_time",
]

LANGUAGE_PALETTE: list[str] = [
    "#3b82f6",  # Blue
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#8b5cf6",  # Purple
    "#ec4899",  # Pink
    "#06b6d4",  # Cyan
]

SVG_ICONS: dict[str, str] = {
    "stars": """<svg width="40" height="40" viewBox="0 0 24 24" fill="var(--accent)" stroke="none">
  <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26" />
</svg>""",
    "forks": """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="18" r="3" />
  <circle cx="6" cy="6" r="3" />
  <circle cx="18" cy="6" r="3" />
  <path d="M18 9v2a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V9" />
  <path d="M12 12v3" />
</svg>""",
    "issues": """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10" />
  <line x1="12" y1="8" x2="12" y2="12" />
  <line x1="12" y1="16" x2="12.01" y2="16" />
</svg>""",
    "license": """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
  <polyline points="14 2 14 8 20 8" />
  <line x1="16" y1="13" x2="8" y2="13" />
  <line x1="16" y1="17" x2="8" y2="17" />
  <polyline points="10 9 9 9 8 9" />
</svg>""",
    "updated": """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10" />
  <polyline points="12 6 12 12 16 14" />
</svg>""",
}


def format_count(n: int) -> str:
    """Format an integer count into compact human-readable representation.

    Rules:
    - Under 1,000: unchanged (e.g. 0 -> '0', 999 -> '999')
    - 1,000 to 9,999: one decimal place with 'k' (e.g. 1000 -> '1.0k', 1234 -> '1.2k')
    - 10,000 to 999,999: rounded integer with 'k' (e.g. 15000 -> '15k', 999999 -> '1000k')
    - 1,000,000 and above: one decimal place with 'M' (e.g. 1500000 -> '1.5M', 2300000 -> '2.3M')

    Args:
        n: Integer number.

    Returns:
        Formatted compact string.
    """
    if n < 0:
        return f"-{format_count(-n)}"
    if n < 1000:
        return str(n)
    if n < 10_000:
        val = n / 1000.0
        return f"{val:.1f}k"
    if n < 1_000_000:
        val = round(n / 1000.0)
        return f"{val}k"
    val = n / 1_000_000.0
    return f"{val:.1f}M"


def relative_time(iso_timestamp: str, now: datetime | None = None) -> str:
    """Format an ISO timestamp into a human relative time string.

    Examples:
    - 'today', 'yesterday', '3 days ago', '2 months ago', '1 year ago'
    - Empty string for invalid or empty input.

    Args:
        iso_timestamp: ISO 8601 date or datetime string.
        now: Optional current datetime reference for deterministic testing.

    Returns:
        Relative time string or empty string.
    """
    if not iso_timestamp or not isinstance(iso_timestamp, str):
        return ""

    try:
        clean_iso = iso_timestamp.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_iso)
    except (ValueError, TypeError):
        return ""

    ref_now = now if now is not None else datetime.now(UTC)
    if dt.tzinfo is not None and ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=UTC)
    elif dt.tzinfo is None and ref_now.tzinfo is not None:
        dt = dt.replace(tzinfo=UTC)

    delta = ref_now - dt
    total_seconds = delta.total_seconds()
    days = int(total_seconds // 86400)

    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    if days < 365:
        months = max(1, round(days / 30.4))
        return f"{months} month{'s' if months != 1 else ''} ago"

    years = max(1, round(days / 365.25))
    return f"{years} year{'s' if years != 1 else ''} ago"


def build_stats_html(
    scene: Scene,
    repo: RepoData,
    *,
    theme: str = "dark",
    accent: str = DEFAULT_ACCENT,
    repo_label: str = "",
    index: int = 1,
    total: int = 1,
    duration: float = 5.0,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> str:
    """Build a complete animated HTML page for a statistics showcase scene.

    Displays number metrics with counting animations, interactive language share bar,
    and repository topics.

    Args:
        scene: Scene definition containing heading and stat_keys.
        repo: Repository data model with stars, forks, issues, license, etc.
        theme: Color theme ('dark' or 'light').
        accent: Hex accent color for highlights and icons.
        repo_label: Repository name or label for the footer.
        index: Current scene index (1-based).
        total: Total number of scenes.
        duration: Scene duration in seconds.
        settings: Application settings.
        now: Optional datetime reference for relative_time.

    Returns:
        Complete HTML document string.

    Raises:
        SceneError: If fewer than 2 valid stat blocks have non-empty data.
    """
    del settings
    desired_keys = scene.stat_keys or [
        "stars",
        "forks",
        "issues",
        "languages",
        "license",
        "updated",
        "topics",
    ]

    # Evaluate each key and filter out empty data
    valid_blocks: list[dict[str, Any]] = []

    for key in desired_keys:
        clean_key = key.strip().lower()
        if clean_key == "stars":
            if repo.stars is not None and repo.stars >= 0:
                valid_blocks.append(
                    {
                        "key": "stars",
                        "type": "number",
                        "label": "Stars",
                        "raw_val": repo.stars,
                        "formatted": format_count(repo.stars),
                    }
                )
        elif clean_key == "forks":
            if repo.forks is not None and repo.forks >= 0:
                valid_blocks.append(
                    {
                        "key": "forks",
                        "type": "number",
                        "label": "Forks",
                        "raw_val": repo.forks,
                        "formatted": format_count(repo.forks),
                    }
                )
        elif clean_key == "issues":
            if repo.open_issues is not None and repo.open_issues >= 0:
                valid_blocks.append(
                    {
                        "key": "issues",
                        "type": "number",
                        "label": "Issues",
                        "raw_val": repo.open_issues,
                        "formatted": format_count(repo.open_issues),
                    }
                )
        elif clean_key == "license":
            if repo.license and repo.license.strip():
                valid_blocks.append(
                    {
                        "key": "license",
                        "type": "text",
                        "label": "License",
                        "raw_val": repo.license.strip(),
                        "formatted": repo.license.strip(),
                    }
                )
        elif clean_key == "updated":
            rel = relative_time(repo.pushed_at, now=now)
            if rel:
                valid_blocks.append(
                    {
                        "key": "updated",
                        "type": "text",
                        "label": "Updated",
                        "raw_val": rel,
                        "formatted": rel,
                    }
                )
        elif clean_key == "languages" and repo.language_percentages:
            valid_blocks.append(
                {
                    "key": "languages",
                    "type": "languages",
                    "label": "Languages",
                    "data": repo.language_percentages,
                }
            )
        elif clean_key == "topics" and repo.topics:
            valid_blocks.append(
                {
                    "key": "topics",
                    "type": "topics",
                    "label": "Topics",
                    "data": repo.topics[:8],
                }
            )

    # Limit to at most 4 blocks in total
    valid_blocks = valid_blocks[:4]

    if len(valid_blocks) < 2:
        raise SceneError(
            f"Stats scene {scene.id} has fewer than 2 valid stat blocks ({len(valid_blocks)} found).",
            hint="Ensure repository metadata contains at least two non-empty stat fields.",
        )

    escaped_heading = html.escape(scene.heading)
    footer_markup = footer_html(repo_label, index, total)

    # Separate card blocks (number/text) from full-width bottom blocks (languages/topics)
    card_blocks = [b for b in valid_blocks if b["type"] in ("number", "text")]
    wide_blocks = [b for b in valid_blocks if b["type"] in ("languages", "topics")]

    # Render card row
    cards_html: list[str] = []
    anim_cards: list[dict[str, Any]] = []

    for i, b in enumerate(card_blocks):
        escaped_label = html.escape(b["label"])
        icon_svg = SVG_ICONS.get(b["key"], "")
        val_id = f"card-val-{i}"
        val_display = "0" if b["type"] == "number" else html.escape(str(b["formatted"]))
        escaped_raw = html.escape(str(b["raw_val"]))

        cards_html.append(
            f'<div id="stat-card-{i}" class="stat-card">'
            f'<div class="card-header">'
            f'<span class="card-icon">{icon_svg}</span>'
            f'<span class="card-label">{escaped_label}</span>'
            f"</div>"
            f'<div id="{val_id}" class="card-val" data-target="{escaped_raw}">{val_display}</div>'
            f"</div>"
        )
        anim_cards.append(
            {
                "index": i,
                "type": b["type"],
                "target": b["raw_val"],
                "formatted": b["formatted"],
            }
        )

    cards_row_html = f'<div class="cards-row">{"".join(cards_html)}</div>' if cards_html else ""

    # Render wide bottom block
    wide_html = ""
    languages_data: list[dict[str, Any]] = []
    topics_data: list[str] = []

    for wb in wide_blocks:
        if wb["type"] == "languages":
            # Process language shares
            langs_dict: dict[str, float] = wb["data"]
            sorted_langs = sorted(langs_dict.items(), key=lambda item: item[1], reverse=True)[:6]
            segments: list[str] = []
            legend_items: list[str] = []
            for idx, (lname, pct) in enumerate(sorted_langs):
                color = LANGUAGE_PALETTE[idx % len(LANGUAGE_PALETTE)]
                segments.append(
                    f'<div id="lang-bar-{idx}" class="lang-segment" '
                    f'style="background: {color}; width: 0%;" data-pct="{pct:.1f}"></div>'
                )
                legend_items.append(
                    f'<div class="lang-legend-item">'
                    f'<span class="lang-dot" style="background: {color};"></span>'
                    f'<span class="lang-name">{html.escape(lname)}</span>'
                    f'<span class="lang-pct">{pct:.1f}%</span>'
                    f"</div>"
                )
                languages_data.append({"index": idx, "pct": round(pct, 1), "color": color})

            wide_html = f"""
            <div id="languages-block" class="wide-block languages-card">
              <div class="block-title">Languages</div>
              <div class="stacked-bar">{"".join(segments)}</div>
              <div id="languages-legend" class="languages-legend">{"".join(legend_items)}</div>
            </div>
            """
        elif wb["type"] == "topics":
            chips: list[str] = []
            for idx, tname in enumerate(wb["data"]):
                escaped_tname = html.escape(tname)
                chips.append(
                    f'<span id="topic-chip-{idx}" class="topic-chip">{escaped_tname}</span>'
                )
                topics_data.append(tname)
            wide_html = f"""
            <div id="topics-block" class="wide-block topics-card">
              <div class="block-title">Topics</div>
              <div class="topics-container">{"".join(chips)}</div>
            </div>
            """

    body_html = f"""
<div class="scene-container">
  <div class="scene-heading">{escaped_heading}</div>
  <div class="visual-area">
    <div class="stats-wrapper">
      {cards_row_html}
      {wide_html}
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

.visual-area {{
  position: absolute;
  top: 200px;
  left: 120px;
  width: 1680px;
  height: 760px;
}}

.stats-wrapper {{
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 32px;
}}

.cards-row {{
  width: 100%;
  display: flex;
  gap: 32px;
  align-items: stretch;
}}

.stat-card {{
  flex: 1;
  border-radius: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 36px 40px;
  box-shadow: 0 20px 48px rgba(0, 0, 0, 0.25);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 260px;
  opacity: 0;
  transform: translateY(20px);
}}

.card-header {{
  display: flex;
  align-items: center;
  gap: 16px;
}}

.card-icon {{
  display: flex;
  align-items: center;
  justify-content: center;
}}

.card-label {{
  font-size: 26px;
  font-weight: 600;
  color: var(--muted);
}}

.card-val {{
  font-size: 96px;
  font-weight: 800;
  line-height: 1.05;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.wide-block {{
  width: 100%;
  border-radius: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 32px 40px;
  box-shadow: 0 20px 48px rgba(0, 0, 0, 0.25);
  opacity: 0;
  transform: translateY(20px);
}}

.block-title {{
  font-size: 26px;
  font-weight: 600;
  color: var(--muted);
  margin-bottom: 20px;
}}

.stacked-bar {{
  width: 100%;
  height: 40px;
  border-radius: 20px;
  background: rgba(0, 0, 0, 0.15);
  overflow: hidden;
  display: flex;
  margin-bottom: 24px;
}}

.lang-segment {{
  height: 100%;
  transition: width 0.05s linear;
}}

.languages-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 32px;
  opacity: 0;
}}

.lang-legend-item {{
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 24px;
  font-weight: 500;
}}

.lang-dot {{
  width: 16px;
  height: 16px;
  border-radius: 50%;
}}

.lang-name {{
  color: var(--text);
  font-weight: 600;
}}

.lang-pct {{
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}}

.topics-container {{
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}}

.topic-chip {{
  padding: 12px 28px;
  border-radius: 9999px;
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid var(--border);
  font-size: 26px;
  font-weight: 600;
  color: var(--text);
  opacity: 0;
  transform: scale(0.7);
  display: inline-block;
}}

.slide-footer {{
  position: absolute;
  bottom: 60px;
  left: 120px;
  right: 120px;
}}
"""

    cards_json = (
        json.dumps(anim_cards)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    langs_json = json.dumps(languages_data)
    topics_count = len(topics_data)

    scene_js = f"""
const CARDS = {cards_json};
const LANGS = {langs_json};
const TOPICS_COUNT = {topics_count};

function formatCountJs(n) {{
  if (n < 0) return "-" + formatCountJs(-n);
  if (n < 1000) return String(Math.round(n));
  if (n < 10000) return (n / 1000).toFixed(1) + "k";
  if (n < 1000000) return String(Math.round(n / 1000)) + "k";
  return (n / 1000000).toFixed(1) + "M";
}}

window.__setTime = function(t) {{
  const D = window.__DURATION || {duration};

  // 1. Staggered card entrances and number counting
  for (let i = 0; i < CARDS.length; i++) {{
    const c = CARDS[i];
    const cardEl = document.getElementById("stat-card-" + i);
    const valEl = document.getElementById("card-val-" + i);
    if (!cardEl || !valEl) continue;

    const cStart = 0.40 + i * 0.15;
    const pCard = progress(t, cStart, cStart + 0.50);
    const eCard = ease.outCubic(pCard);
    cardEl.style.opacity = eCard;
    cardEl.style.transform = "translateY(" + lerp(20, 0, eCard) + "px)";

    if (c.type === "number") {{
      const pNum = ease.outQuart(progress(t, cStart, cStart + 1.40));
      if (pNum >= 1.0) {{
        valEl.textContent = c.formatted;
      }} else {{
        const curVal = lerp(0, c.target, pNum);
        valEl.textContent = formatCountJs(curVal);
      }}
    }} else {{
      valEl.textContent = c.formatted;
    }}
  }}

  // 2. Languages stacked bar growth
  const langBlock = document.getElementById("languages-block");
  const langLegend = document.getElementById("languages-legend");
  if (langBlock && LANGS.length > 0) {{
    const lStart = 0.40 + CARDS.length * 0.15;
    const pBlock = progress(t, lStart, lStart + 0.50);
    langBlock.style.opacity = ease.outCubic(pBlock);
    langBlock.style.transform = "translateY(" + lerp(20, 0, ease.outCubic(pBlock)) + "px)";

    const pBar = ease.outQuart(progress(t, lStart + 0.20, lStart + 1.40));
    for (let j = 0; j < LANGS.length; j++) {{
      const seg = document.getElementById("lang-bar-" + j);
      if (seg) seg.style.width = (LANGS[j].pct * pBar) + "%";
    }}

    if (langLegend) {{
      const pLegend = progress(t, lStart + 0.80, lStart + 1.20);
      langLegend.style.opacity = ease.outCubic(pLegend);
    }}
  }}

  // 3. Topic tag chips pop in
  const topicBlock = document.getElementById("topics-block");
  if (topicBlock && TOPICS_COUNT > 0) {{
    const tStart = 0.40 + CARDS.length * 0.15;
    const pBlock = progress(t, tStart, tStart + 0.50);
    topicBlock.style.opacity = ease.outCubic(pBlock);
    topicBlock.style.transform = "translateY(" + lerp(20, 0, ease.outCubic(pBlock)) + "px)";

    for (let k = 0; k < TOPICS_COUNT; k++) {{
      const chipEl = document.getElementById("topic-chip-" + k);
      if (!chipEl) continue;
      const chipStart = tStart + 0.20 + k * 0.10;
      const pChip = progress(t, chipStart, chipStart + 0.25);
      const eChip = ease.outCubic(pChip);
      chipEl.style.opacity = eChip;
      chipEl.style.transform = "scale(" + lerp(0.70, 1.0, eChip) + ")";
    }}
  }}
}};
"""

    return wrap_page(body_html, scene_js, css=css, theme_css=theme_css(theme, accent))
