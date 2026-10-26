"""Script writer stage that transforms repository analysis into structured scenes using an LLM."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from GRC.config import get_settings
from GRC.errors import LLMError, Repo2VideoError, ScriptError
from GRC.llm import extract_json, get_client
from GRC.models import (
    V2_SCENE_TYPES,
    VALID_NODE_GROUPS,
    VALID_STAT_KEYS,
    ArchitectureSpec,
    FocusRange,
    KeyFile,
    NodeSpec,
    RepoData,
    ScriptData,
    TerminalSession,
    format_validation_error,
    word_count,
)
from GRC.prompts import REPAIR_PROMPT, SYSTEM_PROMPT, build_user_message

if TYPE_CHECKING:
    from GRC.config import Settings
    from GRC.llm import LLMClient

logger = logging.getLogger(__name__)

TYPE_SYNONYMS: dict[str, str] = {
    "intro": "title",
    "introduction": "title",
    "conclusion": "outro",
    "closing": "outro",
    "call_to_action": "outro",
    "cta": "outro",
    "features": "feature",
    "feature_list": "feature",
    "usage": "howto",
    "install": "howto",
    "installation": "howto",
    "how_to": "howto",
    "getting_started": "howto",
    "pain_point": "problem",
    "problems": "problem",
}

# Regex to strip leading bullet markers such as -, *, •, 1., 1), etc.
LEADING_BULLET_MARKER = re.compile(r"^(?:[-*•‣⁃·○●■]|\d+[\.)\-]|\([0-9]+\))\s*")

# Unicode ranges covering modern emojis and pictographs
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\u200d\ufe0f]"
)


def _clean_common_text(text: str) -> str:
    """Normalize text shared across headings, bullets, and narration.

    Replaces curly quotes and dashes with ASCII equivalents, removes markdown
    styling (asterisks, backticks, leading hashes, emphasis underscores),
    strips emojis, and collapses repeated whitespace.

    Args:
        text: Raw text string to clean.

    Returns:
        Cleaned plain text string.
    """
    # Replace curly single quotes
    for q in ("‘", "’", "‚", "‛"):
        text = text.replace(q, "'")

    # Replace curly double quotes
    for q in ("“", "”", "„", "‟"):
        text = text.replace(q, '"')

    # Replace typographic dashes
    for d in ("–", "—", "‒", "―"):
        text = text.replace(d, "-")

    # Remove markdown asterisks and backticks
    text = text.replace("*", "").replace("`", "")

    # Remove leading markdown heading hashes
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)

    # Remove emphasis underscores while preserving identifiers (e.g. my_var)
    text = re.sub(r"(^|[\s(])__?([^_]+)__?([\s).,!?:;]|$)", r"\1\2\3", text)

    # Remove emojis
    text = EMOJI_PATTERN.sub("", text)

    # Collapse repeated whitespace
    return " ".join(text.split())


def _clean_narration_text(text: str) -> str:
    """Normalize narration text specifically for speech synthesis.

    Removes URLs, expands symbols and abbreviations ('&', '%', '->', 'e.g.',
    'i.e.', 'vs.'), and strips markdown formatting.

    Args:
        text: Raw narration string.

    Returns:
        Spoken-style narration string.
    """
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Replace ASCII and unicode arrows with 'to'
    text = text.replace("->", " to ")
    for arrow in ("→", "➔", "➜", "➡", "⇒", "►"):
        text = text.replace(arrow, " to ")

    # Expand ampersands
    text = re.sub(r"\s*&\s*", " and ", text)

    # Expand percent signs (replace '%' with ' percent')
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)
    text = text.replace("%", " percent")

    # Expand common abbreviations
    text = re.sub(r"\be\.g\.\b|\be\.g\b", "for example", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\.e\.\b|\bi\.e\b", "that is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvs\.\b|\bvs\b", "versus", text, flags=re.IGNORECASE)

    # Apply common cleaning (markdown, emojis, curly quotes, dashes)
    return _clean_common_text(text)


def _clean_code_snippet(raw: Any) -> str:
    """Normalize code snippets by stripping surrounding fences and trailing spaces.

    Preserves internal indentation and line breaks.

    Args:
        raw: Raw code object or string.

    Returns:
        Normalized code snippet string.
    """
    if raw is None:
        return ""
    code = str(raw)
    lines = code.splitlines()

    # Strip surrounding markdown code fence header
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    # Strip closing code fence
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    # Reassemble and strip trailing whitespace on each line
    cleaned_lines = [line.rstrip() for line in lines]
    cleaned = "\n".join(cleaned_lines)

    # Strip single surrounding backticks if present
    stripped = cleaned.strip()
    if stripped.startswith("`") and stripped.endswith("`"):
        cleaned = stripped[1:-1]

    return cleaned.strip("\r\n")


def _clean_id(raw: Any) -> str | None:
    """Normalize identifier fields, turning null, 'none', or empty string into None.

    Args:
        raw: Raw identifier value.

    Returns:
        Cleaned identifier string or None.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "null"):
        return None
    return s


def _normalize_focus(raw: Any) -> list[dict[str, int]]:
    """Normalize focus line ranges into a list of dictionaries with start and end.

    Args:
        raw: Raw focus input from model.

    Returns:
        List of dictionaries with 'start' and 'end' integer keys (at most 4).
    """
    if not raw or raw in ("none", "null"):
        return []

    if isinstance(raw, str):
        items = [s.strip() for s in re.split(r"[,;]+", raw) if s.strip()]
    elif isinstance(raw, (list, tuple)):
        # Handle a single pair passed directly as [3, 7]
        if len(raw) == 2 and all(isinstance(x, (int, str)) and str(x).isdigit() for x in raw):
            items = [raw]
        else:
            items = list(raw)
    else:
        return []

    ranges: list[dict[str, int]] = []
    for item in items:
        if isinstance(item, dict):
            try:
                s = int(item.get("start", 0))
                e = int(item.get("end", 0))
                if s >= 1 and e >= 1:
                    ranges.append({"start": s, "end": e})
            except (ValueError, TypeError):
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                s = int(item[0])
                e = int(item[1])
                if s >= 1 and e >= 1:
                    ranges.append({"start": s, "end": e})
            except (ValueError, TypeError):
                continue
        elif isinstance(item, str):
            item_clean = item.strip()
            # Range format "3-7", "3 to 7", "3..7", "3:7"
            m = re.match(r"^(\d+)\s*(?:[-–:.]+|\s+to\s+)\s*(\d+)$", item_clean)
            if m:
                s = int(m.group(1))
                e = int(m.group(2))
                if s >= 1 and e >= 1:
                    ranges.append({"start": s, "end": e})
            else:
                m_single = re.match(r"^(\d+)$", item_clean)
                if m_single:
                    s = int(m_single.group(1))
                    if s >= 1:
                        ranges.append({"start": s, "end": s})

    return ranges[:4]


def _normalize_stat_keys(raw: Any) -> list[str]:
    """Lowercase and de-duplicate stat_keys, dropping unknown keys.

    Args:
        raw: Raw stat_keys list or comma-separated string.

    Returns:
        List of valid stat key strings (at most 4).
    """
    if not raw or raw in ("none", "null"):
        return []
    if isinstance(raw, str):
        items = [s.strip() for s in re.split(r"[,;]+", raw) if s.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw]
    else:
        return []

    cleaned: list[str] = []
    for item in items:
        k = item.lower()
        if k in VALID_STAT_KEYS and k not in cleaned:
            cleaned.append(k)
    return cleaned[:4]


def _normalize_architecture(raw: Any) -> dict[str, Any] | None:
    """Normalize architecture specification.

    Ensures node IDs are lowercase letters and digits, rewrites edges to match,
    drops edges pointing to missing nodes, and drops duplicate nodes.

    Args:
        raw: Raw architecture dictionary.

    Returns:
        Cleaned architecture dictionary or None.
    """
    if not isinstance(raw, dict):
        return None

    raw_nodes = raw.get("nodes", [])
    raw_edges = raw.get("edges", [])
    if not isinstance(raw_nodes, list):
        return None

    cleaned_nodes: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    id_mapping: dict[str, str] = {}

    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        orig_id = str(n.get("id", "")).strip().lower()
        clean_id = re.sub(r"[^a-z0-9]", "", orig_id)
        if not clean_id or clean_id in seen_node_ids:
            continue
        seen_node_ids.add(clean_id)
        id_mapping[orig_id] = clean_id
        id_mapping[clean_id] = clean_id

        raw_label = str(n.get("label", ""))
        label = _clean_common_text(raw_label)
        raw_group = str(n.get("group", "core")).strip().lower()
        group = raw_group if raw_group in VALID_NODE_GROUPS else "core"

        cleaned_nodes.append(
            {
                "id": clean_id,
                "label": label,
                "group": group,
            }
        )

    cleaned_edges: list[dict[str, Any]] = []
    if isinstance(raw_edges, list):
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            s_raw = str(e.get("source", "")).strip().lower()
            t_raw = str(e.get("target", "")).strip().lower()

            src = id_mapping.get(s_raw, re.sub(r"[^a-z0-9]", "", s_raw))
            tgt = id_mapping.get(t_raw, re.sub(r"[^a-z0-9]", "", t_raw))

            # Drop edge if source or target node is missing
            if src not in seen_node_ids or tgt not in seen_node_ids:
                continue

            raw_lbl = str(e.get("label", ""))
            lbl = _clean_common_text(raw_lbl)
            cleaned_edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "label": lbl,
                }
            )

    if not cleaned_nodes:
        return None

    return {
        "nodes": cleaned_nodes,
        "edges": cleaned_edges[:12],
    }


def normalize_script(data: object, repo: RepoData) -> dict[str, Any]:
    """Repair harmless LLM formatting mistakes without requiring extra model calls.

    Normalizes scene types, bullet markers, markdown emphasis, speech abbreviations,
    code fences, asset IDs, focus ranges, stat keys, and architecture specs.
    Renumbers scene IDs from 1 to N.

    Args:
        data: Raw parsed JSON data from the LLM (dict or list).
        repo: Repository data for default title and metadata fallback.

    Returns:
        A cleaned dictionary conforming to the ScriptData schema.

    Raises:
        ScriptError: If data is neither a dict nor a list.
    """
    if isinstance(data, list):
        scenes_raw = data
        title = repo.name
    elif isinstance(data, dict):
        raw_title = data.get("title")
        title = str(raw_title).strip() if raw_title and str(raw_title).strip() else repo.name
        scenes_data = data.get("scenes", [])
        if isinstance(scenes_data, list):
            scenes_raw = scenes_data
        elif isinstance(scenes_data, dict):
            scenes_raw = [scenes_data]
        else:
            scenes_raw = []
    else:
        raise ScriptError(
            "Model output is not a JSON object or list.",
            hint="The LLM must return a JSON object with 'title' and 'scenes'.",
        )

    normalized_scenes: list[dict[str, Any]] = []

    for idx, item in enumerate(scenes_raw, start=1):
        if not isinstance(item, dict):
            continue

        raw_type = str(item.get("type", "")).strip().lower()
        scene_type = TYPE_SYNONYMS.get(raw_type, raw_type)

        raw_heading = str(item.get("heading", ""))
        heading = _clean_common_text(raw_heading)

        # Bullets processing
        raw_bullets = item.get("bullets")
        bullet_list: list[str] = []
        if isinstance(raw_bullets, str):
            bullet_list = re.split(r"[\r\n;]+", raw_bullets)
        elif isinstance(raw_bullets, list):
            bullet_list = [str(b) for b in raw_bullets]

        cleaned_bullets: list[str] = []
        for b in bullet_list:
            b_text = _clean_common_text(b)
            # Strip bullet prefixes iteratively
            while True:
                stripped = LEADING_BULLET_MARKER.sub("", b_text).strip()
                if stripped == b_text:
                    break
                b_text = stripped
            if b_text:
                cleaned_bullets.append(b_text)
            if len(cleaned_bullets) == 4:
                break

        raw_narration = str(item.get("narration", ""))
        narration = _clean_narration_text(raw_narration)

        code = _clean_code_snippet(item.get("code"))

        # Visual v2 scene fields
        media_id = _clean_id(item.get("media_id"))
        file_id = _clean_id(item.get("file_id"))
        focus = _normalize_focus(item.get("focus"))
        terminal_id = _clean_id(item.get("terminal_id"))
        arch_data = _normalize_architecture(item.get("architecture"))
        stat_keys = _normalize_stat_keys(item.get("stat_keys"))

        normalized_scenes.append(
            {
                "id": idx,
                "type": scene_type,
                "heading": heading,
                "bullets": cleaned_bullets,
                "narration": narration,
                "code": code,
                "media_id": media_id,
                "file_id": file_id,
                "focus": focus,
                "terminal_id": terminal_id,
                "architecture": arch_data,
                "stat_keys": stat_keys,
            }
        )

    return {
        "title": title,
        "scenes": normalized_scenes,
    }


def _has_stat_data(key: str, repo: RepoData) -> bool:
    """Check if repository metadata contains data for the given stat key."""
    if key == "stars":
        return repo.stars is not None
    if key == "forks":
        return repo.forks is not None
    if key == "issues":
        return repo.open_issues is not None
    if key == "license":
        return bool(repo.license and repo.license.strip())
    if key == "updated":
        return bool(repo.pushed_at or repo.created_at)
    if key == "languages":
        return bool(repo.languages or repo.language_percentages)
    if key == "topics":
        return bool(repo.topics)
    return False


def resolve_references(
    script: ScriptData,
    repo: RepoData,
    media_assets: list[Any] | None = None,
) -> ScriptData:
    """Validate visual asset references, degrading invalid scenes with warnings.

    Runs after initial model validation and before check_facts. Ensures scenes
    with invalid, hallucinated, or unfulfillable references gracefully degrade
    to plain scenes without failing the overall script generation.

    Args:
        script: Validated ScriptData instance.
        repo: Repository data containing key files, terminals, stats, and media.
        media_assets: Optional list of downloaded MediaAsset objects.

    Returns:
        A sanitized ScriptData instance with resolved references and valid structure.
    """
    # 1. Collect available reference identifiers
    available_media_ids: set[str] = set()
    if media_assets:
        available_media_ids.update(getattr(a, "id", "") for a in media_assets)
    if repo.media:
        available_media_ids.update(m.id for m in repo.media)

    available_files: dict[str, KeyFile] = {kf.id: kf for kf in repo.key_files}
    available_terminals: dict[str, TerminalSession] = {t.id: t for t in repo.terminals}

    # Reference vocabulary for architecture node label matching
    corpus_parts = [repo.name, *repo.top_level, *(kf.path for kf in repo.key_files), repo.readme]
    repo_vocab: set[str] = set()
    for text in corpus_parts:
        repo_vocab.update(re.findall(r"[a-z0-9]+", text.lower()))

    # 2. Inspect each scene and degrade on reference problems
    for scene in script.scenes:
        if scene.type == "media":
            if not scene.media_id or scene.media_id not in available_media_ids:
                logger.warning(
                    "Scene %d: media_id '%s' not found among available media; degrading to feature.",
                    scene.id,
                    scene.media_id,
                )
                scene.type = "feature"
                scene.media_id = None

        elif scene.type == "code":
            if not scene.file_id or scene.file_id not in available_files:
                logger.warning(
                    "Scene %d: file_id '%s' not found among key files; degrading to feature.",
                    scene.id,
                    scene.file_id,
                )
                scene.type = "feature"
                scene.file_id = None
                scene.focus = []
            else:
                kf = available_files[scene.file_id]
                total_snippet_lines = len(kf.content.splitlines()) if kf.content else 0
                if scene.focus:
                    clipped_focus: list[FocusRange] = []
                    for fr in scene.focus:
                        if fr.start > total_snippet_lines:
                            logger.warning(
                                "Scene %d: focus range start %d is outside snippet (%d lines); dropping range.",
                                scene.id,
                                fr.start,
                                total_snippet_lines,
                            )
                            continue
                        clipped_end = min(fr.end, total_snippet_lines)
                        if clipped_end - fr.start + 1 > 8:
                            clipped_end = fr.start + 7
                        if fr.start <= clipped_end:
                            if clipped_end != fr.end:
                                logger.warning(
                                    "Scene %d: focus range end %d clipped to %d.",
                                    scene.id,
                                    fr.end,
                                    clipped_end,
                                )
                            clipped_focus.append(FocusRange(start=fr.start, end=clipped_end))
                    if not clipped_focus and scene.focus:
                        logger.warning(
                            "Scene %d: no valid focus ranges remain; showing whole snippet.",
                            scene.id,
                        )
                    scene.focus = clipped_focus

        elif scene.type == "terminal":
            if not scene.terminal_id or scene.terminal_id not in available_terminals:
                logger.warning(
                    "Scene %d: terminal_id '%s' not found among terminal sessions; degrading to howto.",
                    scene.id,
                    scene.terminal_id,
                )
                scene.type = "howto"
                scene.terminal_id = None

        elif scene.type == "architecture":
            if not scene.architecture:
                logger.warning(
                    "Scene %d: architecture spec missing; degrading to feature.",
                    scene.id,
                )
                scene.type = "feature"
                scene.architecture = None
            else:
                surviving_nodes: list[NodeSpec] = []
                for node in scene.architecture.nodes:
                    node_words = set(re.findall(r"[a-z0-9]+", node.label.lower()))
                    if node_words & repo_vocab:
                        surviving_nodes.append(node)
                    else:
                        logger.warning(
                            "Scene %d: architecture node '%s' (label: '%s') has no matching words in repository data; removing node.",
                            scene.id,
                            node.id,
                            node.label,
                        )

                if len(surviving_nodes) < 3:
                    logger.warning(
                        "Scene %d: only %d architecture nodes survived (< 3); degrading to feature.",
                        scene.id,
                        len(surviving_nodes),
                    )
                    scene.type = "feature"
                    scene.architecture = None
                else:
                    surviving_ids = {n.id for n in surviving_nodes}
                    filtered_edges = [
                        e
                        for e in scene.architecture.edges
                        if e.source in surviving_ids and e.target in surviving_ids
                    ]

                    if len(filtered_edges) >= 2:
                        connected = {e.source for e in filtered_edges} | {
                            e.target for e in filtered_edges
                        }
                        isolated = surviving_ids - connected
                        if isolated:
                            connected_nodes = [n for n in surviving_nodes if n.id not in isolated]
                            if len(connected_nodes) >= 3:
                                surviving_nodes = connected_nodes
                                surviving_ids = {n.id for n in surviving_nodes}
                            else:
                                logger.warning(
                                    "Scene %d: isolated nodes in architecture cannot be resolved with >= 3 nodes; degrading to feature.",
                                    scene.id,
                                )
                                scene.type = "feature"
                                scene.architecture = None

                    if scene.type == "architecture":
                        scene.architecture = ArchitectureSpec(
                            nodes=surviving_nodes, edges=filtered_edges
                        )

        elif scene.type == "stats":
            surviving_keys: list[str] = []
            for k in scene.stat_keys:
                if _has_stat_data(k, repo):
                    surviving_keys.append(k)
                else:
                    logger.warning(
                        "Scene %d: stat key '%s' has empty data in repository; removing key.",
                        scene.id,
                        k,
                    )

            if len(surviving_keys) < 2:
                logger.warning(
                    "Scene %d: stats scene has fewer than 2 valid stat keys; degrading to feature.",
                    scene.id,
                )
                scene.type = "feature"
                scene.stat_keys = []
            else:
                scene.stat_keys = surviving_keys

    # 3. Clear visual fields on degraded plain scenes
    for s in script.scenes:
        if s.type in ("title", "problem", "feature", "howto", "outro"):
            s.media_id = None
            s.file_id = None
            s.focus = []
            s.terminal_id = None
            s.architecture = None
            s.stat_keys = []

    # 4. Re-run structure checks (first and last type, id order)
    if script.scenes:
        if script.scenes[0].type != "title":
            logger.warning(
                "Scene %d: first scene had type '%s'; fixing to 'title'.",
                script.scenes[0].id,
                script.scenes[0].type,
            )
            script.scenes[0].type = "title"

        if script.scenes[-1].type != "outro":
            logger.warning(
                "Scene %d: last scene had type '%s'; fixing to 'outro'.",
                script.scenes[-1].id,
                script.scenes[-1].type,
            )
            script.scenes[-1].type = "outro"

        for idx, s in enumerate(script.scenes, start=1):
            s.id = idx

    # If all visual scenes degraded to plain and script has > 8 scenes, trim to 8
    has_v2 = any(s.type in V2_SCENE_TYPES for s in script.scenes)
    if not has_v2 and len(script.scenes) > 8:
        excess = len(script.scenes) - 8
        logger.warning(
            "Script has %d plain scenes after degradation (> 8); trimming %d scenes.",
            len(script.scenes),
            excess,
        )
        for _ in range(excess):
            script.scenes.pop(-2)
        for idx, s in enumerate(script.scenes, start=1):
            s.id = idx

    return script


def check_facts(script: ScriptData, repo: RepoData) -> ScriptData:
    """Verify that script code snippets, numbers, and architecture labels match repository data.

    Clears hallucinated code lines and logs warnings for unrecognized numbers or labels.

    Args:
        script: Validated ScriptData instance.
        repo: Analyzed RepoData source of truth.

    Returns:
        The verified and sanitized ScriptData instance.
    """
    # 1. Code snippet verification against repository README
    normalized_readme = " ".join(repo.readme.split())

    for scene in script.scenes:
        if scene.code:
            code_lines = [line.strip() for line in scene.code.splitlines() if line.strip()]
            code_valid = True
            for line in code_lines:
                normalized_line = " ".join(line.split())
                if normalized_line not in normalized_readme:
                    code_valid = False
                    break
            if not code_valid:
                logger.warning(
                    "Scene %d: code snippet does not appear in README; clearing code.",
                    scene.id,
                )
                scene.code = ""

    # 2. Number verification against repository metadata, stats, terminals, and README
    corpus = (
        f"{repo.readme} {repo.description} {repo.name} {repo.stars} {repo.forks} {repo.open_issues}"
    )
    for lang, pct in repo.language_percentages.items():
        corpus += f" {lang} {pct:.1f} {round(pct)}"
    for t in repo.terminals:
        corpus += f" {' '.join(t.commands)} {' '.join(t.output)}"
    for kf in repo.key_files:
        corpus += f" {kf.path} {kf.total_lines} {kf.content}"

    known_numbers = set(re.findall(r"\d+", corpus))

    for scene in script.scenes:
        scene_elements = [scene.heading] + scene.bullets + [scene.narration]
        scene_numbers = re.findall(r"\d+", " ".join(scene_elements))
        for num in scene_numbers:
            if num not in known_numbers:
                if scene.type == "stats":
                    logger.warning(
                        "Scene %d: stat narration number '%s' cannot be found in repository data.",
                        scene.id,
                        num,
                    )
                else:
                    logger.warning(
                        "Scene %d: number '%s' does not appear in repository metadata or README.",
                        scene.id,
                        num,
                    )

    # 3. Architecture label verification against repository corpus
    repo_words = set(re.findall(r"[a-z0-9]+", corpus.lower()))
    for scene in script.scenes:
        if scene.type == "architecture" and scene.architecture:
            for node in scene.architecture.nodes:
                node_words = set(re.findall(r"[a-z0-9]+", node.label.lower()))
                if not (node_words & repo_words):
                    logger.warning(
                        "Scene %d: architecture label '%s' cannot be found in repository data.",
                        scene.id,
                        node.label,
                    )

    return script


def estimate_duration_seconds(script: ScriptData) -> float:
    """Calculate the estimated narration duration in seconds.

    Assumes a spoken rate of approximately 150 words per minute (2.5 words/second).

    Args:
        script: ScriptData containing scenes with narration.

    Returns:
        Estimated duration in seconds.
    """
    total_words = sum(word_count(s.narration) for s in script.scenes)
    return total_words / 2.5


def write_script(
    repo: RepoData,
    client: LLMClient,
    settings: Settings | None = None,
    media_assets: list[Any] | None = None,
) -> ScriptData:
    """Generate and validate a structured video script from repository metadata.

    Implements a 3-attempt repair loop that feeds validation errors back to the
    LLM, resolves visual asset references gracefully, and saves raw output for
    inspection on failure.

    Args:
        repo: Repository data to summarize.
        client: Active LLM client implementation.
        settings: Application settings. If None, loaded via get_settings().
        media_assets: Optional list of downloaded MediaAsset objects.

    Returns:
        A validated, resolved, and fact-checked ScriptData instance.

    Raises:
        ScriptError: If the LLM cannot produce a valid script within 3 attempts.
        LLMError: If a connection or server error occurs during chat.
    """
    cfg = settings if settings is not None else get_settings()
    configured_temp = cfg.llm_temperature
    output_dir = Path(cfg.output_dir)

    raw_answers: list[str] = []
    previous_answer = ""
    problem_text = ""

    for attempt in range(1, 4):
        if attempt == 1:
            user_prompt = build_user_message(repo, media_assets=media_assets)
            current_temp = configured_temp
        else:
            truncated_previous = previous_answer[-6000:]
            user_prompt = REPAIR_PROMPT.format(
                error=problem_text,
                previous=truncated_previous,
            )
            current_temp = min(configured_temp, 0.2)

        # Connection and server errors from chat propagate immediately without retry
        raw_answer = client.chat(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=current_temp,
        )
        raw_answers.append(raw_answer)
        previous_answer = raw_answer

        # Pipeline: extract_json -> json.loads -> normalize_script -> model_validate
        try:
            json_str = extract_json(raw_answer)
        except LLMError:
            problem_text = "The answer was not a complete JSON object."
            continue

        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as exc:
            problem_text = f"JSON syntax error: {exc.msg} at line {exc.lineno} column {exc.colno}."
            continue

        try:
            normalized_data = normalize_script(parsed_json, repo)
        except ScriptError as exc:
            problem_text = exc.message
            continue

        try:
            script = ScriptData.model_validate(normalized_data)
        except ValidationError as exc:
            problem_text = format_validation_error(exc)
            continue

        # Degrade invalid asset references to plain scenes with warnings
        resolved_script = resolve_references(script, repo, media_assets=media_assets)

        # Validation passed; check facts and return
        final_script = check_facts(resolved_script, repo)
        duration = estimate_duration_seconds(final_script)

        type_counts = Counter(s.type for s in final_script.scenes)
        counts_str = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))

        logger.info(
            "Script generation succeeded on attempt %d: %d scenes (%s), estimated duration %.1fs",
            attempt,
            len(final_script.scenes),
            counts_str,
            duration,
        )

        if duration < 30.0 or duration > 100.0:
            logger.warning(
                "Estimated video duration %.1fs is outside recommended range (30-100s).",
                duration,
            )

        return final_script

    # All 3 attempts failed; write debug files and raise ScriptError
    debug_dir = output_dir / "debug"
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for idx, ans in enumerate(raw_answers, start=1):
            debug_file = debug_dir / f"script_attempt_{idx}.txt"
            debug_file.write_text(ans, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write debug files to %s: %s", debug_dir, exc)

    raise ScriptError(
        "The model could not produce a valid script after 3 attempts.",
        hint=f"Check the raw model output in {debug_dir} or use a larger model.",
    )


def main() -> None:
    """CLI entrypoint for turning a repo.json into scenes.json."""
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python -m GRC.scriptwriter <path/to/repo.json>\n")
        sys.exit(1)

    repo_path = Path(sys.argv[1])
    try:
        repo = RepoData.load(repo_path)
        client = get_client()
        script = write_script(repo, client)

        out_path = repo_path.parent / "scenes.json"
        script.save(out_path)

        duration = estimate_duration_seconds(script)
        print(
            f"Generated {len(script.scenes)} scenes in {out_path} (estimated duration: {duration:.1f}s)"
        )
    except Repo2VideoError as err:
        sys.stderr.write(f"{err}\n")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Unexpected error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
