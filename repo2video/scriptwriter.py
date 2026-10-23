"""Script writer stage that transforms repository analysis into structured scenes using an LLM."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from repo2video.config import get_settings
from repo2video.errors import LLMError, Repo2VideoError, ScriptError
from repo2video.llm import extract_json, get_client
from repo2video.models import RepoData, ScriptData, format_validation_error, word_count
from repo2video.prompts import REPAIR_PROMPT, SYSTEM_PROMPT, build_user_message

if TYPE_CHECKING:
    from repo2video.config import Settings
    from repo2video.llm import LLMClient

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


def normalize_script(data: object, repo: RepoData) -> dict[str, Any]:
    """Repair harmless LLM formatting mistakes without requiring extra model calls.

    Normalizes scene types, bullet markers, markdown emphasis, speech abbreviations,
    and code fences. Renumbers scene IDs from 1 to N.

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

        normalized_scenes.append(
            {
                "id": idx,
                "type": scene_type,
                "heading": heading,
                "bullets": cleaned_bullets,
                "narration": narration,
                "code": code,
            }
        )

    return {
        "title": title,
        "scenes": normalized_scenes,
    }


def check_facts(script: ScriptData, repo: RepoData) -> ScriptData:
    """Verify that script code snippets and numeric claims match the repository data.

    Clears hallucinated code lines and logs warnings for unrecognized numbers.

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

    # 2. Number verification against repository metadata and README
    corpus = f"{repo.readme} {repo.description} {repo.name} {repo.stars}"
    known_numbers = set(re.findall(r"\d+", corpus))

    for scene in script.scenes:
        scene_elements = [scene.heading] + scene.bullets + [scene.narration]
        scene_numbers = re.findall(r"\d+", " ".join(scene_elements))
        for num in scene_numbers:
            if num not in known_numbers:
                logger.warning(
                    "Scene %d: number '%s' does not appear in repository metadata or README.",
                    scene.id,
                    num,
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
) -> ScriptData:
    """Generate and validate a structured video script from repository metadata.

    Implements a 3-attempt repair loop that feeds validation errors back to the
    LLM and saves raw output for inspection on failure.

    Args:
        repo: Repository data to summarize.
        client: Active LLM client implementation.
        settings: Application settings. If None, loaded via get_settings().

    Returns:
        A validated and fact-checked ScriptData instance.

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
            user_prompt = build_user_message(repo)
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

        # Validation passed; check facts and return
        final_script = check_facts(script, repo)
        duration = estimate_duration_seconds(final_script)

        logger.info(
            "Script generation succeeded on attempt %d: %d scenes, estimated duration %.1fs",
            attempt,
            len(final_script.scenes),
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
        sys.stderr.write("Usage: python -m repo2video.scriptwriter <path/to/repo.json>\n")
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
