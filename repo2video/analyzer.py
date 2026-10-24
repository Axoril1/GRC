"""Repository analyzer for fetching and processing GitHub repository metadata."""

from __future__ import annotations

import logging
import re
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from repo2video.config import get_settings
from repo2video.errors import AnalyzerError, Repo2VideoError
from repo2video.models import KeyFile, MediaRef, RepoData, TerminalSession

if TYPE_CHECKING:
    from repo2video.config import Settings

logger = logging.getLogger(__name__)

EXAMPLE_HINT = "Provide a valid GitHub repository URL, e.g. 'https://github.com/owner/repo'."

RESERVED_SEGMENTS = frozenset(
    {
        "about",
        "explore",
        "features",
        "login",
        "marketplace",
        "notifications",
        "orgs",
        "pricing",
        "settings",
        "sponsors",
        "topics",
    }
)

NOISE_ITEMS = frozenset(
    {
        ".git",
        ".github",
        ".gitignore",
        ".gitattributes",
        ".ds_store",
        "node_modules",
        "__pycache__",
        ".idea",
        ".vscode",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "cargo.lock",
    }
)

BADGE_KEYWORDS = (
    "shields.io",
    "img.shields.io",
    "badge.svg",
    "travis-ci",
    "codecov",
    "coveralls",
)

SKIP_MEDIA_HOSTS = frozenset(
    {
        "shields.io",
        "img.shields.io",
        "badge.fury.io",
        "badgen.net",
        "travis-ci.org",
        "travis-ci.com",
        "codecov.io",
        "coveralls.io",
        "api.codacy.com",
        "snyk.io",
        "opencollective.com",
        "contrib.rocks",
        "avatars.githubusercontent.com",
        "github.githubassets.com",
    }
)

COMMAND_LANGS = frozenset(
    {"bash", "sh", "shell", "zsh", "console", "terminal", "powershell", "cmd"}
)

INSTALL_PATTERNS = (
    "pip install",
    "pipx install",
    "npm install",
    "npm i",
    "yarn add",
    "pnpm add",
    "cargo install",
    "go install",
    "brew install",
    "git clone",
    "docker pull",
    "docker run",
    "apt install",
)

HIGH_VALUE_STEMS = frozenset(
    {"main", "app", "index", "cli", "server", "__main__", "run", "manage", "program", "lib"}
)

PREFERRED_PARENTS = frozenset({"src", "app", "lib", "cmd", "pkg"})

PENALTY_STEM_KEYWORDS = ("test", "spec", "config", "setup", "utils", "helpers")

ALLOWED_KEY_FILE_EXTENSIONS = frozenset(
    {
        "py",
        "js",
        "jsx",
        "ts",
        "tsx",
        "go",
        "rs",
        "java",
        "kt",
        "rb",
        "php",
        "c",
        "cpp",
        "h",
        "cs",
        "swift",
        "dart",
        "sh",
    }
)

SKIP_PATH_SEGMENTS = frozenset(
    {
        "node_modules",
        "vendor",
        "dist",
        "build",
        "test",
        "tests",
        "__tests__",
        "spec",
        "docs",
        "examples",
        "migrations",
        "generated",
        "fixtures",
    }
)

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "go": "go",
    "rs": "rust",
    "java": "java",
    "kt": "kotlin",
    "rb": "ruby",
    "php": "php",
    "c": "c",
    "cpp": "cpp",
    "h": "c",
    "cs": "csharp",
    "swift": "swift",
    "dart": "dart",
    "sh": "shell",
}


def _is_valid_github_name(name: str) -> bool:
    """Check if a name contains only letters, digits, hyphen, underscore, and dot.

    Args:
        name: Name string to check.

    Returns:
        True if all characters are valid, False otherwise.
    """
    if not name:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return all(c in allowed for c in name)


def parse_repo_url(url: str) -> tuple[str, str]:
    """Parse a GitHub repository URL into owner and repository name.

    Accepts full URLs with or without scheme, www subdomains, extra path parts
    (such as branch or file paths), query strings, fragments, and trailing slashes.
    Rejects non-GitHub URLs, empty strings, reserved segment paths, and invalid
    characters.

    Args:
        url: The repository URL or shorthand to parse.

    Returns:
        A tuple of (owner, repo_name).

    Raises:
        AnalyzerError: If the URL is invalid or malformed.
    """
    cleaned_url = url.strip()
    if not cleaned_url:
        raise AnalyzerError(
            "Repository URL cannot be empty.",
            hint=EXAMPLE_HINT,
        )

    # Prepend scheme if missing so urllib.parse works reliably
    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = f"https://{cleaned_url}"

    try:
        parsed = urllib.parse.urlsplit(cleaned_url)
    except Exception as exc:
        raise AnalyzerError(
            f"Failed to parse repository URL '{url}': {exc}",
            hint=EXAMPLE_HINT,
        ) from exc

    hostname = (parsed.hostname or "").lower()
    if hostname not in ("github.com", "www.github.com"):
        raise AnalyzerError(
            f"Unsupported host '{parsed.hostname or 'unknown'}'. Only github.com is supported.",
            hint=EXAMPLE_HINT,
        )

    path_segments = [seg for seg in parsed.path.split("/") if seg]
    if len(path_segments) < 2:
        raise AnalyzerError(
            f"Invalid repository path in URL '{url}'. Expected both owner and repository.",
            hint=EXAMPLE_HINT,
        )

    owner = path_segments[0]
    repo = path_segments[1]

    if owner.lower() in RESERVED_SEGMENTS:
        raise AnalyzerError(
            f"'{owner}' is a reserved GitHub path segment, not a valid repository owner.",
            hint=EXAMPLE_HINT,
        )

    if repo.endswith(".git") and len(repo) > 4:
        repo = repo[:-4]

    if not _is_valid_github_name(owner):
        raise AnalyzerError(
            f"Invalid repository owner '{owner}'. Only alphanumeric characters, hyphens, "
            f"underscores, and dots are permitted.",
            hint=EXAMPLE_HINT,
        )

    if not _is_valid_github_name(repo):
        raise AnalyzerError(
            f"Invalid repository name '{repo}'. Only alphanumeric characters, hyphens, "
            f"underscores, and dots are permitted.",
            hint=EXAMPLE_HINT,
        )

    return owner, repo


def _remove_badge_lines(text: str) -> str:
    """Remove lines that are badges or contain badge service URLs.

    Args:
        text: Raw markdown text.

    Returns:
        Markdown text with badge lines stripped.
    """
    result_lines: list[str] = []
    for line in text.split("\n"):
        line_lower = line.lower()
        if any(kw in line_lower for kw in BADGE_KEYWORDS):
            continue

        # Check if line is made only of markdown images or image links
        stripped = re.sub(r"\[\s*!\[.*?\]\(.*?\)\s*\]\(.*?\)", "", line)
        stripped = re.sub(r"!\[.*?\]\(.*?\)", "", stripped)
        stripped = re.sub(r"<img\b[^>]*\/?>", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"<a\b[^>]*>\s*</a>", "", stripped, flags=re.IGNORECASE)

        if "![" in line and not stripped.strip():
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


def _remove_html_media_and_base64(text: str) -> str:
    """Remove picture blocks, img tags, and base64 data URIs.

    Args:
        text: Markdown text.

    Returns:
        Cleaned markdown text.
    """
    # Remove picture blocks
    text = re.sub(r"<picture\b[^>]*>[\s\S]*?</picture>", "", text, flags=re.IGNORECASE)
    # Remove img tags
    text = re.sub(r"<img\b[^>]*\/?>", "", text, flags=re.IGNORECASE)
    # Remove base64 data URIs
    text = re.sub(r"data:[a-zA-Z0-9/+-]+;base64,[A-Za-z0-9+/=\s]+", "", text)
    return text


def _remove_table_of_contents(text: str) -> str:
    """Remove a Table of Contents section from markdown text.

    Finds a heading named 'Table of Contents' or 'Contents' followed by a list
    whose items contain markdown anchor links (starting with '#'), and removes
    the heading and its list items up to the next section or non-list block.

    Args:
        text: Markdown text.

    Returns:
        Markdown text with table of contents removed.
    """
    lines = text.split("\n")
    i = 0
    new_lines: list[str] = []

    toc_heading_pattern = re.compile(
        r"^(?:#{1,6}\s+|\*{2}|_{2})?(?:Table of Contents|Contents)(?:\*{2}|_{2})?:?\s*$",
        re.IGNORECASE,
    )
    toc_link_pattern = re.compile(r"\[.*?\]\(#.*?\)")

    while i < len(lines):
        line = lines[i]
        if toc_heading_pattern.match(line.strip()):
            j = i + 1
            # Skip blank lines after heading
            while j < len(lines) and not lines[j].strip():
                j += 1
            # Verify if following block is an anchor link list
            is_toc = False
            if j < len(lines):
                stripped_j = lines[j].strip()
                if (
                    stripped_j.startswith(("-", "*", "+"))
                    or (stripped_j and stripped_j[0].isdigit() and "." in stripped_j[:5])
                ) and toc_link_pattern.search(stripped_j):
                    is_toc = True

            if is_toc:
                while j < len(lines):
                    item = lines[j].strip()
                    if not item:
                        # Check lookahead to see if list continues after blank line
                        peek = j + 1
                        while peek < len(lines) and not lines[peek].strip():
                            peek += 1
                        if peek < len(lines):
                            peek_item = lines[peek].strip()
                            if (
                                peek_item.startswith(("-", "*", "+"))
                                or (peek_item and peek_item[0].isdigit() and "." in peek_item[:5])
                            ) and toc_link_pattern.search(peek_item):
                                j = peek
                                continue
                        break

                    if (
                        item.startswith(("-", "*", "+"))
                        or (item[0].isdigit() and "." in item[:5])
                        or lines[j].startswith(("  ", "\t"))
                    ):
                        j += 1
                    else:
                        break
                i = j
                continue

        new_lines.append(line)
        i += 1

    return "\n".join(new_lines)


def _replace_markdown_links_outside_code(text: str) -> str:
    """Replace markdown links [text](url) with 'text' outside code.

    Args:
        text: Markdown text.

    Returns:
        Text with normal links unwrapped to text, keeping code untouched.
    """
    pattern = re.compile(
        r"(?P<fenced>^(?:```+|~~~+)[^\n]*\n[\s\S]*?^(?:```+|~~~+)[^\n]*(?:\n|$))"
        r"|(?P<inline>`[^`\n]*`)"
        r"|(?P<image>!\[[^\]\n]*\]\([^)\n]*\))"
        r"|(?P<link>\[(?P<link_text>[^\]\n]+)\]\([^)\n]*\))",
        re.MULTILINE,
    )

    def replacer(match: re.Match[str]) -> str:
        if match.group("fenced"):
            return match.group("fenced")
        if match.group("inline"):
            return match.group("inline")
        if match.group("image"):
            return match.group("image")
        if match.group("link"):
            return match.group("link_text")
        return match.group(0)

    return pattern.sub(replacer, text)


def _find_fenced_code_blocks(text: str) -> list[tuple[int, int]]:
    """Find start and end character positions of all fenced code blocks.

    Args:
        text: Markdown text.

    Returns:
        List of (start_idx, end_idx) tuples for each fenced code block.
    """
    blocks: list[tuple[int, int]] = []
    pattern = re.compile(r"(?m)^(```+|~~~+)[^\n]*\n[\s\S]*?(?:^\1[^\n]*(?:\n|$)|$)")
    for match in pattern.finditer(text):
        blocks.append((match.start(), match.end()))
    return blocks


def _find_last_sentence_end(s: str) -> int:
    """Find the character index immediately after the last sentence end in s.

    Args:
        s: String to search.

    Returns:
        Index after the sentence end punctuation, or -1 if none found.
    """
    for i in range(len(s) - 1, -1, -1):
        if s[i] in ".!?" and (i == len(s) - 1 or s[i + 1] in " \t\n\r"):
            return i + 1
    return -1


def clean_readme(text: str, limit: int = 6000) -> str:
    """Clean and truncate repository README text to fit within a character limit.

    Processing order:
    1. Normalise line endings to newlines and strip leading byte order mark.
    2. Remove HTML comments.
    3. Remove badge lines.
    4. Remove HTML img tags, picture blocks, and base64 data URIs.
    5. Remove a table of contents.
    6. Replace markdown links [text](url) with just text, leaving code untouched.
    7. Collapse 3 or more blank lines into 2 and strip trailing spaces.
    8. Truncate at last blank line before limit (or sentence end if none after 50%),
       never cutting inside a fenced code block.

    Args:
        text: Raw README markdown text.
        limit: Maximum character length allowed (default 6000).

    Returns:
        Cleaned and truncated markdown text.
    """
    if not text:
        return ""

    # 1. Normalise line endings and strip BOM
    text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove HTML comments
    text = re.sub(r"<!--[\s\S]*?-->", "", text)

    # 3. Remove badge lines
    text = _remove_badge_lines(text)

    # 4. Remove HTML img tags, picture blocks, and base64 data URIs
    text = _remove_html_media_and_base64(text)

    # 5. Remove Table of Contents
    text = _remove_table_of_contents(text)

    # 6. Replace markdown links with text (leave inline and fenced code untouched)
    text = _replace_markdown_links_outside_code(text)

    # 7. Collapse 3+ blank lines into 2 and strip trailing whitespace on lines
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # 8. Truncate if still over limit
    if len(text) > limit:
        half_limit = limit // 2
        prefix = text[:limit]
        last_blank = prefix.rfind("\n\n")

        if last_blank >= half_limit:
            cut_pos = last_blank
        else:
            last_sentence = _find_last_sentence_end(prefix)
            if last_sentence > 0:
                cut_pos = last_sentence
            else:
                cut_pos = limit

        # Never cut inside a fenced code block
        fences = _find_fenced_code_blocks(text)
        for fence_start, fence_end in fences:
            if fence_start < cut_pos < fence_end:
                cut_pos = fence_start
                break

        text = text[:cut_pos].rstrip()

    return text


def resolve_relative_url(url: str, owner: str, repo: str, default_branch: str) -> str:
    """Resolve a relative URL against raw.githubusercontent.com for the repository.

    Args:
        url: Raw URL from the README (may be relative or absolute).
        owner: Repository owner.
        repo: Repository name.
        default_branch: Default git branch.

    Returns:
        Absolute resolved URL string.
    """
    clean_url = url.strip()
    if clean_url.startswith(("http://", "https://", "data:")):
        return clean_url

    path = clean_url.removeprefix("./").removeprefix("/").lstrip("/")

    # URL-encode spaces while preserving other valid URL characters
    encoded_path = path.replace(" ", "%20")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{encoded_path}"


def extract_media_refs(
    raw_readme: str,
    owner: str,
    repo: str,
    default_branch: str,
) -> list[MediaRef]:
    """Extract media references (images, gifs, videos) from raw README text in document order.

    Resolves relative URLs, skips badges, SVGs, data URIs, and known badge hosts,
    deduplicates by URL, and assigns identifiers m1, m2, up to 12 items.

    Args:
        raw_readme: Uncleaned, raw README markdown content.
        owner: Repository owner.
        repo: Repository name.
        default_branch: Default git branch.

    Returns:
        List of at most 12 MediaRef objects.
    """
    if not raw_readme:
        return []

    # Candidates: tuple of (start_position, raw_url, alt, forced_kind)
    candidates: list[tuple[int, str, str, str | None]] = []

    # 1. Markdown images: ![alt](url)
    md_img_pattern = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\(\s*<?(?P<url>[^\s\)>\"']+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)"
    )
    for m in md_img_pattern.finditer(raw_readme):
        candidates.append((m.start(), m.group("url"), m.group("alt"), None))

    # 2. HTML <img> tags: <img src="..." alt="...">
    html_img_pattern = re.compile(r"<img\b(?P<attrs>[^>]*)>", re.IGNORECASE)
    src_attr_pattern = re.compile(r"""\bsrc\s*=\s*["']?([^"'\s>]+)""", re.IGNORECASE)
    alt_attr_pattern = re.compile(r"""\balt\s*=\s*["']?([^"'>]*)""", re.IGNORECASE)

    for m in html_img_pattern.finditer(raw_readme):
        attrs = m.group("attrs")
        src_m = src_attr_pattern.search(attrs)
        if src_m:
            raw_url = src_m.group(1)
            alt_m = alt_attr_pattern.search(attrs)
            alt_text = alt_m.group(1) if alt_m else ""
            candidates.append((m.start(), raw_url, alt_text, None))

    # 3. HTML <video> and <source> tags: <video src="..."> or <source src="...">
    html_video_pattern = re.compile(r"<(?:video|source)\b(?P<attrs>[^>]*)>", re.IGNORECASE)
    for m in html_video_pattern.finditer(raw_readme):
        attrs = m.group("attrs")
        src_m = src_attr_pattern.search(attrs)
        if src_m:
            candidates.append((m.start(), src_m.group(1), "", "video"))

    # 4. Lines that contain only a github.com/user-attachments/assets/ link
    attachment_line_pattern = re.compile(
        r"^\s*(?P<url>https?://(?:www\.)?github\.com/user-attachments/assets/[a-zA-Z0-9_.-]+)\s*$",
        re.MULTILINE,
    )
    for m in attachment_line_pattern.finditer(raw_readme):
        candidates.append((m.start(), m.group("url"), "", "video"))

    # 5. Links ending in .mp4, .webm, or .mov: [alt](url.mp4)
    video_link_pattern = re.compile(
        r"\[(?P<alt>[^\]]*)\]\(\s*<?(?P<url>[^\s\)>\"']+\.(?:mp4|webm|mov)(?:\?[^\s\)>\"']*)?)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)",
        re.IGNORECASE,
    )
    for m in video_link_pattern.finditer(raw_readme):
        candidates.append((m.start(), m.group("url"), m.group("alt"), "video"))

    # Sort candidates strictly in README document order
    candidates.sort(key=lambda c: c[0])

    media_refs: list[MediaRef] = []
    seen_urls: set[str] = set()

    for _, raw_url, alt, forced_kind in candidates:
        url_clean = raw_url.strip().strip("<>\"'")
        if not url_clean:
            continue

        resolved = resolve_relative_url(url_clean, owner, repo, default_branch)

        # Skip data: URIs
        if resolved.lower().startswith("data:"):
            continue

        try:
            parsed = urllib.parse.urlsplit(resolved)
        except (ValueError, AttributeError):
            continue

        host = (parsed.hostname or "").lower()

        # Skip host check
        if host in SKIP_MEDIA_HOSTS or any(host.endswith("." + sh) for sh in SKIP_MEDIA_HOSTS):
            continue

        # Skip .svg files
        path_lower = parsed.path.lower()
        if path_lower.endswith(".svg") or ".svg" in resolved.lower():
            continue

        # Skip actions workflows badges
        if "/actions/workflows/" in resolved and "badge.svg" in resolved:
            continue

        # Skip emoji images
        res_lower = resolved.lower()
        if (
            "images/icons/emoji" in res_lower
            or "/emojis/" in res_lower
            or "githubassets.com/images/icons" in res_lower
        ):
            continue

        # Determine media kind
        if forced_kind == "video" or "user-attachments/assets" in resolved:
            kind = "video"
        elif path_lower.endswith(".gif") or ".gif?" in res_lower:
            kind = "gif"
        elif path_lower.endswith((".mp4", ".webm", ".mov")):
            kind = "video"
        else:
            kind = "image"

        # Deduplicate
        if resolved in seen_urls:
            continue
        seen_urls.add(resolved)

        media_refs.append(
            MediaRef(
                id=f"m{len(media_refs) + 1}",
                url=resolved,
                alt=alt.strip(),
                kind=kind,
            )
        )

        if len(media_refs) >= 12:
            break

    return media_refs


def extract_terminal_sessions(raw_readme: str) -> list[TerminalSession]:
    """Extract terminal command and output sessions from raw README text.

    Identifies command blocks by language tag or prompt characters, parses
    following output blocks, classifies purpose as install or usage, and returns
    at most 4 sessions (first install, then usage).

    Args:
        raw_readme: Raw markdown content of repository README.

    Returns:
        List of at most 4 TerminalSession instances.
    """
    if not raw_readme:
        return []

    fence_pattern = re.compile(
        r"```(?P<lang>[a-zA-Z0-9_-]*)[ \t]*\r?\n(?P<code>.*?)```",
        re.DOTALL,
    )

    blocks: list[dict[str, Any]] = []
    for m in fence_pattern.finditer(raw_readme):
        blocks.append(
            {
                "start": m.start(),
                "end": m.end(),
                "lang": m.group("lang").strip().lower(),
                "code": m.group("code"),
            }
        )

    parsed_sessions: list[dict[str, Any]] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        lang = b["lang"]
        code = b["code"]

        # Check if block qualifies as command block
        is_cmd = False
        if lang in COMMAND_LANGS:
            is_cmd = True
        elif not lang:
            non_empty = [line.strip() for line in code.splitlines() if line.strip()]
            if non_empty:
                prompt_count = sum(
                    1 for line in non_empty if line.startswith(("$ ", "> ", "$", ">"))
                )
                if prompt_count >= len(non_empty) / 2:
                    is_cmd = True

        if not is_cmd:
            i += 1
            continue

        # Clean command lines
        cleaned_cmds: list[str] = []
        skip_block = False
        for line in code.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            cmd_line = line
            if cmd_line.startswith(("$ ", "> ")):
                cmd_line = cmd_line[2:]
            elif cmd_line.startswith(("$", ">")):
                cmd_line = cmd_line[1:].lstrip()

            cmd_line = cmd_line.strip()
            if not cmd_line or cmd_line.startswith("#"):
                continue

            if len(cmd_line) > 100:
                skip_block = True
                break

            cleaned_cmds.append(cmd_line)

        # Skip block if more than 8 command lines, line > 100 chars, or empty
        if skip_block or len(cleaned_cmds) > 8 or len(cleaned_cmds) == 0:
            i += 1
            continue

        # Keep 1 to 4 commands
        commands = cleaned_cmds[:4]

        # Classify purpose
        is_install = any(
            any(pat in cmd.lower() for pat in INSTALL_PATTERNS) for cmd in cleaned_cmds
        )
        purpose = "install" if is_install else "usage"

        # Look ahead for paired output block
        output_lines: list[str] = []
        if i + 1 < len(blocks):
            next_b = blocks[i + 1]
            between_text = raw_readme[b["end"] : next_b["start"]]
            non_empty_between = [line for line in between_text.splitlines() if line.strip()]

            if len(non_empty_between) <= 3:
                next_lang = next_b["lang"]
                if next_lang in ("text", "console", "none", ""):
                    next_code_lines = [line for line in next_b["code"].splitlines() if line.strip()]
                    # Check if next block lines do not look like commands
                    looks_like_cmds = any(
                        line.startswith(("$ ", "> "))
                        or any(pat in line.lower() for pat in INSTALL_PATTERNS)
                        for line in next_code_lines
                    )
                    if not looks_like_cmds:
                        output_lines = [line[:100] for line in next_code_lines[:8]]
                        # Consume the output block
                        i += 1

        parsed_sessions.append(
            {
                "purpose": purpose,
                "commands": commands,
                "output": output_lines,
            }
        )
        i += 1

    # Keep at most 4 sessions: the first install session, then usage sessions
    install_sessions = [s for s in parsed_sessions if s["purpose"] == "install"]
    usage_sessions = [s for s in parsed_sessions if s["purpose"] == "usage"]

    selected: list[dict[str, Any]] = []
    if install_sessions:
        selected.append(install_sessions[0])
        selected.extend(usage_sessions[:3])
    else:
        selected.extend(usage_sessions[:4])

    return [
        TerminalSession(
            id=f"t{idx}",
            purpose=s["purpose"],
            commands=s["commands"],
            output=s["output"],
        )
        for idx, s in enumerate(selected, start=1)
    ]


def score_key_file(
    path: str,
    repo_name: str,
    top_language: str = "",
) -> int:
    """Score a candidate repository source file for relevance as a key code file.

    Args:
        path: Relative file path in the repository.
        repo_name: Repository name.
        top_language: Primary programming language of the repository.

    Returns:
        Integer score representing relevance.
    """
    stem = Path(path).stem.lower()
    parts = path.split("/")
    parent_folder = parts[-2].lower() if len(parts) > 1 else ""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

    score = 0
    if stem in HIGH_VALUE_STEMS:
        score += 40

    if parent_folder in PREFERRED_PARENTS or parent_folder == repo_name.lower():
        score += 15

    if stem == repo_name.lower():
        score += 10

    # +10 when extension matches repository's top language
    lang_for_ext = EXTENSION_TO_LANGUAGE.get(ext, "")
    if top_language and (
        lang_for_ext == top_language.lower()
        or ext == top_language.lower()
        or top_language.lower() in (lang_for_ext, ext)
    ):
        score += 10

    # -5 for each folder level beyond the first
    folder_levels = len(parts) - 1
    if folder_levels > 1:
        score -= (folder_levels - 1) * 5

    # -30 when stem contains test, spec, config, setup, utils or helpers
    if any(kw in stem for kw in PENALTY_STEM_KEYWORDS):
        score -= 30

    return score


def prepare_file_content(raw_text: str) -> tuple[str, int]:
    """Prepare and clean file content excerpt for KeyFile.

    Converts tabs to 4 spaces, drops leading license comment header (>3 lines),
    caps at 40 lines, 110 chars per line, and 2500 chars at line boundary.

    Args:
        raw_text: Raw file content fetched from GitHub.

    Returns:
        Tuple of (cleaned_content, total_lines).
    """
    total_lines = len(raw_text.splitlines())
    text = raw_text.replace("\t", "    ")
    lines = text.splitlines()

    # Strip leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)

    # Check for leading comment block > 3 lines
    comment_lines_count = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith(("#", "//", "/*", "*", "*/", '"""', "'''")):
            comment_lines_count += 1
        else:
            break

    if comment_lines_count > 3:
        idx = 0
        while idx < len(lines):
            s = lines[idx].strip()
            if not s or s.startswith(("#", "//", "/*", "*", "*/", '"""', "'''")):
                idx += 1
            else:
                break
        lines = lines[idx:]
        while lines and not lines[0].strip():
            lines.pop(0)

    # Keep first 40 lines
    lines = lines[:40]

    # Cut each line to 110 characters
    lines = [line[:110] for line in lines]

    # Cut to 2500 characters at a line boundary
    kept_lines: list[str] = []
    current_chars = 0
    for line in lines:
        added = len(line) + (1 if kept_lines else 0)
        if current_chars + added > 2500:
            break
        kept_lines.append(line)
        current_chars += added

    return "\n".join(kept_lines), total_lines


def _calculate_language_percentages(raw_languages: dict[str, Any]) -> dict[str, float]:
    """Calculate byte percentages for the top 5 languages.

    Args:
        raw_languages: Mapping of language name to byte count.

    Returns:
        Dictionary mapping language name to percentage float.
    """
    valid_items = [
        (lang, count)
        for lang, count in raw_languages.items()
        if isinstance(count, (int, float)) and count > 0
    ]
    if not valid_items:
        return {}

    total_bytes = sum(count for _, count in valid_items)
    if total_bytes <= 0:
        return {}

    sorted_langs = sorted(valid_items, key=lambda x: x[1], reverse=True)[:5]
    return {lang: round((count / total_bytes) * 100.0, 1) for lang, count in sorted_langs}


def _process_languages(data: dict[str, int]) -> list[str]:
    """Process languages dictionary from GitHub REST API.

    Sorts by bytes descending, drops languages below 2% of total bytes
    unless that would leave none, and keeps at most 5 names.

    Args:
        data: Mapping of language name to byte count.

    Returns:
        List of at most 5 language names.
    """
    if not data:
        return []

    sorted_langs = sorted(data.items(), key=lambda item: item[1], reverse=True)
    total_bytes = sum(data.values())

    if total_bytes == 0:
        return [name for name, _ in sorted_langs[:5]]

    threshold = total_bytes * 0.02
    filtered = [name for name, count in sorted_langs if count >= threshold]

    if not filtered and sorted_langs:
        filtered = [sorted_langs[0][0]]

    return [name for name in filtered[:5]]


def _process_tree(tree_items: list[dict[str, Any]]) -> list[str]:
    """Filter and sort top-level directory items from GitHub Git Trees API.

    Excludes hidden files, version control metadata, cache directories, and lockfiles.
    Sorts directories first, then files alphabetically, and caps at 30 items.

    Args:
        tree_items: List of item dictionaries from the tree response.

    Returns:
        Filtered list of at most 30 top-level file and directory names.
    """
    candidates: list[tuple[bool, str]] = []

    for item in tree_items:
        path = str(item.get("path", ""))
        # Only consider top-level entries (no forward slash in path)
        if "/" in path or not path:
            continue

        path_lower = path.lower()
        if path_lower in NOISE_ITEMS or path.startswith("."):
            continue

        is_tree = item.get("type") == "tree"
        candidates.append((is_tree, path))

    # Sort directories first (is_tree=True), then alphabetically case-insensitive
    candidates.sort(key=lambda x: (not x[0], x[1].lower()))
    return [name for _, name in candidates[:30]]


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> requests.Response:
    """Execute an HTTP request with exponential backoff on transient errors.

    Retries on 502, 503, 504 and connection errors up to max_retries times.

    Args:
        session: Active requests session.
        method: HTTP method string (e.g. 'GET').
        url: Target URL.
        headers: Optional request headers.
        max_retries: Maximum number of retry attempts.
        base_backoff: Initial sleep duration in seconds before backoff.

    Returns:
        The successful or final HTTP Response.

    Raises:
        AnalyzerError: If all retry attempts fail due to network error.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = session.request(method, url, headers=headers, timeout=30.0)
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise AnalyzerError(
                    f"Network error communicating with GitHub: {exc}",
                    hint="Check your internet connection or GitHub status.",
                ) from exc
            sleep_time = base_backoff * (2**attempt)
            logger.warning(
                "Request to %s failed (%s). Retrying in %.1fs (attempt %d/%d)...",
                url,
                exc,
                sleep_time,
                attempt + 1,
                max_retries + 1,
            )
            time.sleep(sleep_time)
            continue

        if resp.status_code in (502, 503, 504) and attempt < max_retries:
            sleep_time = base_backoff * (2**attempt)
            logger.warning(
                "GitHub returned HTTP %d for %s. Retrying in %.1fs (attempt %d/%d)...",
                resp.status_code,
                url,
                sleep_time,
                attempt + 1,
                max_retries + 1,
            )
            time.sleep(sleep_time)
            continue

        return resp

    raise AnalyzerError(
        "Could not reach GitHub.",
        hint="Check your internet connection.",
    )


def _check_rate_limit(resp: requests.Response, settings: Settings) -> None:
    """Log a warning when GitHub API rate limit is low and no token is configured."""
    if settings.github_token:
        return
    rem = resp.headers.get("X-RateLimit-Remaining")
    if rem is not None and rem.isdigit():
        val = int(rem)
        if val < 10:
            logger.warning(
                "GitHub API rate limit remaining is low (%d remaining). "
                "Consider setting GITHUB_TOKEN to raise the limit from 60 to 5000 requests per hour.",
                val,
            )


def _handle_response_errors(resp: requests.Response, is_tree: bool = False) -> None:
    """Validate HTTP response status and map errors to AnalyzerError.

    Args:
        resp: The response to inspect.
        is_tree: Whether this call is the tree endpoint (409 indicates empty repo).

    Raises:
        AnalyzerError: If status_code indicates an error.
    """
    if resp.status_code < 400:
        return

    if resp.status_code == 401:
        raise AnalyzerError(
            "The GitHub personal access token was rejected.",
            hint="Remove or renew GITHUB_TOKEN.",
        )

    if resp.status_code in (403, 429) and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset_val = resp.headers.get("X-RateLimit-Reset")
        reset_time_str = "soon"
        if reset_val and reset_val.isdigit():
            try:
                dt = datetime.fromtimestamp(int(reset_val)).astimezone()
                reset_time_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except (ValueError, OSError):
                reset_time_str = "soon"
        raise AnalyzerError(
            f"GitHub API rate limit reached. Limit resets at {reset_time_str}.",
            hint="Set GITHUB_TOKEN (a token with no scopes is enough) to raise the limit "
            "from 60 to 5000 requests per hour.",
        )

    if resp.status_code == 403:
        gh_msg = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                gh_msg = str(body.get("message", ""))
        except (ValueError, requests.exceptions.JSONDecodeError):
            gh_msg = ""
        msg = f"Access forbidden by GitHub: {gh_msg}" if gh_msg else "Access forbidden by GitHub."
        raise AnalyzerError(
            msg,
            hint="Check repository access permissions or GitHub service status.",
        )

    if resp.status_code == 404:
        raise AnalyzerError(
            "Repository not found, or it is private.",
            hint="Check the spelling; private repositories are not supported.",
        )

    if resp.status_code == 409 and is_tree:
        raise AnalyzerError(
            "The repository is empty.",
            hint="Ensure the repository has at least one commit and files.",
        )

    if resp.status_code >= 500:
        raise AnalyzerError(
            f"GitHub service error (HTTP {resp.status_code}).",
            hint="Please try again later or check GitHub service status.",
        )

    raise AnalyzerError(
        f"GitHub API error (HTTP {resp.status_code}).",
        hint="Check the repository URL or try again later.",
    )


def analyze(
    url: str,
    settings: Settings | None = None,
    session: requests.Session | None = None,
) -> RepoData:
    """Analyze a public GitHub repository and return structured RepoData.

    Fetches repository metadata, README, language statistics, directory structure,
    media references, terminal sessions, and key source code files using the GitHub
    REST API.

    Args:
        url: The GitHub repository URL to analyze.
        settings: Application settings containing optional GitHub token.
        session: Optional injected requests.Session for testing.

    Returns:
        Validated RepoData model containing repository details.

    Raises:
        AnalyzerError: If repository is missing, empty, or API errors occur.
    """
    resolved_settings = settings or get_settings()
    owner, repo = parse_repo_url(url)

    sess = session or requests.Session()
    should_close = session is None

    # Base headers
    sess.headers["User-Agent"] = "repo2video/0.1"
    sess.headers["Accept"] = "application/vnd.github+json"
    sess.headers["X-GitHub-Api-Version"] = "2022-11-28"
    if resolved_settings.github_token:
        sess.headers["Authorization"] = f"Bearer {resolved_settings.github_token}"

    request_count = 0

    try:
        # 1. GET /repos/{owner}/{repo}
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        repo_resp = _request_with_retry(sess, "GET", repo_url)
        request_count += 1
        _check_rate_limit(repo_resp, resolved_settings)
        _handle_response_errors(repo_resp)

        repo_info = repo_resp.json()
        canonical_name: str = repo_info["name"]
        canonical_owner: str = repo_info["owner"]["login"]
        description: str = repo_info.get("description") or ""
        stars: int = repo_info.get("stargazers_count", 0)
        default_branch: str = repo_info.get("default_branch", "main")
        homepage: str = str(repo_info.get("homepage") or "").strip()
        raw_topics: list[Any] = repo_info.get("topics") or []
        topics: list[str] = [str(t).strip() for t in raw_topics if str(t).strip()][:10]

        license_str = ""
        license_obj = repo_info.get("license")
        if isinstance(license_obj, dict):
            spdx_id = str(license_obj.get("spdx_id") or "").strip()
            if spdx_id and spdx_id != "NOASSERTION":
                license_str = spdx_id

        forks: int = int(repo_info.get("forks_count", 0))
        open_issues: int = int(repo_info.get("open_issues_count", 0))
        created_at: str = str(repo_info.get("created_at") or "").strip()
        pushed_at: str = str(repo_info.get("pushed_at") or "").strip()

        logger.info(
            "Repository metadata retrieved: %s/%s (stars: %d, default branch: %s)",
            canonical_owner,
            canonical_name,
            stars,
            default_branch,
        )

        # 2. GET /repos/{owner}/{repo}/readme
        readme_url = f"https://api.github.com/repos/{canonical_owner}/{canonical_name}/readme"
        readme_resp = _request_with_retry(
            sess,
            "GET",
            readme_url,
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        request_count += 1
        _check_rate_limit(readme_resp, resolved_settings)

        raw_readme = ""
        if readme_resp.status_code == 404:
            logger.warning("No README found for %s/%s", canonical_owner, canonical_name)
        else:
            _handle_response_errors(readme_resp)
            raw_readme = readme_resp.text

        # Extract media and terminal sessions from RAW README
        media = extract_media_refs(raw_readme, canonical_owner, canonical_name, default_branch)
        terminals = extract_terminal_sessions(raw_readme)

        cleaned_readme = clean_readme(raw_readme, limit=6000)
        logger.info(
            "README processed: %d characters before cleaning, %d characters after, "
            "%d media references, %d terminal sessions",
            len(raw_readme),
            len(cleaned_readme),
            len(media),
            len(terminals),
        )

        # 3. GET /repos/{owner}/{repo}/languages
        langs_url = f"https://api.github.com/repos/{canonical_owner}/{canonical_name}/languages"
        langs_resp = _request_with_retry(sess, "GET", langs_url)
        request_count += 1
        _check_rate_limit(langs_resp, resolved_settings)
        _handle_response_errors(langs_resp)

        raw_langs_data = langs_resp.json()
        languages = _process_languages(raw_langs_data)
        language_percentages = _calculate_language_percentages(raw_langs_data)
        logger.info("Languages analyzed: %d languages selected", len(languages))

        # 4. GET /repos/{owner}/{repo}/git/trees/{default_branch} (top-level)
        tree_url = (
            f"https://api.github.com/repos/{canonical_owner}/{canonical_name}/"
            f"git/trees/{default_branch}"
        )
        tree_resp = _request_with_retry(sess, "GET", tree_url)
        request_count += 1
        _check_rate_limit(tree_resp, resolved_settings)

        top_level: list[str] = []
        if tree_resp.status_code == 409:
            _handle_response_errors(tree_resp, is_tree=True)
        elif tree_resp.status_code >= 400:
            logger.warning(
                "Tree request failed with HTTP %d for %s/%s. Using empty tree.",
                tree_resp.status_code,
                canonical_owner,
                canonical_name,
            )
        else:
            tree_data = tree_resp.json().get("tree", [])
            top_level = _process_tree(tree_data)

        logger.info("Top-level structure analyzed: %d entries retained", len(top_level))

        # 5. GET /repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1 (key files)
        key_files: list[KeyFile] = []
        rec_tree_url = (
            f"https://api.github.com/repos/{canonical_owner}/{canonical_name}/"
            f"git/trees/{default_branch}?recursive=1"
        )
        try:
            rec_tree_resp = _request_with_retry(sess, "GET", rec_tree_url)
            request_count += 1
            _check_rate_limit(rec_tree_resp, resolved_settings)
            if rec_tree_resp.status_code == 200:
                rec_tree_data = rec_tree_resp.json().get("tree", [])
                candidates: list[tuple[int, str, str]] = []
                top_lang = languages[0] if languages else ""
                for item in rec_tree_data:
                    if item.get("type") != "blob":
                        continue
                    f_path = str(item.get("path", ""))
                    f_size = item.get("size")
                    if isinstance(f_size, int) and f_size > 60000:
                        continue
                    f_ext = f_path.rsplit(".", 1)[-1].lower() if "." in f_path else ""
                    if f_ext not in ALLOWED_KEY_FILE_EXTENSIONS:
                        continue
                    segments = [seg.lower() for seg in f_path.split("/")]
                    if any(seg in SKIP_PATH_SEGMENTS for seg in segments):
                        continue
                    if ".min." in f_path.lower():
                        continue
                    score = score_key_file(f_path, canonical_name, top_lang)
                    if score >= 20:
                        candidates.append((score, f_path, f_ext))

                # Sort candidates by score descending, then path ascending
                candidates.sort(key=lambda x: (-x[0], x[1]))
                top_3 = candidates[:3]

                for idx, (_, k_path, k_ext) in enumerate(top_3, start=1):
                    content_url = (
                        f"https://api.github.com/repos/{canonical_owner}/{canonical_name}/"
                        f"contents/{urllib.parse.quote(k_path)}?ref={default_branch}"
                    )
                    c_resp = _request_with_retry(
                        sess,
                        "GET",
                        content_url,
                        headers={"Accept": "application/vnd.github.raw+json"},
                    )
                    request_count += 1
                    _check_rate_limit(c_resp, resolved_settings)
                    if c_resp.status_code == 200:
                        c_content, c_total_lines = prepare_file_content(c_resp.text)
                        lang_name = EXTENSION_TO_LANGUAGE.get(k_ext, k_ext)
                        key_files.append(
                            KeyFile(
                                id=f"f{idx}",
                                path=k_path,
                                language=lang_name,
                                content=c_content,
                                total_lines=c_total_lines,
                            )
                        )
                    else:
                        logger.warning(
                            "Failed to fetch content for key file %s (HTTP %d)",
                            k_path,
                            c_resp.status_code,
                        )
            else:
                logger.warning(
                    "Recursive tree call returned HTTP %d for %s/%s. Skipping key files.",
                    rec_tree_resp.status_code,
                    canonical_owner,
                    canonical_name,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to fetch recursive tree for key files in %s/%s: %s",
                canonical_owner,
                canonical_name,
                exc,
            )

        logger.info("Key files identified: %d files retained", len(key_files))

        # Validation: either README or description must exist
        if not cleaned_readme and not description:
            raise AnalyzerError(
                "This repository has no README and no description, so there is not enough to make a video.",
                hint="Add a README or description to the repository.",
            )

        logger.info("Repository analysis complete using %d API requests", request_count)

        return RepoData(
            name=canonical_name,
            owner=canonical_owner,
            description=description,
            stars=stars,
            languages=languages,
            readme=cleaned_readme,
            top_level=top_level,
            homepage=homepage,
            topics=topics,
            license=license_str,
            forks=forks,
            open_issues=open_issues,
            created_at=created_at,
            pushed_at=pushed_at,
            language_percentages=language_percentages,
            media=media,
            key_files=key_files,
            terminals=terminals,
        )

    finally:
        if should_close:
            sess.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python -m repo2video.analyzer <repository_url>\n")
        sys.exit(1)

    target_url = sys.argv[1]
    try:
        data = analyze(target_url)
        print(data.model_dump_json(indent=2))
    except Repo2VideoError as error:
        sys.stderr.write(f"{error}\n")
        sys.exit(1)
    except Exception as error:  # noqa: BLE001
        sys.stderr.write(f"Unexpected error: {error}\n")
        sys.exit(1)
