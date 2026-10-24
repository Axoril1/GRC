"""Repository analyzer for fetching and processing GitHub repository metadata."""

from __future__ import annotations

import logging
import re
import sys
import time
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any

import requests

from repo2video.config import get_settings
from repo2video.errors import AnalyzerError, Repo2VideoError
from repo2video.models import RepoData

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

    return filtered[:5]


def _process_tree(tree_items: list[dict[str, Any]]) -> list[str]:
    """Process top-level git tree entries.

    Separates directories and files, skips noise items, sorts directories
    alphabetically ignoring case, sorts files alphabetically ignoring case,
    and returns directories first followed by files, up to 30 entries.

    Args:
        tree_items: List of git tree entries from GitHub REST API.

    Returns:
        List of at most 30 top-level entry names.
    """
    dirs: list[str] = []
    files: list[str] = []

    for item in tree_items:
        path = item.get("path", "")
        item_type = item.get("type", "")

        if not path or path.lower() in NOISE_ITEMS:
            continue

        if item_type == "tree":
            dirs.append(f"{path}/")
        else:
            files.append(path)

    dirs.sort(key=lambda s: s.lower())
    files.sort(key=lambda s: s.lower())

    return (dirs + files)[:30]


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> requests.Response:
    """Execute an HTTP request with retry logic for transient errors.

    Retries only connection errors, timeouts, and HTTP 500, 502, 503, 504 status codes
    up to 2 times with delays of 1s and 2s. Never retries 4xx responses.

    Args:
        session: The requests.Session to use.
        method: HTTP method (e.g. 'GET').
        url: Request URL.
        headers: Optional request headers.
        timeout: Request timeout in seconds (default 30).

    Returns:
        The requests.Response object.

    Raises:
        AnalyzerError: On connection failure, rate limiting, or HTTP errors.
    """
    max_retries = 2
    delays = [1, 2]

    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(delays[attempt - 1])

        try:
            resp = session.request(method, url, headers=headers, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt < max_retries:
                logger.warning(
                    "Network error on %s %s (attempt %d/%d): %s. Retrying...",
                    method,
                    url,
                    attempt + 1,
                    max_retries + 1,
                    exc.__class__.__name__,
                )
                continue
            raise AnalyzerError(
                "Could not reach GitHub.",
                hint="Check your internet connection.",
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise AnalyzerError(
                "Could not reach GitHub.",
                hint="Check your internet connection.",
            ) from exc

        if resp.status_code in (500, 502, 503, 504) and attempt < max_retries:
            logger.warning(
                "HTTP %d on %s %s (attempt %d/%d). Retrying...",
                resp.status_code,
                method,
                url,
                attempt + 1,
                max_retries + 1,
            )
            continue

        return resp

    raise AnalyzerError(
        "Could not reach GitHub.",
        hint="Check your internet connection.",
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

    Fetches repository metadata, README, language statistics, and top-level
    directory structure using the GitHub REST API.

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
        # a) GET /repos/{owner}/{repo}
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        repo_resp = _request_with_retry(sess, "GET", repo_url)
        request_count += 1
        _handle_response_errors(repo_resp)

        repo_info = repo_resp.json()
        canonical_name: str = repo_info["name"]
        canonical_owner: str = repo_info["owner"]["login"]
        description: str = repo_info.get("description") or ""
        stars: int = repo_info.get("stargazers_count", 0)
        default_branch: str = repo_info.get("default_branch", "main")

        logger.info(
            "Repository metadata retrieved: %s/%s (stars: %d, default branch: %s)",
            canonical_owner,
            canonical_name,
            stars,
            default_branch,
        )

        # b) GET /repos/{owner}/{repo}/readme
        readme_url = f"https://api.github.com/repos/{canonical_owner}/{canonical_name}/readme"
        readme_resp = _request_with_retry(
            sess,
            "GET",
            readme_url,
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        request_count += 1

        raw_readme = ""
        if readme_resp.status_code == 404:
            logger.warning("No README found for %s/%s", canonical_owner, canonical_name)
        else:
            _handle_response_errors(readme_resp)
            raw_readme = readme_resp.text

        cleaned_readme = clean_readme(raw_readme, limit=6000)
        logger.info(
            "README processed: %d characters before cleaning, %d characters after",
            len(raw_readme),
            len(cleaned_readme),
        )

        # c) GET /repos/{owner}/{repo}/languages
        langs_url = f"https://api.github.com/repos/{canonical_owner}/{canonical_name}/languages"
        langs_resp = _request_with_retry(sess, "GET", langs_url)
        request_count += 1
        _handle_response_errors(langs_resp)

        languages = _process_languages(langs_resp.json())
        logger.info("Languages analyzed: %d languages selected", len(languages))

        # d) GET /repos/{owner}/{repo}/git/trees/{default_branch}
        tree_url = (
            f"https://api.github.com/repos/{canonical_owner}/{canonical_name}/"
            f"git/trees/{default_branch}"
        )
        tree_resp = _request_with_retry(sess, "GET", tree_url)
        request_count += 1

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
