"""Tests for GitHub repository analyzer, URL parser, and README cleaner."""

from __future__ import annotations

import json
from typing import Any

import pytest

from repo2video.analyzer import (
    analyze,
    clean_readme,
    parse_repo_url,
)
from repo2video.config import Settings
from repo2video.errors import AnalyzerError


class FakeResponse:
    """Fake requests.Response for testing without network calls."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text if text else (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {}

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        if self.text:
            return json.loads(self.text)
        raise ValueError("No JSON data")


class FakeSession:
    """Fake requests.Session to inject prepared responses."""

    def __init__(self, routes: dict[str, list[FakeResponse] | FakeResponse] | None = None) -> None:
        self.routes: dict[str, list[FakeResponse]] = {}
        if routes:
            for pattern, resp in routes.items():
                if isinstance(resp, list):
                    self.routes[pattern] = list(resp)
                else:
                    self.routes[pattern] = [resp]
        self.headers: dict[str, str] = {}
        self.call_history: list[tuple[str, str, dict[str, str] | None]] = []

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> FakeResponse:
        self.call_history.append((method, url, headers))

        # Check for matching route (longest pattern first)
        for pattern in sorted(self.routes.keys(), key=len, reverse=True):
            if pattern in url:
                resps = self.routes[pattern]
                if len(resps) > 1:
                    return resps.pop(0)
                return resps[0]

        # Default 404
        return FakeResponse(status_code=404, json_data={"message": "Not Found"})

    def close(self) -> None:
        pass


URL_TEST_CASES = [
    # Valid URLs
    ("https://github.com/owner/repo", ("owner", "repo")),
    ("http://github.com/owner/repo", ("owner", "repo")),
    ("https://www.github.com/owner/repo", ("owner", "repo")),
    ("http://www.github.com/owner/repo", ("owner", "repo")),
    ("github.com/owner/repo", ("owner", "repo")),
    ("www.github.com/owner/repo", ("owner", "repo")),
    ("https://github.com/owner/repo/", ("owner", "repo")),
    ("https://github.com/owner/repo.git", ("owner", "repo")),
    ("github.com/owner/repo.git", ("owner", "repo")),
    ("https://github.com/owner/repo/tree/main/src", ("owner", "repo")),
    ("https://github.com/owner/repo/blob/main/README.md", ("owner", "repo")),
    ("https://github.com/owner/repo/issues/42", ("owner", "repo")),
    ("https://github.com/owner/repo?tab=readme-ov-file#installation", ("owner", "repo")),
    ("   https://github.com/owner/repo   ", ("owner", "repo")),
    ("https://github.com/owner-name_1.0/repo.name-2_0", ("owner-name_1.0", "repo.name-2_0")),
    # Invalid URLs
    ("", None),
    ("   ", None),
    ("https://gitlab.com/owner/repo", None),
    ("https://bitbucket.org/owner/repo", None),
    ("https://github.com/owner", None),
    ("https://github.com/owner/", None),
    ("https://github.com/settings/profile", None),
    ("https://github.com/orgs/my-org", None),
    ("https://github.com/marketplace", None),
    ("https://github.com/about/team", None),
    ("https://github.com/owner/repo$bad", None),
    ("https://github.com/bad@owner/repo", None),
]


@pytest.mark.parametrize("input_url, expected", URL_TEST_CASES)
def test_parse_repo_url(input_url: str, expected: tuple[str, str] | None) -> None:
    """Verify parse_repo_url handles valid and invalid URLs across a wide table."""
    if expected is not None:
        assert parse_repo_url(input_url) == expected
    else:
        with pytest.raises(AnalyzerError) as exc_info:
            parse_repo_url(input_url)
        assert exc_info.value.hint is not None
        assert "https://github.com/owner/repo" in exc_info.value.hint


def test_clean_readme_removes_badges_and_comments() -> None:
    """Verify clean_readme removes badges, comments, and empty lines."""
    raw = """<!-- This is an HTML comment -->
# Cool Project

[![Build Status](https://img.shields.io/travis/user/repo.svg)](https://travis-ci.org/user/repo)
![Coverage](https://coveralls.io/repos/user/repo/badge.svg)
![Standalone Badge](https://example.com/badge.png)

This is actual repository documentation.
"""
    cleaned = clean_readme(raw)
    assert "HTML comment" not in cleaned
    assert "shields.io" not in cleaned
    assert "coveralls.io" not in cleaned
    assert "Standalone Badge" not in cleaned
    assert "# Cool Project" in cleaned
    assert "This is actual repository documentation." in cleaned


def test_clean_readme_removes_table_of_contents() -> None:
    """Verify clean_readme removes table of contents and its anchor link list."""
    raw = """# Guide

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)

## Introduction
Welcome to the guide.
"""
    cleaned = clean_readme(raw)
    assert "Table of Contents" not in cleaned
    assert "[Introduction](#introduction)" not in cleaned
    assert "[Prerequisites](#prerequisites)" not in cleaned
    assert "## Introduction" in cleaned
    assert "Welcome to the guide." in cleaned


def test_clean_readme_removes_base64_and_html_images() -> None:
    """Verify clean_readme strips base64 data URIs, picture blocks, and img tags."""
    raw = """# Title
<picture>
  <source srcset="dark.png" media="(prefers-color-scheme: dark)">
  <img src="light.png" alt="Theme image">
</picture>

<img src="https://example.com/logo.png" width="200" />

![Inline Base64](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA==)

Real content stays here.
"""
    cleaned = clean_readme(raw)
    assert "<picture>" not in cleaned
    assert "<img" not in cleaned
    assert "data:image/png;base64" not in cleaned
    assert "Real content stays here." in cleaned


def test_clean_readme_markdown_links_and_code_preservation() -> None:
    """Verify markdown links are replaced with text while code remains untouched."""
    raw = """Check out [Requests](https://requests.readthedocs.io) for HTTP.

Here is inline code: `[link](http://example.com)`.

```python
# Fenced code block with link
url = "[link](http://example.com)"
```
"""
    cleaned = clean_readme(raw)
    assert "Check out Requests for HTTP." in cleaned
    assert "https://requests.readthedocs.io" not in cleaned
    # Inside inline code, preserved
    assert "`[link](http://example.com)`" in cleaned
    # Inside fenced block, preserved
    assert 'url = "[link](http://example.com)"' in cleaned


def test_clean_readme_never_cuts_inside_fence() -> None:
    """Verify clean_readme never cuts inside a fenced code block."""
    filler = "Word " * 50  # ~250 chars
    code_block = (
        "\n\n```python\ndef run_pipeline():\n    for i in range(100):\n        process(i)\n```\n\n"
    )
    more_text = "Trailing content that exceeds the limit " * 30

    full_text = f"{filler}{code_block}{more_text}"
    # Choose a limit that lands right inside the code block
    fence_start = full_text.index("```python")
    cut_inside_target = fence_start + 15

    cleaned = clean_readme(full_text, limit=cut_inside_target)
    # Must cut BEFORE the opening fence!
    assert "```python" not in cleaned
    assert len(cleaned) <= fence_start


def test_analyze_rate_limit_error() -> None:
    """Verify rate limit error includes reset time and instructions in the hint."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo": FakeResponse(
                status_code=403,
                json_data={"message": "API rate limit exceeded"},
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "1770000000",
                },
            )
        }
    )
    with pytest.raises(AnalyzerError) as exc_info:
        analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]

    msg = str(exc_info.value)
    assert "rate limit reached" in msg
    assert "resets at" in msg
    assert "Set GITHUB_TOKEN" in exc_info.value.hint
    assert "5000 requests per hour" in exc_info.value.hint


def test_analyze_404_mapping() -> None:
    """Verify 404 response produces clear not found message and spelling hint."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo": FakeResponse(
                status_code=404,
                json_data={"message": "Not Found"},
            )
        }
    )
    with pytest.raises(AnalyzerError) as exc_info:
        analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]

    assert "Repository not found, or it is private." in exc_info.value.message
    assert "Check the spelling; private repositories are not supported." in exc_info.value.hint


def test_analyze_missing_readme_gives_empty_readme() -> None:
    """Verify a 404 on the readme call results in an empty readme string."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/readme": FakeResponse(status_code=404),
            "/repos/owner/repo/languages": FakeResponse(
                status_code=200, json_data={"Python": 1000}
            ),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=200, json_data={"tree": [{"path": "main.py", "type": "blob"}]}
            ),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "Has a description but no readme.",
                    "stargazers_count": 10,
                    "default_branch": "main",
                },
            ),
        }
    )
    repo_data = analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]
    assert repo_data.readme == ""
    assert repo_data.name == "repo"
    assert repo_data.description == "Has a description but no readme."


def test_analyze_tree_failure_still_succeeds() -> None:
    """Verify that a tree endpoint failure does not cause analysis to fail."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/git/trees/main": FakeResponse(status_code=500),
            "/repos/owner/repo/readme": FakeResponse(
                status_code=200, text="# My Readme\nDocumentation here."
            ),
            "/repos/owner/repo/languages": FakeResponse(status_code=200, json_data={"Python": 500}),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "Repo description",
                    "stargazers_count": 25,
                    "default_branch": "main",
                },
            ),
        }
    )
    repo_data = analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]
    assert repo_data.top_level == []
    assert repo_data.name == "repo"


def test_analyze_retry_503_then_200() -> None:
    """Verify that transient 503 is retried and succeeds on subsequent attempt."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/readme": FakeResponse(status_code=200, text="# Readme"),
            "/repos/owner/repo/languages": FakeResponse(status_code=200, json_data={"Python": 100}),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=200, json_data={"tree": []}
            ),
            "/repos/owner/repo": [
                FakeResponse(status_code=503),
                FakeResponse(
                    status_code=200,
                    json_data={
                        "name": "repo",
                        "owner": {"login": "owner"},
                        "description": "Recovered from 503",
                        "stargazers_count": 5,
                        "default_branch": "main",
                    },
                ),
            ],
        }
    )
    repo_data = analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]
    assert repo_data.name == "repo"
    assert repo_data.description == "Recovered from 503"


def test_token_never_appears_in_error_text() -> None:
    """Verify personal access token is never leaked in errors, hints, or logs."""
    secret_token = "ghp_super_confidential_token_999888"
    settings = Settings(github_token=secret_token)

    fake_session = FakeSession(
        routes={
            "/repos/owner/repo": FakeResponse(
                status_code=401,
                json_data={"message": "Bad credentials"},
            )
        }
    )

    with pytest.raises(AnalyzerError) as exc_info:
        analyze("https://github.com/owner/repo", settings=settings, session=fake_session)  # type: ignore[arg-type]

    full_error_text = f"{exc_info.value.message} {exc_info.value.hint} {exc_info.value!s}"
    assert secret_token not in full_error_text


def test_empty_repository_409() -> None:
    """Verify 409 on tree call raises AnalyzerError indicating repository is empty."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/readme": FakeResponse(status_code=404),
            "/repos/owner/repo/languages": FakeResponse(status_code=200, json_data={}),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=409, json_data={"message": "Git Repository is empty."}
            ),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "Just initialized repo",
                    "stargazers_count": 0,
                    "default_branch": "main",
                },
            ),
        }
    )
    with pytest.raises(AnalyzerError) as exc_info:
        analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]

    assert "The repository is empty." in exc_info.value.message


def test_no_readme_and_no_description_raises_analyzer_error() -> None:
    """Verify repository with neither readme nor description raises AnalyzerError."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/readme": FakeResponse(status_code=404),
            "/repos/owner/repo/languages": FakeResponse(status_code=200, json_data={}),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=200, json_data={"tree": []}
            ),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "",
                    "stargazers_count": 0,
                    "default_branch": "main",
                },
            ),
        }
    )
    with pytest.raises(AnalyzerError) as exc_info:
        analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]

    assert "This repository has no README and no description" in exc_info.value.message
