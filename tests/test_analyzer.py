"""Tests for GitHub repository analyzer, URL parser, and README cleaner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from repo2video.analyzer import (
    _calculate_language_percentages,
    analyze,
    clean_readme,
    extract_media_refs,
    extract_terminal_sessions,
    parse_repo_url,
    prepare_file_content,
    resolve_relative_url,
    score_key_file,
)
from repo2video.config import Settings
from repo2video.errors import AnalyzerError
from repo2video.models import RepoData


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
    ("https://github.com/owner", None),
    ("https://github.com/about/repo", None),
    ("https://github.com/settings/repo", None),
    ("https://github.com/invalid@user/repo", None),
    ("https://github.com/owner/invalid$repo", None),
]


@pytest.mark.parametrize(("url_input", "expected"), URL_TEST_CASES)
def test_parse_repo_url(url_input: str, expected: tuple[str, str] | None) -> None:
    """Verify URL parsing handles valid GitHub URLs and raises AnalyzerError on invalid."""
    if expected is not None:
        owner, repo = parse_repo_url(url_input)
        assert (owner, repo) == expected
    else:
        with pytest.raises(AnalyzerError):
            parse_repo_url(url_input)


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
    assert "Cool Project" in cleaned
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
    assert "#introduction" not in cleaned
    assert "#installation" not in cleaned
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
    assert "`[link](http://example.com)`" in cleaned
    assert 'url = "[link](http://example.com)"' in cleaned


def test_clean_readme_never_cuts_inside_fence() -> None:
    """Verify clean_readme never cuts inside a fenced code block."""
    filler = "Word " * 50  # ~250 chars
    code_block = (
        "\n\n```python\ndef run_pipeline():\n    for i in range(100):\n        process(i)\n```\n\n"
    )
    more_text = "Trailing content that exceeds the limit " * 30

    full_text = f"{filler}{code_block}{more_text}"
    limit = len(filler) + len(code_block) // 2

    cleaned = clean_readme(full_text, limit=limit)
    assert "def run_pipeline" not in cleaned
    assert cleaned.endswith(filler.rstrip())


def test_process_tree_noise_filtering_and_sort() -> None:
    """Verify tree processing filters hidden items, locks, caches, and sorts dirs first."""
    from repo2video.analyzer import _process_tree

    raw_items = [
        {"path": ".git", "type": "tree"},
        {"path": ".github", "type": "tree"},
        {"path": "node_modules", "type": "tree"},
        {"path": "__pycache__", "type": "tree"},
        {"path": "package-lock.json", "type": "blob"},
        {"path": "poetry.lock", "type": "blob"},
        {"path": ".gitignore", "type": "blob"},
        {"path": "deep/nested/file.py", "type": "blob"},
        {"path": "src", "type": "tree"},
        {"path": "docs", "type": "tree"},
        {"path": "app.py", "type": "blob"},
        {"path": "README.md", "type": "blob"},
    ]
    processed = _process_tree(raw_items)

    assert "docs" in processed
    assert "src" in processed
    assert "app.py" in processed
    assert "README.md" in processed

    # Excluded noise
    assert ".git" not in processed
    assert "node_modules" not in processed
    assert "package-lock.json" not in processed
    assert ".gitignore" not in processed
    assert "deep/nested/file.py" not in processed

    # Dirs first
    docs_idx = processed.index("docs")
    src_idx = processed.index("src")
    app_idx = processed.index("app.py")
    assert docs_idx < app_idx
    assert src_idx < app_idx


def test_process_languages_dropping_minor() -> None:
    """Verify languages below 2% threshold are dropped, but top 5 are kept."""
    from repo2video.analyzer import _process_languages

    data = {
        "Python": 9000,
        "Shell": 700,
        "HTML": 250,  # 2.5% (above 2%)
        "CSS": 50,  # 0.5% (below 2%)
    }
    langs = _process_languages(data)
    assert "Python" in langs
    assert "Shell" in langs
    assert "HTML" in langs
    assert "CSS" not in langs


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


def test_analyze_success() -> None:
    """Verify end-to-end analyze with fake HTTP session."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/readme": FakeResponse(
                status_code=200, text="# My Project\nDocumentation here."
            ),
            "/repos/owner/repo/languages": FakeResponse(
                status_code=200, json_data={"Python": 5000, "Rust": 1000}
            ),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=200,
                json_data={
                    "tree": [
                        {"path": "src", "type": "tree"},
                        {"path": "README.md", "type": "blob"},
                    ]
                },
            ),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "A demo repository.",
                    "stargazers_count": 42,
                    "default_branch": "main",
                },
            ),
        }
    )
    repo_data = analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]

    assert repo_data.name == "repo"
    assert repo_data.owner == "owner"
    assert repo_data.description == "A demo repository."
    assert repo_data.stars == 42
    assert repo_data.languages == ["Python", "Rust"]
    assert "My Project" in repo_data.readme
    assert repo_data.top_level == ["src", "README.md"]


def test_analyze_no_readme_succeeds_if_description_exists() -> None:
    """Verify analyze succeeds when repository lacks a README but has a description."""
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


# =========================================================================
# Prompt 9 (Version 2) Tests
# =========================================================================


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        ("image.png", "https://raw.githubusercontent.com/owner/repo/main/image.png"),
        ("./image.png", "https://raw.githubusercontent.com/owner/repo/main/image.png"),
        ("/docs/image.png", "https://raw.githubusercontent.com/owner/repo/main/docs/image.png"),
        (
            "./docs/sub/pic.png",
            "https://raw.githubusercontent.com/owner/repo/main/docs/sub/pic.png",
        ),
        (
            "assets/my image.png",
            "https://raw.githubusercontent.com/owner/repo/main/assets/my%20image.png",
        ),
        (
            "/images/screenshot 1.jpg",
            "https://raw.githubusercontent.com/owner/repo/main/images/screenshot%201.jpg",
        ),
        ("https://example.com/pic.png", "https://example.com/pic.png"),
        ("http://foo.bar/demo.gif", "http://foo.bar/demo.gif"),
        ("data:image/png;base64,123", "data:image/png;base64,123"),
    ],
)
def test_relative_url_resolution_table(raw_url: str, expected: str) -> None:
    """Verify relative URL resolution across a table of at least 8 inputs."""
    res = resolve_relative_url(raw_url, owner="owner", repo="repo", default_branch="main")
    assert res == expected


def test_media_extraction() -> None:
    """Verify media extraction from markdown, HTML, relative paths, badges, svg, duplicates and attachment links."""
    readme = """# Sample Project
    
[![Build Status](https://img.shields.io/badge.svg)](link)
[![Workflow](https://github.com/owner/repo/actions/workflows/ci.yml/badge.svg)](link)
<img src="https://img.shields.io/badge2.png" alt="shield" />
<img src="https://example.com/logo.svg" alt="svg logo" />
<img src="data:image/png;base64,..." alt="data uri" />
<img src="https://github.githubassets.com/images/icons/emoji/smile.png" alt="emoji" />

![Hero Banner](./docs/hero.png)
<img src="/screenshots/overview.jpg" alt="Overview Screenshot">
<video src="https://example.com/demo.mp4"></video>
<source src="assets/clip.webm">

https://github.com/user-attachments/assets/12345-abcde

[Watch Tutorial](https://example.com/walkthrough.mov)
![Duplicate Hero](./docs/hero.png)
![Animation](assets/animated.gif)
"""
    media = extract_media_refs(readme, owner="owner", repo="repo", default_branch="main")

    assert len(media) == 7
    assert media[0].id == "m1"
    assert media[0].url == "https://raw.githubusercontent.com/owner/repo/main/docs/hero.png"
    assert media[0].alt == "Hero Banner"
    assert media[0].kind == "image"

    assert media[1].id == "m2"
    assert (
        media[1].url == "https://raw.githubusercontent.com/owner/repo/main/screenshots/overview.jpg"
    )
    assert media[1].alt == "Overview Screenshot"
    assert media[1].kind == "image"

    assert media[2].id == "m3"
    assert media[2].url == "https://example.com/demo.mp4"
    assert media[2].kind == "video"

    assert media[3].id == "m4"
    assert media[3].url == "https://raw.githubusercontent.com/owner/repo/main/assets/clip.webm"
    assert media[3].kind == "video"

    assert media[4].id == "m5"
    assert media[4].url == "https://github.com/user-attachments/assets/12345-abcde"
    assert media[4].kind == "video"

    assert media[5].id == "m6"
    assert media[5].url == "https://example.com/walkthrough.mov"
    assert media[5].alt == "Watch Tutorial"
    assert media[5].kind == "video"

    assert media[6].id == "m7"
    assert media[6].url == "https://raw.githubusercontent.com/owner/repo/main/assets/animated.gif"
    assert media[6].kind == "gif"


def test_terminal_session_extraction() -> None:
    """Verify terminal session extraction with install/usage detection, prompt stripping, output pairing, and skipped oversized blocks."""
    readme = (
        """# Installation and Usage
```bash
$ pip install my-cool-tool
> npm install -g helper
# This is a comment
```
```text
Successfully installed my-cool-tool-1.0.0
```

```shell
$ my-cool-tool --help
$ my-cool-tool run
```
Here is some output from running:
```console
Starting tool...
Done in 0.5s
```

```
$ echo "No tag command"
```

```bash
line 1
line 2
line 3
line 4
line 5
line 6
line 7
line 8
line 9
```

```bash
$ this is a command with more than one hundred characters """
        + ("a" * 120)
        + """
```
"""
    )
    sessions = extract_terminal_sessions(readme)
    assert len(sessions) == 3
    # Session 1: install
    assert sessions[0].id == "t1"
    assert sessions[0].purpose == "install"
    assert sessions[0].commands == ["pip install my-cool-tool", "npm install -g helper"]
    assert sessions[0].output == ["Successfully installed my-cool-tool-1.0.0"]

    # Session 2: usage
    assert sessions[1].id == "t2"
    assert sessions[1].purpose == "usage"
    assert sessions[1].commands == ["my-cool-tool --help", "my-cool-tool run"]
    assert sessions[1].output == ["Starting tool...", "Done in 0.5s"]

    # Session 3: usage (no tag, prompt lines)
    assert sessions[2].id == "t3"
    assert sessions[2].purpose == "usage"
    assert sessions[2].commands == ['echo "No tag command"']
    assert sessions[2].output == []


def test_key_file_scoring() -> None:
    """Verify key file scoring prioritizes entry points, preferred folders, and penalizes tests/deep folders."""
    # Stem main in src: 40 (main) + 15 (src) + 10 (python match) = 65
    assert score_key_file("src/main.py", "myrepo", top_language="Python") == 65
    # app.py at root: 40 (app) + 10 (python match) = 50
    assert score_key_file("app.py", "myrepo", top_language="Python") == 50
    # Deep folder penalty: -5 for each level beyond first
    # src/app/main.py: 40 (main) + 15 (app) + 10 (python) - 5 (1 level beyond first) = 60
    assert score_key_file("src/app/main.py", "myrepo", top_language="Python") == 60
    # Penalty keyword: test in stem -> -30
    assert score_key_file("src/test_main.py", "myrepo", top_language="Python") < 20


def test_content_preparation() -> None:
    """Verify content preparation drops license comment headers, converts tabs, caps lines at 40, and truncates characters."""
    raw = (
        "/*\n * MIT License\n * Copyright (c) 2026\n * All rights reserved.\n */\n\n"
        + "\tdef hello():\n"
        + "".join(f"\t    print('line {i}')\n" for i in range(1, 60))
    )
    content, total_lines = prepare_file_content(raw)
    assert total_lines > 50
    assert "MIT License" not in content
    assert "\t" not in content
    assert "    def hello():" in content
    lines = content.splitlines()
    assert len(lines) <= 40
    assert len(content) <= 2500


def test_failing_tree_call_returns_no_key_files_but_analysis_succeeds() -> None:
    """Verify that a 500 error on the recursive tree call does not fail analysis and returns empty key_files."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/git/trees/main?recursive=1": FakeResponse(status_code=500),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=200, json_data={"tree": []}
            ),
            "/repos/owner/repo/readme": FakeResponse(status_code=200, text="# My Readme"),
            "/repos/owner/repo/languages": FakeResponse(
                status_code=200, json_data={"Python": 1000}
            ),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "Repo description",
                    "stargazers_count": 10,
                    "default_branch": "main",
                },
            ),
        }
    )
    data = analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]
    assert data.key_files == []
    assert data.name == "repo"


def test_old_repo_json_still_loads(tmp_path: Path) -> None:
    """Verify backward compatibility: old v1 repo.json files without new fields still load cleanly."""
    old_data = {
        "name": "oldrepo",
        "owner": "oldowner",
        "description": "An old repo description",
        "stars": 42,
        "languages": ["Python", "Rust"],
        "readme": "Old cleaned readme content.",
        "top_level": ["src", "README.md"],
    }
    file_path = tmp_path / "repo.json"
    file_path.write_text(json.dumps(old_data), encoding="utf-8")

    loaded = RepoData.load(file_path)
    assert loaded.name == "oldrepo"
    assert loaded.homepage == ""
    assert loaded.topics == []
    assert loaded.license == ""
    assert loaded.forks == 0
    assert loaded.open_issues == 0
    assert loaded.language_percentages == {}
    assert loaded.media == []
    assert loaded.key_files == []
    assert loaded.terminals == []


def test_media_extraction_runs_on_raw_readme() -> None:
    """Verify media extraction runs on raw README so images are captured before clean_readme strips them."""
    raw_readme = """# Project With Media
![Banner](docs/banner.png)
<img src="docs/screen.png" alt="Screenshot" />

Actual documentation text.
"""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/readme": FakeResponse(status_code=200, text=raw_readme),
            "/repos/owner/repo/languages": FakeResponse(status_code=200, json_data={"Python": 100}),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=200, json_data={"tree": []}
            ),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "Test repo",
                    "stargazers_count": 5,
                    "default_branch": "main",
                },
            ),
        }
    )
    repo_data = analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]
    # Media was extracted from raw README
    assert len(repo_data.media) == 2
    assert (
        repo_data.media[0].url
        == "https://raw.githubusercontent.com/owner/repo/main/docs/banner.png"
    )
    assert (
        repo_data.media[1].url
        == "https://raw.githubusercontent.com/owner/repo/main/docs/screen.png"
    )
    # But clean_readme removed the images from repo_data.readme
    assert "![" not in repo_data.readme
    assert "<img" not in repo_data.readme
    assert "Actual documentation text." in repo_data.readme


def test_rate_limit_warning_when_low(caplog: pytest.LogCaptureFixture) -> None:
    """Verify rate limit warning is logged when remaining is below 10 and no token is set."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/readme": FakeResponse(status_code=200, text="# Readme"),
            "/repos/owner/repo/languages": FakeResponse(status_code=200, json_data={"Python": 100}),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=200, json_data={"tree": []}
            ),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                headers={"X-RateLimit-Remaining": "5"},
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "Rate limit test",
                    "stargazers_count": 1,
                    "default_branch": "main",
                },
            ),
        }
    )
    with caplog.at_level(logging.WARNING):
        analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]
    assert any("rate limit remaining is low" in record.message for record in caplog.records)


def test_language_percentages_calculation() -> None:
    """Verify language percentages calculates top 5 correctly."""
    raw = {
        "Python": 600,
        "JavaScript": 300,
        "HTML": 100,
    }
    percentages = _calculate_language_percentages(raw)
    assert percentages["Python"] == 60.0
    assert percentages["JavaScript"] == 30.0
    assert percentages["HTML"] == 10.0


def test_full_v2_metadata_populated() -> None:
    """Verify that all v2 metadata fields (homepage, topics, license, forks, etc.) are correctly populated."""
    fake_session = FakeSession(
        routes={
            "/repos/owner/repo/readme": FakeResponse(status_code=200, text="# Readme"),
            "/repos/owner/repo/languages": FakeResponse(
                status_code=200, json_data={"Python": 800, "Rust": 200}
            ),
            "/repos/owner/repo/git/trees/main": FakeResponse(
                status_code=200, json_data={"tree": []}
            ),
            "/repos/owner/repo": FakeResponse(
                status_code=200,
                json_data={
                    "name": "repo",
                    "owner": {"login": "owner"},
                    "description": "Full v2 metadata test",
                    "stargazers_count": 100,
                    "default_branch": "main",
                    "homepage": "https://example.com/project",
                    "topics": ["ai", "video", "python"],
                    "license": {"spdx_id": "MIT"},
                    "forks_count": 15,
                    "open_issues_count": 3,
                    "created_at": "2026-01-01T00:00:00Z",
                    "pushed_at": "2026-10-24T12:00:00Z",
                },
            ),
        }
    )
    data = analyze("https://github.com/owner/repo", session=fake_session)  # type: ignore[arg-type]
    assert data.homepage == "https://example.com/project"
    assert data.topics == ["ai", "video", "python"]
    assert data.license == "MIT"
    assert data.forks == 15
    assert data.open_issues == 3
    assert data.created_at == "2026-01-01T00:00:00Z"
    assert data.pushed_at == "2026-10-24T12:00:00Z"
    assert data.language_percentages == {"Python": 80.0, "Rust": 20.0}
