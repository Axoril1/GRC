"""Tests for HTML slide generation, auto-fitting, and Playwright scene rendering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from repo2video.config import Settings
from repo2video.errors import SceneError
from repo2video.models import Scene, ScriptData
from repo2video.scenes import (
    DEFAULT_ACCENT,
    build_html,
    main,
    render_scenes,
)


def is_chromium_available() -> bool:
    """Check if the Playwright Chromium browser is installed and launchable."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except (PlaywrightError, OSError, RuntimeError):
        return False


skip_if_no_chromium = pytest.mark.skipif(
    not is_chromium_available(),
    reason="Playwright Chromium browser executable is not installed",
)


def make_test_scene(
    scene_id: int = 1,
    scene_type: str = "feature",
    heading: str = "Key Architecture Features",
    bullets: list[str] | None = None,
    narration: str = (
        "This is a standard narration sentence designed for testing automated slide "
        "creation, layout fitting, and rendering pipelines."
    ),
    code: str = "",
) -> Scene:
    """Create a valid Scene model for testing."""
    return Scene(
        id=scene_id,
        type=scene_type,
        heading=heading,
        bullets=bullets if bullets is not None else ["Feature bullet one", "Feature bullet two"],
        narration=narration,
        code=code,
    )


# ==============================================================================
# 1. Fast tests with no browser
# ==============================================================================


class TestBuildHtmlFast:
    """Fast unit tests for HTML generation and escaping without launching a browser."""

    def test_every_scene_type_produces_html_containing_heading(self) -> None:
        scene_types = ("title", "problem", "feature", "howto", "outro")
        for stype in scene_types:
            heading_text = f"Heading For {stype.capitalize()}"
            scene = make_test_scene(scene_id=1, scene_type=stype, heading=heading_text)
            html_doc = build_html(scene, 1, 5, repo_label="owner/repo")

            assert heading_text in html_doc
            assert "<!DOCTYPE html>" in html_doc
            assert "window.__fitted = true" in html_doc
            assert "Inter" in html_doc
            assert "JetBrains Mono" in html_doc

    def test_escaping_of_script_tags_quotes_and_ampersands(self) -> None:
        malicious_heading = "<script>alert(\"heading\")</script> & 'single'"
        malicious_bullets = [
            '<img src="x" onerror="alert(1)">',
            'Tom & Jerry "Special" edition',
        ]
        malicious_code = 'if (a < b && c > d) {\n  alert("code");\n}'

        scene = make_test_scene(
            heading=malicious_heading,
            bullets=malicious_bullets,
            code=malicious_code,
        )
        html_doc = build_html(scene, 1, 1, repo_label="owner/<safe>&'repo")

        # Plain unescaped script and img tags must NEVER appear in the document
        assert '<script>alert("heading")</script>' not in html_doc
        assert '<img src="x"' not in html_doc

        # Escaped entities must be present
        assert "&lt;script&gt;alert(&quot;heading&quot;)&lt;/script&gt;" in html_doc
        assert "&lt;img src=&quot;x&quot; onerror=&quot;alert(1)&quot;&gt;" in html_doc
        assert "Tom &amp; Jerry &quot;Special&quot; edition" in html_doc
        assert "if (a &lt; b &amp;&amp; c &gt; d)" in html_doc
        assert "owner/&lt;safe&gt;&amp;&#x27;repo" in html_doc

    def test_dark_and_light_themes_change_css_variables(self) -> None:
        scene = make_test_scene()

        dark_html = build_html(scene, 1, 5, repo_label="repo", theme="dark")
        assert "--bg: #0b1020;" in dark_html
        assert "--surface: #151b30;" in dark_html
        assert "--text: #f4f6fb;" in dark_html
        assert "--muted: #9aa3bd;" in dark_html

        light_html = build_html(scene, 1, 5, repo_label="repo", theme="light")
        assert "--bg: #f7f8fc;" in light_html
        assert "--surface: #ffffff;" in light_html
        assert "--text: #121528;" in light_html
        assert "--muted: #5b6382;" in light_html

    def test_invalid_accent_falls_back_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        scene = make_test_scene()
        with caplog.at_level(logging.WARNING):
            html_doc = build_html(scene, 1, 5, repo_label="repo", accent="invalid-color")

        assert f"--accent: {DEFAULT_ACCENT};" in html_doc
        assert "Invalid accent color 'invalid-color'; falling back to" in caplog.text

    def test_valid_short_and_long_hex_accents(self) -> None:
        scene = make_test_scene()
        html_3 = build_html(scene, 1, 5, repo_label="repo", accent="#f00")
        assert "--accent: #f00;" in html_3

        html_6 = build_html(scene, 1, 5, repo_label="repo", accent="#10b981")
        assert "--accent: #10b981;" in html_6

    def test_footer_shows_correct_index_and_total(self) -> None:
        scene = make_test_scene()
        html_doc = build_html(scene, 3, 7, repo_label="facebook/react")

        assert "3 / 7" in html_doc
        assert "facebook/react" in html_doc

    def test_very_long_heading_and_multiline_code_render_without_error(self) -> None:
        # 60-word heading (bypassing model validator by instantiating via construct / raw)
        long_heading = " ".join(["VeryLongHeadingWord"] * 60)
        # 20-line code string
        long_code = "\n".join(f"line_{i} = process_data_entry({i})" for i in range(20))

        # We construct Scene without triggering pydantic validation limits for this edge-case HTML test
        scene = Scene.model_construct(
            id=1,
            type="feature",
            heading=long_heading,
            bullets=["Bullet 1"],
            narration="Narration text for testing.",
            code=long_code,
        )

        html_doc = build_html(scene, 1, 1, repo_label="my/repo")
        assert "VeryLongHeadingWord" in html_doc
        assert "line_19 = process_data_entry(19)" in html_doc

    def test_title_scene_layout_elements(self) -> None:
        scene = make_test_scene(
            scene_type="title",
            heading="Title Scene Heading",
            bullets=["Subtitle descriptive text under title"],
        )
        html_doc = build_html(scene, 1, 5, repo_label="repo")
        assert "title-heading" in html_doc
        assert "title-subtitle" in html_doc
        assert "Subtitle descriptive text under title" in html_doc

    def test_problem_scene_layout_cards(self) -> None:
        scene = make_test_scene(
            scene_type="problem",
            bullets=["Problem item one", "Problem item two"],
        )
        html_doc = build_html(scene, 2, 5, repo_label="repo")
        assert "problem-card" in html_doc
        assert "Problem item one" in html_doc

    def test_feature_scene_two_columns_when_code_present(self) -> None:
        scene = make_test_scene(
            scene_type="feature",
            bullets=["Bullet item"],
            code="x = 1\ny = 2",
        )
        html_doc = build_html(scene, 3, 5, repo_label="repo")
        assert "two-column-container" in html_doc
        assert "code-column" in html_doc
        assert "feature-badge" in html_doc

    def test_howto_terminal_box_elements(self) -> None:
        scene = make_test_scene(
            scene_type="howto",
            bullets=["Step 1"],
            code="pip install repo2video",
        )
        html_doc = build_html(scene, 4, 5, repo_label="repo")
        assert "terminal-window" in html_doc
        assert "term-dot red" in html_doc
        assert "pip install repo2video" in html_doc

    def test_outro_pill_element(self) -> None:
        scene = make_test_scene(
            scene_type="outro",
            heading="Thank You",
            bullets=["Check it out"],
        )
        html_doc = build_html(scene, 5, 5, repo_label="myorg/myproject")
        assert "outro-pill" in html_doc
        assert "Star it on GitHub" in html_doc
        assert "myorg/myproject" in html_doc

    def test_new_scene_types_render_with_feature_layout(self) -> None:
        """Verify new v2 scene types render with feature layout (badge and bullets) as fallbacks."""
        from repo2video.models import ArchitectureSpec, NodeSpec

        narr = (
            "This is a standard narration sentence designed for testing automated slide "
            "creation, layout fitting, and rendering pipelines."
        )
        scenes = [
            Scene(
                id=1,
                type="code",
                heading="Heading for code",
                bullets=["Code fallback"],
                narration=narr,
                code="print('hi')",
                file_id="f1",
            ),
            Scene(
                id=2,
                type="terminal",
                heading="Heading for terminal",
                bullets=["Terminal fallback"],
                narration=narr,
                code="ls -la",
                terminal_id="t1",
            ),
            Scene(
                id=3,
                type="architecture",
                heading="Heading for architecture",
                bullets=["Arch fallback"],
                narration=narr,
                architecture=ArchitectureSpec(
                    nodes=[
                        NodeSpec(id="n1", label="Input", group="input"),
                        NodeSpec(id="n2", label="Core", group="core"),
                        NodeSpec(id="n3", label="Output", group="output"),
                    ]
                ),
            ),
            Scene(
                id=4,
                type="stats",
                heading="Heading for stats",
                bullets=["Stats fallback"],
                narration=narr,
                stat_keys=["stars", "forks"],
            ),
            Scene(
                id=5,
                type="media",
                heading="Heading for media",
                bullets=["Media fallback"],
                narration=narr,
                media_id="m1",
            ),
        ]
        for scene in scenes:
            html_doc = build_html(scene, 2, 5, repo_label="owner/repo")
            assert scene.heading in html_doc
            assert "feature-bullets" in html_doc
            assert "feature-badge" in html_doc
            assert scene.bullets[0] in html_doc


# ==============================================================================
# 2. Slow test with Playwright browser
# ==============================================================================


@pytest.mark.browser
@skip_if_no_chromium
class TestPlaywrightBrowserRendering:
    """Slow integration tests that launch Chromium to render real PNG slides."""

    def test_render_3_scenes_caching_and_theme_change(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        scenes = [
            make_test_scene(1, "title", "Project Title", ["The primary subtitle sentence."]),
            make_test_scene(2, "problem", "The Scaling Pain", ["Bottleneck A", "Bottleneck B"]),
            make_test_scene(
                3, "feature", "Key Capabilities", ["Feature bullet"], code="pip install tool"
            ),
            make_test_scene(
                4, "howto", "Setup Instructions", ["Run install command"], code="python main.py"
            ),
            make_test_scene(5, "outro", "Get Started Now", ["Try today"]),
        ]
        script = ScriptData(title="Project Title", scenes=scenes)

        out_dir = tmp_path / "slides"
        settings = Settings(video_width=1920, video_height=1080)

        # 1. First render run: creates 5 PNGs and 5 sidecars
        paths = render_scenes(
            script=script,
            out_dir=out_dir,
            repo_label="owner/project",
            theme="dark",
            settings=settings,
        )

        assert len(paths) == 5
        for p in paths:
            assert p.is_file()
            sidecar = out_dir / f"{p.name}.sha256"
            assert sidecar.is_file()

            # Read PNG header bytes 16 to 24 (no Pillow dependency)
            png_bytes = p.read_bytes()
            assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
            w = int.from_bytes(png_bytes[16:20], "big")
            h = int.from_bytes(png_bytes[20:24], "big")
            assert w == 1920
            assert h == 1080

        # Record mtimes to check cache
        mtimes_first = [p.stat().st_mtime_ns for p in paths]

        # 2. Rerunning with identical arguments must hit cache
        caplog.clear()
        with caplog.at_level(logging.INFO):
            paths_rerun = render_scenes(
                script=script,
                out_dir=out_dir,
                repo_label="owner/project",
                theme="dark",
                settings=settings,
            )

        assert len(paths_rerun) == 5
        mtimes_second = [p.stat().st_mtime_ns for p in paths_rerun]
        assert mtimes_first == mtimes_second
        assert "Scene 01 cached, skipping render" in caplog.text

        # 3. Changing theme re-renders the slides
        paths_light = render_scenes(
            script=script,
            out_dir=out_dir,
            repo_label="owner/project",
            theme="light",
            settings=settings,
        )
        assert len(paths_light) == 5
        mtimes_third = [p.stat().st_mtime_ns for p in paths_light]
        # At least one file re-rendered and updated
        assert mtimes_third != mtimes_first

    def test_missing_chromium_raises_scene_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from playwright.sync_api import BrowserType
        from playwright.sync_api import Error as PlaywrightError

        def fake_launch(*args: Any, **kwargs: Any) -> Any:
            raise PlaywrightError("Executable doesn't exist at /root/.cache/ms-playwright/chromium")

        monkeypatch.setattr(BrowserType, "launch", fake_launch)

        script = ScriptData(
            title="Title",
            scenes=[
                make_test_scene(1, "title"),
                make_test_scene(2, "problem"),
                make_test_scene(3, "feature"),
                make_test_scene(4, "howto"),
                make_test_scene(5, "outro"),
            ],
        )

        with pytest.raises(SceneError) as exc_info:
            render_scenes(script, tmp_path, repo_label="label")

        err = exc_info.value
        assert err.message == "The Chromium browser for Playwright is not installed."
        assert err.hint == "Run: playwright install chromium"

    def test_scene_render_failure_names_id_and_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from playwright.sync_api import Page

        def fake_screenshot(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Internal GPU context lost")

        monkeypatch.setattr(Page, "screenshot", fake_screenshot)

        script = ScriptData(
            title="Title",
            scenes=[
                make_test_scene(1, "title"),
                make_test_scene(2, "problem"),
                make_test_scene(3, "feature"),
                make_test_scene(4, "howto"),
                make_test_scene(5, "outro"),
            ],
        )

        with pytest.raises(SceneError) as exc_info:
            render_scenes(script, tmp_path, repo_label="label")

        err = exc_info.value
        assert "Failed to render scene 1 (title)" in err.message

    def test_render_60_word_heading_and_20_line_code_fits(self, tmp_path: Path) -> None:
        words = "This is a test of a sixty word heading that demonstrates how our automated typography and font fitting system dynamically resizes large titles so that developer presentation slides look completely professional clean and readable without any overflowing text or clipped characters even when artificial intelligence models generate unusually lengthy titles during automated video production cycles for modern open source developer software pipelines."
        words_60 = " ".join(words.split()[:60])
        code_20 = "\n".join(f"step_{i:02d} = execute_pipeline_task(task_id={i})" for i in range(20))

        scene = Scene.model_construct(
            id=1,
            type="howto",
            heading=words_60,
            bullets=["Step 1", "Step 2"],
            narration="Narration text for testing slide creation.",
            code=code_20,
        )
        script = ScriptData.model_construct(
            title="Title",
            scenes=[scene],
        )

        out_dir = tmp_path / "fits_test"
        paths = render_scenes(script, out_dir, repo_label="repo/test")
        assert len(paths) == 1
        assert paths[0].is_file()
        assert paths[0].stat().st_size > 0


# ==============================================================================
# 3. CLI __main__ execution tests
# ==============================================================================


class TestScenesMainCLI:
    """Tests for the python -m repo2video.scenes entrypoint."""

    @skip_if_no_chromium
    def test_main_cli_renders_slides_folder(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        scenes_file = tmp_path / "scenes.json"
        script = ScriptData(
            title="CLI Project",
            scenes=[
                make_test_scene(1, "title", "CLI Title"),
                make_test_scene(2, "problem", "CLI Problem"),
                make_test_scene(3, "feature", "CLI Feature"),
                make_test_scene(4, "howto", "CLI Howto"),
                make_test_scene(5, "outro", "CLI Outro"),
            ],
        )
        script.save(scenes_file)

        monkeypatch.setattr(
            "sys.argv",
            ["scenes.py", str(scenes_file), "--theme", "dark", "--label", "cli/test"],
        )

        main()

        captured = capsys.readouterr()
        slides_dir = tmp_path / "slides"
        assert slides_dir.is_dir()
        assert (slides_dir / "scene_01.png").is_file()
        assert "scene_01.png" in captured.out

    def test_main_cli_missing_file_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        nonexistent = tmp_path / "missing.json"
        monkeypatch.setattr("sys.argv", ["scenes.py", str(nonexistent)])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err
