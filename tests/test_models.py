"""Tests for repo2video data models and validation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo2video.errors import Repo2VideoError
from repo2video.models import (
    ArchitectureSpec,
    EdgeSpec,
    FocusRange,
    NodeSpec,
    RepoData,
    Scene,
    ScriptData,
    format_validation_error,
    word_count,
)


def _make_scene(
    scene_id: int = 1,
    scene_type: str = "feature",
    heading: str = "Scene Heading",
    bullets: list[str] | None = None,
    narration: str = "This is a valid narration passage designed to meet the minimum length requirements for testing.",
    code: str = "",
    **extra: object,
) -> Scene:
    """Helper to create a valid Scene with customizable overrides."""
    return Scene(
        id=scene_id,
        type=scene_type,
        heading=heading,
        bullets=bullets if bullets is not None else ["Key bullet point", "Another point"],
        narration=narration,
        code=code,
        **extra,
    )


def _make_valid_script() -> ScriptData:
    """Helper to create a valid 5-scene ScriptData."""
    scenes = [
        _make_scene(1, "title", "Project Overview"),
        _make_scene(2, "problem", "The Challenge"),
        _make_scene(3, "feature", "Key Capabilities"),
        _make_scene(4, "howto", "Quick Setup"),
        _make_scene(5, "outro", "Get Started Today"),
    ]
    return ScriptData(title="Repo Demo Video", scenes=scenes)


def test_word_count() -> None:
    """Verify word_count accurately counts whitespace-separated words."""
    assert word_count("Python 3.11") == 2
    assert word_count("") == 0
    assert word_count("   ") == 0
    assert word_count("Word") == 1
    assert word_count("One   Two \n Three\tFour") == 4


def test_script_data_round_trip(tmp_path: Path) -> None:
    """Verify a valid ScriptData round trip through save and load."""
    script = _make_valid_script()
    file_path = tmp_path / "subdir" / "script.json"

    script.save(file_path)
    assert file_path.is_file()

    loaded = ScriptData.load(file_path)
    assert loaded == script
    assert len(loaded.scenes) == 5
    assert loaded.title == "Repo Demo Video"


def test_repo_data_round_trip(tmp_path: Path) -> None:
    """Verify RepoData round trip through save and load."""
    repo = RepoData(
        name="repo2video",
        owner="axoril",
        description="Demo video generator",
        stars=100,
        languages=["Python", "Shell"],
        readme="# README content",
        top_level=["pyproject.toml", "repo2video/"],
    )
    file_path = tmp_path / "repo.json"
    repo.save(file_path)

    loaded = RepoData.load(file_path)
    assert loaded == repo
    assert loaded.name == "repo2video"


def test_v1_scenes_json_still_validates(tmp_path: Path) -> None:
    """Verify that a legacy v1 scenes.json file still loads and validates cleanly."""
    v1_content = {
        "title": "Legacy Project Video",
        "scenes": [
            {
                "id": 1,
                "type": "title",
                "heading": "Welcome to Project",
                "bullets": ["Open source", "Python powered"],
                "narration": "Welcome to this legacy demo video showcasing our python project in full action across multiple developer platforms today.",
                "code": "",
            },
            {
                "id": 2,
                "type": "problem",
                "heading": "The Pain Point",
                "bullets": ["Complex setup", "Slow builds"],
                "narration": "Developers struggle every day with slow manual video creation pipelines that take hours to run in production.",
                "code": "",
            },
            {
                "id": 3,
                "type": "feature",
                "heading": "Instant Conversion",
                "bullets": ["Automated slides", "Fast speech"],
                "narration": "Our engine automates the entire process transforming your repository source code into smooth video presentation files.",
                "code": "",
            },
            {
                "id": 4,
                "type": "howto",
                "heading": "Getting Started",
                "bullets": ["Single pip install", "Run CLI"],
                "narration": "Install with pip in seconds and run the simple command line tool to produce your complete video.",
                "code": "pip install repo2video",
            },
            {
                "id": 5,
                "type": "outro",
                "heading": "Try It Today",
                "bullets": ["Free software", "MIT license"],
                "narration": "Check out the repository on GitHub today and star the project to support our ongoing open source development.",
                "code": "",
            },
        ],
    }
    json_path = tmp_path / "v1_scenes.json"
    json_path.write_text(json.dumps(v1_content, indent=2), encoding="utf-8")

    loaded = ScriptData.load(json_path)
    assert loaded.title == "Legacy Project Video"
    assert len(loaded.scenes) == 5
    assert loaded.scenes[0].type == "title"
    assert loaded.scenes[0].media_id is None
    assert loaded.scenes[0].architecture is None
    assert loaded.scenes[-1].type == "outro"


def test_scene_limit_heading_zero_words() -> None:
    """Verify heading with 0 words fails with message naming scene id."""
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, heading="")
    msg = format_validation_error(exc_info.value)
    assert "Scene 3" in msg
    assert "heading" in msg
    assert "0 words" in msg or "minimum" in msg


def test_scene_limit_heading_too_many_words() -> None:
    """Verify heading with > 6 words fails with message naming scene id."""
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, heading="One Two Three Four Five Six Seven")
    msg = format_validation_error(exc_info.value)
    assert "Scene 3" in msg
    assert "heading has 7 words, the limit is 6" in msg


def test_scene_limit_heading_too_long_chars() -> None:
    """Verify heading > 60 chars fails with message naming scene id."""
    long_heading = "A" * 65
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, heading=long_heading)
    msg = format_validation_error(exc_info.value)
    assert "Scene 3" in msg
    assert "65 characters, the limit is 60" in msg


def test_scene_limit_bullets_too_many_items() -> None:
    """Verify bullets > 4 items fails with message naming scene id."""
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, bullets=["one", "two", "three", "four", "five"])
    msg = format_validation_error(exc_info.value)
    assert "Scene 3" in msg
    assert "bullets has 5 items, the limit is 4" in msg


def test_scene_limit_bullet_too_many_words() -> None:
    """Verify bullet > 8 words fails with message naming scene id."""
    long_bullet = "One two three four five six seven eight nine"
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, bullets=[long_bullet])
    msg = format_validation_error(exc_info.value)
    assert "Scene 3" in msg
    assert "bullet has 9 words, the limit is 8" in msg


def test_scene_limit_narration_too_few_words() -> None:
    """Verify narration < 15 words fails with message naming scene id."""
    short_narration = "Only five words in this."
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, narration=short_narration)
    msg = format_validation_error(exc_info.value)
    assert "Scene 3" in msg
    assert "narration has 5 words, the minimum is 15" in msg


def test_scene_limit_narration_too_many_words() -> None:
    """Verify narration > 40 words fails with message naming scene id."""
    long_narration = " ".join(["word"] * 52)
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, narration=long_narration)
    msg = format_validation_error(exc_info.value)
    assert "Scene 3: narration has 52 words, the limit is 40." in msg


def test_scene_limit_code_too_many_lines() -> None:
    """Verify code > 6 lines fails with message naming scene id."""
    code_7_lines = "\n".join([f"line_{i}" for i in range(7)])
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, code=code_7_lines)
    msg = format_validation_error(exc_info.value)
    assert "Scene 3" in msg
    assert "code has 7 lines, the limit is 6" in msg


def test_scene_invalid_type() -> None:
    """Verify invalid scene type fails with message naming scene id."""
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, scene_type="invalid_type")
    msg = format_validation_error(exc_info.value)
    assert "Scene 3" in msg
    assert "invalid_type" in msg


def test_scene_id_less_than_one() -> None:
    """Verify scene id < 1 fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=0)
    msg = format_validation_error(exc_info.value)
    assert "Scene id must be at least 1" in msg


def test_new_types_missing_required_fields_give_clear_messages() -> None:
    """Verify each visual scene type gives a clear plain-English message when required field is missing."""
    # media needs media_id
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=2, scene_type="media", media_id=None)
    assert "Scene 2: type media needs a media_id." in format_validation_error(exc_info.value)

    # code needs file_id
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=3, scene_type="code", file_id=None)
    assert "Scene 3: type code needs a file_id." in format_validation_error(exc_info.value)

    # terminal needs terminal_id
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=4, scene_type="terminal", terminal_id=None)
    assert "Scene 4: type terminal needs a terminal_id." in format_validation_error(exc_info.value)

    # architecture needs architecture
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=5, scene_type="architecture", architecture=None)
    assert "Scene 5: type architecture needs architecture." in format_validation_error(
        exc_info.value
    )

    # stats needs 2 to 4 stat_keys
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=6, scene_type="stats", stat_keys=[])
    assert "Scene 6: type stats needs 2 to 4 stat_keys." in format_validation_error(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        _make_scene(scene_id=7, scene_type="stats", stat_keys=["stars"])
    assert "Scene 7: type stats needs 2 to 4 stat_keys." in format_validation_error(exc_info.value)


def test_limits_per_type() -> None:
    """Verify script limits: at most 2 media, 2 code, 1 terminal, 1 architecture, 1 stats."""
    # 3 media scenes should fail
    arch = ArchitectureSpec(
        nodes=[
            NodeSpec(id="cli", label="CLI Tool"),
            NodeSpec(id="core", label="Core Engine"),
            NodeSpec(id="out", label="Output Store"),
        ],
        edges=[
            EdgeSpec(source="cli", target="core", label="calls"),
            EdgeSpec(source="core", target="out", label="writes"),
        ],
    )
    scenes_3_media = [
        _make_scene(1, "title"),
        _make_scene(2, "media", media_id="m1"),
        _make_scene(3, "media", media_id="m2"),
        _make_scene(4, "media", media_id="m3"),
        _make_scene(5, "problem"),
        _make_scene(6, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes_3_media)
    assert "at most 2 media scenes" in format_validation_error(exc_info.value)

    # 3 code scenes should fail
    scenes_3_code = [
        _make_scene(1, "title"),
        _make_scene(2, "code", file_id="f1"),
        _make_scene(3, "code", file_id="f2"),
        _make_scene(4, "code", file_id="f3"),
        _make_scene(5, "problem"),
        _make_scene(6, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes_3_code)
    assert "at most 2 code scenes" in format_validation_error(exc_info.value)

    # 2 terminal scenes should fail
    scenes_2_term = [
        _make_scene(1, "title"),
        _make_scene(2, "terminal", terminal_id="t1"),
        _make_scene(3, "terminal", terminal_id="t2"),
        _make_scene(4, "problem"),
        _make_scene(5, "feature"),
        _make_scene(6, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes_2_term)
    assert "at most 1 terminal scene" in format_validation_error(exc_info.value)

    # 2 architecture scenes should fail
    scenes_2_arch = [
        _make_scene(1, "title"),
        _make_scene(2, "architecture", architecture=arch),
        _make_scene(3, "architecture", architecture=arch),
        _make_scene(4, "problem"),
        _make_scene(5, "feature"),
        _make_scene(6, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes_2_arch)
    assert "at most 1 architecture scene" in format_validation_error(exc_info.value)

    # 2 stats scenes should fail
    scenes_2_stats = [
        _make_scene(1, "title"),
        _make_scene(2, "stats", stat_keys=["stars", "forks"]),
        _make_scene(3, "stats", stat_keys=["issues", "license"]),
        _make_scene(4, "problem"),
        _make_scene(5, "feature"),
        _make_scene(6, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes_2_stats)
    assert "at most 1 stats scene" in format_validation_error(exc_info.value)


def test_focus_validation() -> None:
    """Verify focus line range validation rules."""
    # Overlap allowed
    r1 = FocusRange(start=1, end=5)
    r2 = FocusRange(start=3, end=8)
    assert r1.start == 1 and r1.end == 5
    assert r2.start == 3 and r2.end == 8

    # More than 8 lines rejected (e.g. 1 to 10 spans 10 lines)
    with pytest.raises(ValidationError) as exc_info:
        FocusRange(start=1, end=10)
    assert "spans 10 lines, the limit is 8" in format_validation_error(exc_info.value)

    # Start greater than end rejected
    with pytest.raises(ValidationError) as exc_info:
        FocusRange(start=5, end=2)
    assert "must be less than or equal to end line" in format_validation_error(exc_info.value)

    # More than 4 focus items on a Scene rejected
    with pytest.raises(ValidationError) as exc_info:
        _make_scene(
            scene_id=3,
            scene_type="code",
            file_id="f1",
            focus=[FocusRange(start=i, end=i) for i in range(1, 6)],
        )
    assert "focus has 5 items, the limit is 4" in format_validation_error(exc_info.value)


def test_architecture_validation() -> None:
    """Verify architecture specification validation rules."""
    # Too few nodes (< 3)
    with pytest.raises(ValidationError) as exc_info:
        ArchitectureSpec(
            nodes=[
                NodeSpec(id="a", label="A"),
                NodeSpec(id="b", label="B"),
            ],
            edges=[],
        )
    assert "between 3 and 8 nodes" in format_validation_error(exc_info.value)

    # Too many nodes (> 8)
    with pytest.raises(ValidationError) as exc_info:
        ArchitectureSpec(
            nodes=[NodeSpec(id=f"n{i}", label=f"Node {i}") for i in range(9)],
            edges=[],
        )
    assert "between 3 and 8 nodes" in format_validation_error(exc_info.value)

    # Duplicate node ids
    with pytest.raises(ValidationError) as exc_info:
        ArchitectureSpec(
            nodes=[
                NodeSpec(id="core", label="Core One"),
                NodeSpec(id="core", label="Core Two"),
                NodeSpec(id="out", label="Output"),
            ],
            edges=[],
        )
    assert "unique ids" in format_validation_error(exc_info.value)

    # Unknown edge ends
    with pytest.raises(ValidationError) as exc_info:
        ArchitectureSpec(
            nodes=[
                NodeSpec(id="a", label="A"),
                NodeSpec(id="b", label="B"),
                NodeSpec(id="c", label="C"),
            ],
            edges=[EdgeSpec(source="a", target="ghost", label="calls")],
        )
    assert "does not exist in nodes" in format_validation_error(exc_info.value)

    # Isolated nodes when total edges >= 2
    with pytest.raises(ValidationError) as exc_info:
        ArchitectureSpec(
            nodes=[
                NodeSpec(id="a", label="A"),
                NodeSpec(id="b", label="B"),
                NodeSpec(id="c", label="C"),
            ],
            edges=[
                EdgeSpec(source="a", target="b", label="calls"),
                EdgeSpec(source="b", target="a", label="returns"),
            ],
        )
    assert "isolated nodes: c" in format_validation_error(exc_info.value)

    # Fewer than 2 edges allows isolated nodes
    spec_single_edge = ArchitectureSpec(
        nodes=[
            NodeSpec(id="a", label="A"),
            NodeSpec(id="b", label="B"),
            NodeSpec(id="c", label="C"),
        ],
        edges=[EdgeSpec(source="a", target="b", label="calls")],
    )
    assert len(spec_single_edge.edges) == 1


def test_visual_script_scene_count_bounds() -> None:
    """Verify script scene count bounds for visual vs plain scripts."""
    # Plain script allows 5 to 8
    plain_5 = [
        _make_scene(1, "title"),
        _make_scene(2, "problem"),
        _make_scene(3, "feature"),
        _make_scene(4, "howto"),
        _make_scene(5, "outro"),
    ]
    assert len(ScriptData(title="T", scenes=plain_5).scenes) == 5

    # 5 scenes with a visual scene fails (visual requires 6 to 9)
    vis_5 = [
        _make_scene(1, "title"),
        _make_scene(2, "problem"),
        _make_scene(3, "media", media_id="m1"),
        _make_scene(4, "howto"),
        _make_scene(5, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="T", scenes=vis_5)
    assert "between 6 and 9 scenes" in format_validation_error(exc_info.value)

    # 6 to 9 scenes with visual scene succeeds
    vis_9 = (
        [_make_scene(1, "title"), _make_scene(2, "media", media_id="m1")]
        + [_make_scene(i, "feature") for i in range(3, 9)]
        + [_make_scene(9, "outro")]
    )
    assert len(ScriptData(title="T", scenes=vis_9).scenes) == 9

    # 10 scenes with visual scene fails
    vis_10 = vis_9 + [_make_scene(10, "outro")]
    vis_10[-2].type = "feature"
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="T", scenes=vis_10)
    assert "between 6 and 9 scenes" in format_validation_error(exc_info.value)


def test_script_wrong_first_scene_type() -> None:
    """Verify first scene must be type title."""
    scenes = [
        _make_scene(1, "problem"),
        _make_scene(2, "feature"),
        _make_scene(3, "feature"),
        _make_scene(4, "howto"),
        _make_scene(5, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes)
    msg = format_validation_error(exc_info.value)
    assert "Scene 1" in msg
    assert "first scene must have type 'title'" in msg


def test_script_wrong_last_scene_type() -> None:
    """Verify last scene must be type outro."""
    scenes = [
        _make_scene(1, "title"),
        _make_scene(2, "problem"),
        _make_scene(3, "feature"),
        _make_scene(4, "howto"),
        _make_scene(5, "feature"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes)
    msg = format_validation_error(exc_info.value)
    assert "Scene 5" in msg
    assert "last scene must have type 'outro'" in msg


def test_script_non_sequential_ids() -> None:
    """Verify non-sequential scene ids fail validation."""
    scenes = [
        _make_scene(1, "title"),
        _make_scene(2, "problem"),
        _make_scene(4, "feature"),
        _make_scene(5, "howto"),
        _make_scene(6, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes)
    msg = format_validation_error(exc_info.value)
    assert "Scene ids must be exactly 1 to 5 in order" in msg


def test_script_scenes_count_bounds() -> None:
    """Verify script must have 5 to 8 scenes for plain scripts."""
    # 4 scenes is too few
    four_scenes = [
        _make_scene(1, "title"),
        _make_scene(2, "problem"),
        _make_scene(3, "feature"),
        _make_scene(4, "outro"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=four_scenes)
    assert "between 5 and 8 scenes" in format_validation_error(exc_info.value)

    # 9 plain scenes is too many
    nine_scenes = (
        [_make_scene(1, "title")]
        + [_make_scene(i, "feature") for i in range(2, 9)]
        + [_make_scene(9, "outro")]
    )
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=nine_scenes)
    assert "between 5 and 8 scenes" in format_validation_error(exc_info.value)


def test_script_narration_total_over_240_words() -> None:
    """Verify total narration words over 240 fails validation while 228 words passes."""
    # 6 scenes with 38 words each = 228 words (passed in v2!)
    narr_38 = " ".join(["word"] * 38)
    scenes_228 = [
        _make_scene(1, "title", narration=narr_38),
        _make_scene(2, "problem", narration=narr_38),
        _make_scene(3, "feature", narration=narr_38),
        _make_scene(4, "feature", narration=narr_38),
        _make_scene(5, "howto", narration=narr_38),
        _make_scene(6, "outro", narration=narr_38),
    ]
    valid_script = ScriptData(title="Test", scenes=scenes_228)
    assert valid_script is not None

    # 7 scenes with 36 words each = 252 words (> 240 words fails)
    narr_36 = " ".join(["word"] * 36)
    scenes_252 = [
        _make_scene(1, "title", narration=narr_36),
        _make_scene(2, "problem", narration=narr_36),
        _make_scene(3, "feature", narration=narr_36),
        _make_scene(4, "feature", narration=narr_36),
        _make_scene(5, "feature", narration=narr_36),
        _make_scene(6, "howto", narration=narr_36),
        _make_scene(7, "outro", narration=narr_36),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes_252)
    msg = format_validation_error(exc_info.value)
    assert "Total narration has 252 words, the limit is 240." in msg


def test_extra_fields_ignored() -> None:
    """Verify extra unknown fields are ignored on Scene and ScriptData."""
    scene = _make_scene(
        scene_id=1,
        scene_type="title",
        extra_unknown_field="should be ignored",
    )
    assert not hasattr(scene, "extra_unknown_field")

    script = ScriptData(
        title="Test Title",
        scenes=[
            _make_scene(1, "title"),
            _make_scene(2, "problem"),
            _make_scene(3, "feature"),
            _make_scene(4, "howto"),
            _make_scene(5, "outro"),
        ],
        unexpected_extra_property=42,
    )
    assert not hasattr(script, "unexpected_extra_property")


def test_whitespace_stripped() -> None:
    """Verify leading and trailing whitespace is stripped before validation."""
    scene = Scene(
        id=1,
        type="  title  ",
        heading="   Demo Title   ",
        bullets=["   bullet item one   ", "  bullet item two  "],
        narration="   This is a valid narration passage designed to meet the minimum length requirements for testing.   ",
        code="   def foo(): pass   ",
    )
    assert scene.type == "title"
    assert scene.heading == "Demo Title"
    assert scene.bullets == ["bullet item one", "bullet item two"]
    assert (
        scene.narration
        == "This is a valid narration passage designed to meet the minimum length requirements for testing."
    )
    assert scene.code == "def foo(): pass"

    script = ScriptData(
        title="   Trimmed Title   ",
        scenes=[
            scene,
            _make_scene(2, "problem"),
            _make_scene(3, "feature"),
            _make_scene(4, "howto"),
            _make_scene(5, "outro"),
        ],
    )
    assert script.title == "Trimmed Title"


def test_load_missing_file_raises_repo2video_error(tmp_path: Path) -> None:
    """Verify loading a missing file raises Repo2VideoError with a hint."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(Repo2VideoError) as exc_info:
        ScriptData.load(missing)
    assert "File not found" in exc_info.value.message
    assert exc_info.value.hint is not None
    assert "Hint:" in str(exc_info.value)


def test_load_invalid_json_raises_repo2video_error(tmp_path: Path) -> None:
    """Verify loading an invalid JSON file raises Repo2VideoError with a hint."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(Repo2VideoError) as exc_info:
        ScriptData.load(bad_json)
    assert "Failed to parse JSON" in exc_info.value.message
    assert exc_info.value.hint is not None


def test_load_invalid_schema_raises_repo2video_error(tmp_path: Path) -> None:
    """Verify loading valid JSON that violates schema raises Repo2VideoError with a hint."""
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text('{"title": "", "scenes": []}', encoding="utf-8")
    with pytest.raises(Repo2VideoError) as exc_info:
        ScriptData.load(bad_schema)
    assert "Validation failed" in exc_info.value.message
    assert exc_info.value.hint is not None


def test_repo_data_validation() -> None:
    """Verify RepoData validation constraints."""
    with pytest.raises(ValidationError):
        RepoData(name="", owner="axoril", stars=0)

    with pytest.raises(ValidationError):
        RepoData(name="test", owner="", stars=0)

    with pytest.raises(ValidationError):
        RepoData(name="test", owner="axoril", stars=-1)

    with pytest.raises(ValidationError):
        RepoData(name="test", owner="axoril", stars=0, readme="a" * 6001)

    with pytest.raises(ValidationError):
        RepoData(name="test", owner="axoril", stars=0, top_level=["file"] * 31)
