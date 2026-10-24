"""Tests for repo2video data models and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repo2video.errors import Repo2VideoError
from repo2video.models import (
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
    """Verify script must have 5 to 8 scenes."""
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

    # 9 scenes is too many
    nine_scenes = (
        [_make_scene(1, "title")]
        + [_make_scene(i, "feature") for i in range(2, 9)]
        + [_make_scene(9, "outro")]
    )
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=nine_scenes)
    assert "between 5 and 8 scenes" in format_validation_error(exc_info.value)


def test_script_narration_total_over_220_words() -> None:
    """Verify total narration words over 220 fails validation."""
    # 6 scenes with 38 words each = 228 words
    narr_38 = " ".join(["word"] * 38)
    scenes = [
        _make_scene(1, "title", narration=narr_38),
        _make_scene(2, "problem", narration=narr_38),
        _make_scene(3, "feature", narration=narr_38),
        _make_scene(4, "feature", narration=narr_38),
        _make_scene(5, "howto", narration=narr_38),
        _make_scene(6, "outro", narration=narr_38),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptData(title="Test", scenes=scenes)
    msg = format_validation_error(exc_info.value)
    assert "Total narration has 228 words, the limit is 220." in msg


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
