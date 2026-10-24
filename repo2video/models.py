"""Data models and validation logic for repository data and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from repo2video.errors import Repo2VideoError

VALID_SCENE_TYPES = ("title", "problem", "feature", "howto", "outro")


def word_count(text: str) -> int:
    """Count whitespace-separated words in a string.

    Args:
        text: The string to inspect.

    Returns:
        The number of whitespace-separated words.
    """
    return len(text.split())


def format_validation_error(error: ValidationError) -> str:
    """Format a Pydantic ValidationError into plain-English feedback lines.

    Args:
        error: The ValidationError to format.

    Returns:
        A newline-separated string containing one plain-English sentence per line.
    """
    lines: list[str] = []
    for err in error.errors():
        msg = err.get("msg", "")
        msg = msg.removeprefix("Value error, ")
        for subline in msg.splitlines():
            subline = subline.strip()
            if subline and subline not in lines:
                lines.append(subline)
    return "\n".join(lines)


class BaseRepoModel(BaseModel):
    """Base model with shared configuration and persistence methods."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    def save(self, path: Path | str) -> None:
        """Save model instance to a JSON file with indent 2 and UTF-8 encoding.

        Creates parent folders if they do not exist.

        Args:
            path: Target file path.

        Raises:
            Repo2VideoError: If writing to the file fails.
        """
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            data = self.model_dump(mode="json")
            content = json.dumps(data, indent=2, ensure_ascii=False)
            p.write_text(content, encoding="utf-8")
        except Exception as exc:
            raise Repo2VideoError(
                f"Failed to save {self.__class__.__name__} to {p}: {exc}",
                hint=f"Ensure parent directory {p.parent} is writable.",
            ) from exc

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load and validate a model instance from a JSON file.

        Args:
            path: Path to the JSON file to read.

        Returns:
            An instance of the model.

        Raises:
            Repo2VideoError: If the file is missing or invalid.
        """
        p = Path(path)
        if not p.is_file():
            raise Repo2VideoError(
                f"File not found: {p}",
                hint=f"Ensure that {p} exists and is a valid file.",
            )
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as exc:
            raise Repo2VideoError(
                f"Failed to read file {p}: {exc}",
                hint=f"Check read permissions for {p}.",
            ) from exc

        try:
            raw_data = json.loads(content)
        except Exception as exc:
            raise Repo2VideoError(
                f"Failed to parse JSON from {p}: {exc}",
                hint=f"Ensure {p} contains valid JSON.",
            ) from exc

        try:
            return cls.model_validate(raw_data)
        except ValidationError as exc:
            formatted = format_validation_error(exc)
            raise Repo2VideoError(
                f"Validation failed for {cls.__name__} in {p}:\n{formatted}",
                hint=f"Update {p} to adhere to the {cls.__name__} schema.",
            ) from exc


class RepoData(BaseRepoModel):
    """Metadata extracted from a public GitHub repository.

    Attributes:
        name: Non-empty repository name.
        owner: Non-empty repository owner.
        description: Repository description (may be empty).
        stars: Star count (at least 0).
        languages: List of programming languages used.
        readme: Excerpt from repository README (at most 6000 characters).
        top_level: List of top-level file and directory names (at most 30 items).
    """

    name: str = Field(..., min_length=1, description="Repository name.")
    owner: str = Field(..., min_length=1, description="Repository owner.")
    description: str = Field(default="", description="Repository description.")
    stars: int = Field(default=0, ge=0, description="Star count.")
    languages: list[str] = Field(default_factory=list, description="Programming languages.")
    readme: str = Field(default="", max_length=6000, description="README excerpt.")
    top_level: list[str] = Field(
        default_factory=list, max_length=30, description="Top-level entries."
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate repository name is non-empty.

        Args:
            v: Raw repository name string.

        Returns:
            Validated string.

        Raises:
            ValueError: If name is empty.
        """
        if not v:
            raise ValueError("Repository name cannot be empty.")
        return v

    @field_validator("owner")
    @classmethod
    def validate_owner(cls, v: str) -> str:
        """Validate repository owner is non-empty.

        Args:
            v: Raw repository owner string.

        Returns:
            Validated string.

        Raises:
            ValueError: If owner is empty.
        """
        if not v:
            raise ValueError("Repository owner cannot be empty.")
        return v

    @field_validator("stars")
    @classmethod
    def validate_stars(cls, v: int) -> int:
        """Validate repository stars is non-negative.

        Args:
            v: Raw star count.

        Returns:
            Validated star count.

        Raises:
            ValueError: If stars is negative.
        """
        if v < 0:
            raise ValueError(f"Repository stars cannot be negative, got {v}.")
        return v

    @field_validator("readme")
    @classmethod
    def validate_readme(cls, v: str) -> str:
        """Validate repository readme length.

        Args:
            v: README string.

        Returns:
            Validated string.

        Raises:
            ValueError: If readme exceeds 6000 characters.
        """
        if len(v) > 6000:
            raise ValueError(f"Repository readme has {len(v)} characters, the limit is 6000.")
        return v

    @field_validator("top_level")
    @classmethod
    def validate_top_level(cls, v: list[str]) -> list[str]:
        """Validate repository top_level items count.

        Args:
            v: List of top level files/directories.

        Returns:
            Validated list.

        Raises:
            ValueError: If top_level has more than 30 items.
        """
        if len(v) > 30:
            raise ValueError(f"Repository top_level has {len(v)} items, the limit is 30.")
        return v


class Scene(BaseRepoModel):
    """A single scene within a video script.

    Attributes:
        id: Scene identifier (at least 1).
        type: Scene type ('title', 'problem', 'feature', 'howto', 'outro').
        heading: Heading text (1 to 6 words, at most 60 characters).
        bullets: List of bullet points (at most 4 items, each at most 8 words).
        narration: Narration script (15 to 40 words).
        code: Optional code snippet (default empty, at most 6 lines).
    """

    id: int = Field(..., description="Scene ID, starting at 1.")
    type: str = Field(..., description="Scene type.")
    heading: str = Field(..., description="Scene heading text.")
    bullets: list[str] = Field(default_factory=list, description="Scene bullet points.")
    narration: str = Field(..., description="Scene narration text.")
    code: str = Field(default="", description="Code snippet displayed on the slide.")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: int) -> int:
        """Validate that scene id is at least 1.

        Args:
            v: Scene ID.

        Returns:
            Validated ID.

        Raises:
            ValueError: If ID is less than 1.
        """
        if v < 1:
            raise ValueError(f"Scene id must be at least 1, got {v}.")
        return v

    @model_validator(mode="after")
    def validate_scene(self) -> Self:
        """Validate all constraints for this scene.

        Returns:
            Self if validation succeeds.

        Raises:
            ValueError: If any scene limit is exceeded or constraint is violated.
        """
        sid = self.id

        if self.type not in VALID_SCENE_TYPES:
            raise ValueError(
                f"Scene {sid}: type '{self.type}' is invalid, must be one of: "
                f"{', '.join(sorted(VALID_SCENE_TYPES))}."
            )

        h_words = word_count(self.heading)
        if h_words < 1:
            raise ValueError(f"Scene {sid}: heading has 0 words, the minimum is 1.")
        if h_words > 6:
            raise ValueError(f"Scene {sid}: heading has {h_words} words, the limit is 6.")
        if len(self.heading) > 60:
            raise ValueError(
                f"Scene {sid}: heading has {len(self.heading)} characters, the limit is 60."
            )

        if len(self.bullets) > 4:
            raise ValueError(f"Scene {sid}: bullets has {len(self.bullets)} items, the limit is 4.")
        for bullet in self.bullets:
            b_words = word_count(bullet)
            if b_words > 8:
                raise ValueError(f"Scene {sid}: bullet has {b_words} words, the limit is 8.")

        n_words = word_count(self.narration)
        if n_words < 15:
            raise ValueError(f"Scene {sid}: narration has {n_words} words, the minimum is 15.")
        if n_words > 40:
            raise ValueError(f"Scene {sid}: narration has {n_words} words, the limit is 40.")

        code_lines = self.code.splitlines()
        if len(code_lines) > 6:
            raise ValueError(f"Scene {sid}: code has {len(code_lines)} lines, the limit is 6.")

        return self


class ScriptData(BaseRepoModel):
    """Complete video script consisting of a title and ordered scenes.

    Attributes:
        title: Script title (non-empty, at most 80 characters).
        scenes: Ordered list of scenes (5 to 8 items).
    """

    title: str = Field(..., description="Script title.")
    scenes: list[Scene] = Field(..., description="Ordered list of scenes.")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate that the title is non-empty and at most 80 characters.

        Args:
            v: Title string.

        Returns:
            Validated title.

        Raises:
            ValueError: If title is empty or exceeds 80 characters.
        """
        if not v:
            raise ValueError("Script title cannot be empty.")
        if len(v) > 80:
            raise ValueError(f"Script title has {len(v)} characters, the limit is 80.")
        return v

    @field_validator("scenes")
    @classmethod
    def validate_scenes_count(cls, v: list[Scene]) -> list[Scene]:
        """Validate that the scenes list has between 5 and 8 scenes.

        Args:
            v: List of Scene instances.

        Returns:
            Validated scenes list.

        Raises:
            ValueError: If scenes list count is outside [5, 8].
        """
        if len(v) < 5 or len(v) > 8:
            raise ValueError(f"Script must have between 5 and 8 scenes, got {len(v)}.")
        return v

    @model_validator(mode="after")
    def validate_script_constraints(self) -> Self:
        """Validate script-level constraints across scenes.

        Returns:
            Self if validation succeeds.

        Raises:
            ValueError: If first scene is not title, last is not outro,
                ids are not 1..N in order, or total narration exceeds 220 words.
        """
        if not self.scenes:
            return self

        # 1. First scene has type title
        first_scene = self.scenes[0]
        if first_scene.type != "title":
            raise ValueError(
                f"Scene {first_scene.id}: first scene must have type 'title', got '{first_scene.type}'."
            )

        # 2. Last scene has type outro
        last_scene = self.scenes[-1]
        if last_scene.type != "outro":
            raise ValueError(
                f"Scene {last_scene.id}: last scene must have type 'outro', got '{last_scene.type}'."
            )

        # 3. Ids are exactly 1 to N in order
        expected_ids = list(range(1, len(self.scenes) + 1))
        actual_ids = [s.id for s in self.scenes]
        if actual_ids != expected_ids:
            raise ValueError(
                f"Scene ids must be exactly 1 to {len(self.scenes)} in order, got {actual_ids}."
            )

        # 4. Total narration is at most 220 words
        total_words = sum(word_count(s.narration) for s in self.scenes)
        if total_words > 220:
            raise ValueError(f"Total narration has {total_words} words, the limit is 220.")

        return self
