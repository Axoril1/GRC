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


class MediaRef(BaseRepoModel):
    """Reference to an image, gif, or video extracted from repository README.

    Attributes:
        id: Identifier in sequence (m1, m2, ...).
        url: Direct media URL.
        alt: Alternative description text (may be empty).
        kind: Media type ('image', 'gif', or 'video').
    """

    id: str = Field(..., description="Media identifier (m1, m2, ...).")
    url: str = Field(..., description="Direct media URL.")
    alt: str = Field(default="", description="Alternative description text.")
    kind: str = Field(default="image", description="Media kind: image, gif, or video.")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        """Validate that media kind is image, gif, or video."""
        clean = v.strip().lower()
        if clean not in ("image", "gif", "video"):
            raise ValueError(f"Invalid media kind '{v}'. Must be image, gif, or video.")
        return clean


class KeyFile(BaseRepoModel):
    """Key source code file excerpt extracted from the repository.

    Attributes:
        id: Identifier in sequence (f1, f2, f3).
        path: Relative file path from repository root.
        language: Lowercase programming language identifier.
        content: Cleaned excerpt of the file (at most 40 lines and 2500 characters).
        total_lines: Total line count of the full source file.
    """

    id: str = Field(..., description="File identifier (f1, f2, ...).")
    path: str = Field(..., description="Relative file path.")
    language: str = Field(..., description="Programming language name.")
    content: str = Field(..., description="File content excerpt.")
    total_lines: int = Field(default=0, ge=0, description="Total line count in the full file.")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content line count and character limits."""
        lines = v.splitlines()
        if len(lines) > 40:
            raise ValueError(f"KeyFile content has {len(lines)} lines, the limit is 40.")
        if len(v) > 2500:
            raise ValueError(f"KeyFile content has {len(v)} characters, the limit is 2500.")
        return v


class TerminalSession(BaseRepoModel):
    """Terminal command session extracted from repository README.

    Attributes:
        id: Identifier in sequence (t1, t2, ...).
        purpose: Session purpose ('install' or 'usage').
        commands: List of 1 to 4 command strings.
        output: List of 0 to 8 output lines (each at most 100 characters).
    """

    id: str = Field(..., description="Session identifier (t1, t2, ...).")
    purpose: str = Field(..., description="Session purpose ('install' or 'usage').")
    commands: list[str] = Field(..., description="Command lines (1 to 4).")
    output: list[str] = Field(default_factory=list, description="Output lines (0 to 8).")

    @field_validator("purpose")
    @classmethod
    def validate_purpose(cls, v: str) -> str:
        """Validate purpose is install or usage."""
        clean = v.strip().lower()
        if clean not in ("install", "usage"):
            raise ValueError(f"Invalid terminal purpose '{v}'. Must be 'install' or 'usage'.")
        return clean

    @field_validator("commands")
    @classmethod
    def validate_commands(cls, v: list[str]) -> list[str]:
        """Validate commands count is between 1 and 4."""
        if not (1 <= len(v) <= 4):
            raise ValueError(f"TerminalSession commands must have 1 to 4 items, got {len(v)}.")
        return v

    @field_validator("output")
    @classmethod
    def validate_output(cls, v: list[str]) -> list[str]:
        """Validate output has at most 8 lines, each at most 100 characters."""
        if len(v) > 8:
            raise ValueError(f"TerminalSession output cannot exceed 8 lines, got {len(v)}.")
        for idx, line in enumerate(v):
            if len(line) > 100:
                raise ValueError(
                    f"TerminalSession output line {idx + 1} exceeds 100 characters ({len(line)})."
                )
        return v


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
        homepage: Project homepage URL (default empty).
        topics: List of topic tags (at most 10 items).
        license: SPDX license identifier or empty string.
        forks: Fork count (at least 0).
        open_issues: Open issues count (at least 0).
        created_at: ISO timestamp of repository creation.
        pushed_at: ISO timestamp of last push.
        language_percentages: Dict of top 5 languages to percentage float.
        media: List of extracted MediaRef items (at most 12).
        key_files: List of extracted KeyFile items (at most 3).
        terminals: List of extracted TerminalSession items (at most 4).
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
    homepage: str = Field(default="", description="Project homepage URL.")
    topics: list[str] = Field(default_factory=list, max_length=10, description="Repository topics.")
    license: str = Field(default="", description="SPDX license ID or empty.")
    forks: int = Field(default=0, ge=0, description="Forks count.")
    open_issues: int = Field(default=0, ge=0, description="Open issues count.")
    created_at: str = Field(default="", description="Repository creation timestamp.")
    pushed_at: str = Field(default="", description="Repository last pushed timestamp.")
    language_percentages: dict[str, float] = Field(
        default_factory=dict, description="Top 5 languages byte percentage."
    )
    media: list[MediaRef] = Field(default_factory=list, description="Extracted README media.")
    key_files: list[KeyFile] = Field(default_factory=list, description="Key source files.")
    terminals: list[TerminalSession] = Field(default_factory=list, description="Terminal sessions.")

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

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v: list[str]) -> list[str]:
        """Validate topics item count.

        Args:
            v: List of topics.

        Returns:
            Validated list.

        Raises:
            ValueError: If topics has more than 10 items.
        """
        if len(v) > 10:
            raise ValueError(f"Repository topics has {len(v)} items, the limit is 10.")
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
