"""Data models and validation logic for repository data and scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from GRC.errors import Repo2VideoError

V1_SCENE_TYPES = ("title", "problem", "feature", "howto", "outro")
V2_SCENE_TYPES = ("media", "code", "terminal", "architecture", "stats")
VALID_SCENE_TYPES = V1_SCENE_TYPES + V2_SCENE_TYPES

VALID_STAT_KEYS = (
    "stars",
    "forks",
    "issues",
    "license",
    "updated",
    "languages",
    "topics",
)

VALID_NODE_GROUPS = ("input", "core", "output", "storage", "external")


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


@dataclass(frozen=True)
class MediaAsset:
    """Downloaded media asset descriptor.

    Attributes:
        id: Identifier matching a MediaRef id or screenshot (e.g. m1, h1).
        kind: Media kind ('image' or 'clip').
        path: File path on disk.
        width: Image or video width in pixels.
        height: Image or video height in pixels.
        duration_seconds: Duration in seconds (0.0 for static images).
        alt: Alternative description text.
        origin: Source origin ('readme' or 'homepage').
        source_url: Source URL where the asset was fetched from.
    """

    id: str
    kind: str
    path: Path = Path()
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0
    alt: str = ""
    origin: str = "readme"
    source_url: str = ""


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
        """Validate repository name is non-empty."""
        if not v:
            raise ValueError("Repository name cannot be empty.")
        return v

    @field_validator("owner")
    @classmethod
    def validate_owner(cls, v: str) -> str:
        """Validate repository owner is non-empty."""
        if not v:
            raise ValueError("Repository owner cannot be empty.")
        return v

    @field_validator("stars")
    @classmethod
    def validate_stars(cls, v: int) -> int:
        """Validate repository stars is non-negative."""
        if v < 0:
            raise ValueError(f"Repository stars cannot be negative, got {v}.")
        return v

    @field_validator("readme")
    @classmethod
    def validate_readme(cls, v: str) -> str:
        """Validate repository readme length."""
        if len(v) > 6000:
            raise ValueError(f"Repository readme has {len(v)} characters, the limit is 6000.")
        return v

    @field_validator("top_level")
    @classmethod
    def validate_top_level(cls, v: list[str]) -> list[str]:
        """Validate repository top_level items count."""
        if len(v) > 30:
            raise ValueError(f"Repository top_level has {len(v)} items, the limit is 30.")
        return v

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v: list[str]) -> list[str]:
        """Validate topics item count."""
        if len(v) > 10:
            raise ValueError(f"Repository topics has {len(v)} items, the limit is 10.")
        return v


class FocusRange(BaseRepoModel):
    """Line number range to highlight in a code scene.

    Attributes:
        start: 1-based inclusive start line number.
        end: 1-based inclusive end line number.
    """

    start: int = Field(..., ge=1, description="1-based inclusive start line number.")
    end: int = Field(..., ge=1, description="1-based inclusive end line number.")

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Validate that start <= end and the span is at most 8 lines."""
        if self.start > self.end:
            raise ValueError(
                f"Focus start line {self.start} must be less than or equal to end line {self.end}."
            )
        span = self.end - self.start + 1
        if span > 8:
            raise ValueError(
                f"Focus range {self.start}-{self.end} spans {span} lines, the limit is 8."
            )
        return self


class NodeSpec(BaseRepoModel):
    """A node in an architecture diagram.

    Attributes:
        id: Short lowercase identifier (letters and digits).
        label: Short component label (1 to 3 words, at most 24 characters).
        group: Node category ('input', 'core', 'output', 'storage', 'external').
    """

    id: str = Field(..., description="Short lowercase identifier (letters and digits).")
    label: str = Field(..., description="Node label (1 to 3 words, at most 24 characters).")
    group: str = Field(default="core", description="Node group category.")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate node id consists of lowercase letters and digits."""
        clean = v.strip().lower()
        if not clean or not re.match(r"^[a-z0-9]+$", clean):
            raise ValueError(f"Node id '{v}' must consist of lowercase letters and digits.")
        return clean

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate node label length and word count."""
        words = word_count(v)
        if words < 1:
            raise ValueError("Node label cannot be empty.")
        if words > 3:
            raise ValueError(f"Node label '{v}' has {words} words, the limit is 3.")
        if len(v) > 24:
            raise ValueError(f"Node label '{v}' has {len(v)} characters, the limit is 24.")
        return v

    @field_validator("group")
    @classmethod
    def validate_group(cls, v: str) -> str:
        """Validate node group category."""
        clean = v.strip().lower()
        if clean not in VALID_NODE_GROUPS:
            raise ValueError(
                f"Node group '{v}' is invalid, must be one of: {', '.join(VALID_NODE_GROUPS)}."
            )
        return clean


class EdgeSpec(BaseRepoModel):
    """A directed edge between two architecture nodes.

    Attributes:
        source: Origin node id.
        target: Destination node id.
        label: Optional edge label (at most 3 words, may be empty).
    """

    source: str = Field(..., description="Source node id.")
    target: str = Field(..., description="Target node id.")
    label: str = Field(default="", description="Edge label (at most 3 words).")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate edge label word count."""
        words = word_count(v)
        if words > 3:
            raise ValueError(f"Edge label '{v}' has {words} words, the limit is 3.")
        return v


class ArchitectureSpec(BaseRepoModel):
    """Architecture diagram specification with nodes and directed edges.

    Attributes:
        nodes: List of 3 to 8 NodeSpec instances.
        edges: List of at most 12 EdgeSpec instances.
    """

    nodes: list[NodeSpec] = Field(..., description="Nodes in the diagram (3 to 8).")
    edges: list[EdgeSpec] = Field(default_factory=list, description="Directed edges (at most 12).")

    @field_validator("nodes")
    @classmethod
    def validate_nodes_count(cls, v: list[NodeSpec]) -> list[NodeSpec]:
        """Validate node count and uniqueness of ids."""
        if len(v) < 3 or len(v) > 8:
            raise ValueError(f"ArchitectureSpec must have between 3 and 8 nodes, got {len(v)}.")
        ids = [n.id for n in v]
        if len(ids) != len(set(ids)):
            raise ValueError("ArchitectureSpec nodes must have unique ids.")
        return v

    @field_validator("edges")
    @classmethod
    def validate_edges_count(cls, v: list[EdgeSpec]) -> list[EdgeSpec]:
        """Validate edge count."""
        if len(v) > 12:
            raise ValueError(f"ArchitectureSpec edges cannot exceed 12, got {len(v)}.")
        return v

    @model_validator(mode="after")
    def validate_connections(self) -> Self:
        """Validate that all edge endpoints exist and nodes are connected."""
        node_ids = {n.id for n in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source '{edge.source}' does not exist in nodes.")
            if edge.target not in node_ids:
                raise ValueError(f"Edge target '{edge.target}' does not exist in nodes.")

        if len(self.edges) >= 2:
            connected = {edge.source for edge in self.edges} | {edge.target for edge in self.edges}
            isolated = node_ids - connected
            if isolated:
                raise ValueError(
                    f"Every node must have at least one edge unless there are fewer than 2 edges in total; "
                    f"isolated nodes: {', '.join(sorted(isolated))}."
                )
        return self


class Scene(BaseRepoModel):
    """A single scene within a video script.

    Attributes:
        id: Scene identifier (at least 1).
        type: Scene type ('title', 'problem', 'feature', 'howto', 'outro', 'media', 'code', 'terminal', 'architecture', 'stats').
        heading: Heading text (1 to 6 words, at most 60 characters).
        bullets: List of bullet points (at most 4 items, each at most 8 words).
        narration: Narration script (15 to 40 words).
        code: Optional code snippet (default empty, at most 6 lines).
        media_id: Optional media identifier for media scenes.
        file_id: Optional key file identifier for code scenes.
        focus: Optional list of line ranges to highlight in a code scene (at most 4).
        terminal_id: Optional terminal session identifier for terminal scenes.
        architecture: Optional architecture diagram spec for architecture scenes.
        stat_keys: Optional repository stat keys to display (at most 4).
    """

    id: int = Field(..., description="Scene ID, starting at 1.")
    type: str = Field(..., description="Scene type.")
    heading: str = Field(..., description="Scene heading text.")
    bullets: list[str] = Field(default_factory=list, description="Scene bullet points.")
    narration: str = Field(..., description="Scene narration text.")
    code: str = Field(default="", description="Code snippet displayed on the slide.")
    media_id: str | None = Field(default=None, description="Referenced media ID.")
    file_id: str | None = Field(default=None, description="Referenced key file ID.")
    focus: list[FocusRange] = Field(default_factory=list, description="Highlighted line ranges.")
    terminal_id: str | None = Field(default=None, description="Referenced terminal session ID.")
    architecture: ArchitectureSpec | None = Field(
        default=None, description="Architecture diagram spec."
    )
    stat_keys: list[str] = Field(
        default_factory=list, description="Repository stat keys to display."
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: int) -> int:
        """Validate that scene id is at least 1."""
        if v < 1:
            raise ValueError(f"Scene id must be at least 1, got {v}.")
        return v

    @model_validator(mode="after")
    def validate_scene(self) -> Self:
        """Validate all constraints for this scene."""
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

        if len(self.focus) > 4:
            raise ValueError(f"Scene {sid}: focus has {len(self.focus)} items, the limit is 4.")

        for key in self.stat_keys:
            if key not in VALID_STAT_KEYS:
                raise ValueError(
                    f"Scene {sid}: invalid stat_key '{key}', must be one of: {', '.join(VALID_STAT_KEYS)}."
                )
        if len(self.stat_keys) > 4:
            raise ValueError(
                f"Scene {sid}: stat_keys has {len(self.stat_keys)} items, the limit is 4."
            )

        # Type-specific requirements
        if self.type == "media" and not self.media_id:
            raise ValueError(f"Scene {sid}: type media needs a media_id.")
        if self.type == "code" and not self.file_id:
            raise ValueError(f"Scene {sid}: type code needs a file_id.")
        if self.type == "terminal" and not self.terminal_id:
            raise ValueError(f"Scene {sid}: type terminal needs a terminal_id.")
        if self.type == "architecture" and not self.architecture:
            raise ValueError(f"Scene {sid}: type architecture needs architecture.")
        if self.type == "stats" and (len(self.stat_keys) < 2 or len(self.stat_keys) > 4):
            raise ValueError(f"Scene {sid}: type stats needs 2 to 4 stat_keys.")

        return self


class ScriptData(BaseRepoModel):
    """Complete video script consisting of a title and ordered scenes.

    Attributes:
        title: Script title (non-empty, at most 80 characters).
        scenes: Ordered list of scenes (5 to 8 for plain, 6 to 9 if visual scenes used).
    """

    title: str = Field(..., description="Script title.")
    scenes: list[Scene] = Field(..., description="Ordered list of scenes.")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate that the title is non-empty and at most 80 characters."""
        if not v:
            raise ValueError("Script title cannot be empty.")
        if len(v) > 80:
            raise ValueError(f"Script title has {len(v)} characters, the limit is 80.")
        return v

    @field_validator("scenes")
    @classmethod
    def validate_scenes_count(cls, v: list[Scene]) -> list[Scene]:
        """Validate scene counts based on whether visual scene types are used."""
        has_v2 = any(s.type in V2_SCENE_TYPES for s in v)
        if has_v2:
            if len(v) < 6 or len(v) > 9:
                raise ValueError(
                    f"Script with visual scenes must have between 6 and 9 scenes, got {len(v)}."
                )
        else:
            if len(v) < 5 or len(v) > 8:
                raise ValueError(f"Script must have between 5 and 8 scenes, got {len(v)}.")
        return v

    @model_validator(mode="after")
    def validate_script_constraints(self) -> Self:
        """Validate script-level constraints across scenes."""
        if not self.scenes:
            return self

        # Limits per visual scene type
        media_count = sum(1 for s in self.scenes if s.type == "media")
        if media_count > 2:
            raise ValueError(f"Script can have at most 2 media scenes, got {media_count}.")

        code_count = sum(1 for s in self.scenes if s.type == "code")
        if code_count > 2:
            raise ValueError(f"Script can have at most 2 code scenes, got {code_count}.")

        terminal_count = sum(1 for s in self.scenes if s.type == "terminal")
        if terminal_count > 1:
            raise ValueError(f"Script can have at most 1 terminal scene, got {terminal_count}.")

        arch_count = sum(1 for s in self.scenes if s.type == "architecture")
        if arch_count > 1:
            raise ValueError(f"Script can have at most 1 architecture scene, got {arch_count}.")

        stats_count = sum(1 for s in self.scenes if s.type == "stats")
        if stats_count > 1:
            raise ValueError(f"Script can have at most 1 stats scene, got {stats_count}.")

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

        # 4. Total narration is at most 240 words
        total_words = sum(word_count(s.narration) for s in self.scenes)
        if total_words > 240:
            raise ValueError(f"Total narration has {total_words} words, the limit is 240.")

        return self
