"""Command-line interface built with Typer for running the video generation pipeline."""

from __future__ import annotations

import logging
import re
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer

from GRC import __version__
from GRC.config import Settings, get_settings
from GRC.errors import Repo2VideoError
from GRC.logging_setup import setup_logging
from GRC.models import RepoData, ScriptData
from GRC.pipeline import (
    PipelineOptions,
    PipelineResult,
    check_doctor,
    render_only,
)
from GRC.pipeline import run as run_pipeline

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="grc",
    help="Turn a public GitHub repository into a narrated demo video.",
    no_args_is_help=True,
)

STAGE_DESCRIPTIONS: dict[str, str] = {
    "analyze": "Analyzing repository",
    "media": "Collecting media assets",
    "script": "Writing the script",
    "voice": "Synthesizing voice narration",
    "visuals": "Rendering visual scenes",
    "render": "Compositing video with FFmpeg",
}

HEX_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
ALLOWED_MUSIC_EXTENSIONS: frozenset[str] = frozenset([".mp3", ".wav", ".ogg", ".m4a", ".flac"])


def _version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(f"GRC {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Show application version and exit.",
        ),
    ] = None,
) -> None:
    """GRC CLI: Turn GitHub repositories into narrated demo videos."""


class _StageProgressTracker:
    """Tracks stage progress, timing each stage and reporting to standard error."""

    def __init__(self) -> None:
        self.current_stage: str | None = None
        self.stage_start: float | None = None

    def on_stage(self, name: str, number: int, total: int) -> None:
        """Callback invoked at the start of each pipeline stage."""
        now = time.monotonic()
        if self.stage_start is not None and self.current_stage is not None:
            elapsed = now - self.stage_start
            typer.echo(f"Done in {elapsed:.1f}s", err=True)
        self.current_stage = name
        self.stage_start = now
        desc = STAGE_DESCRIPTIONS.get(name, name)
        typer.echo(f"[{number}/{total}] {desc}...", err=True)

    def on_scene(self, current: int, total: int, scene_type: str) -> None:
        """Callback invoked as each individual scene finishes visual rendering."""
        logger.info("Scene %d/%d (%s) visuals ready", current, total, scene_type)

    def finish(self) -> None:
        """Report completion timing for the final stage."""
        if self.stage_start is not None and self.current_stage is not None:
            elapsed = time.monotonic() - self.stage_start
            typer.echo(f"Done in {elapsed:.1f}s", err=True)
            self.current_stage = None
            self.stage_start = None


def _print_summary(
    result: PipelineResult,
    total_time: float,
    *,
    no_media: bool = False,
) -> None:
    """Print a structured execution summary to standard error."""
    lines = ["\nSummary:"]
    if result.scene_counts:
        counts_str = ", ".join(
            f"{count}x {stype}" for stype, count in sorted(result.scene_counts.items())
        )
        lines.append(f"  Scenes:   {counts_str}")

    if result.fallbacks:
        lines.append("  Fallbacks:")
        for f in result.fallbacks:
            lines.append(f"    - Scene {f['scene_id']} ({f['type']}): {f['reason']}")

    if result.video is not None and result.video.exists():
        size_mb = result.video.stat().st_size / (1024 * 1024)
        dur = f"{result.duration_seconds:.1f}s" if result.duration_seconds is not None else "N/A"
        lines.append(f"  Video:    {result.video}")
        lines.append(f"  Duration: {dur}")
        lines.append(f"  Size:     {size_mb:.2f} MB")
        if result.captions is not None and result.captions.exists():
            lines.append(f"  Captions: {result.captions}")
    else:
        lines.append(f"  Script:   {result.scenes_json}")
    lines.append(f"  Total:    {total_time:.2f}s")
    typer.echo("\n".join(lines), err=True)

    if result.fallbacks:
        typer.secho(
            "Some scenes fell back to plain static slides:", fg=typer.colors.YELLOW, err=True
        )
        for f in result.fallbacks:
            typer.secho(
                f"  Scene {f['scene_id']} ({f['type']}): {f['reason']}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    if result.media_count == 0 and not no_media:
        typer.secho(
            "This repository has no usable screenshots. Try a repository with images in its README or a homepage.",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _validate_options(
    theme: str,
    accent: str,
    provider: str | None,
    voice_speed: float | None,
    captions: str = "srt",
    music: Path | None = None,
    anim_fps: int | None = None,
) -> tuple[str, str, str | None, float | None, str, Path | None, int | None]:
    """Validate CLI option values and return normalized forms.

    Raises:
        typer.BadParameter: If any option value is invalid.
    """
    clean_theme = theme.strip().lower()
    if clean_theme not in ("dark", "light"):
        raise typer.BadParameter(f"Unknown theme '{theme}'. Must be 'dark' or 'light'.")

    clean_accent = accent.strip()
    if not HEX_COLOR_PATTERN.match(clean_accent):
        raise typer.BadParameter(
            f"Invalid accent color '{accent}'. Must be a hex code (e.g. #6366f1)."
        )

    clean_provider: str | None = None
    if provider is not None:
        clean_provider = provider.strip().lower()
        if clean_provider not in ("ollama", "openai"):
            raise typer.BadParameter(
                f"Unknown provider '{provider}'. Must be 'ollama' or 'openai'."
            )

    clean_speed: float | None = None
    if voice_speed is not None:
        if not (0.5 <= voice_speed <= 1.5):
            raise typer.BadParameter(f"voice_speed must be between 0.5 and 1.5, got {voice_speed}.")
        clean_speed = voice_speed

    clean_captions = captions.strip().lower()
    if clean_captions not in ("off", "srt", "burn"):
        raise typer.BadParameter(
            f"Unknown captions mode '{captions}'. Must be 'off', 'srt', or 'burn'."
        )

    clean_music: Path | None = None
    if music is not None:
        clean_music = Path(music).resolve()
        if not clean_music.is_file():
            raise typer.BadParameter(f"Music file not found at '{music}'.")
        if clean_music.suffix.lower() not in ALLOWED_MUSIC_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_MUSIC_EXTENSIONS))
            raise typer.BadParameter(
                f"Unsupported music file extension '{clean_music.suffix}'. Allowed: {allowed}."
            )
        typer.echo("Make sure you have the right to use this music.", err=True)

    clean_anim_fps: int | None = None
    if anim_fps is not None:
        if not (10 <= anim_fps <= 30):
            raise typer.BadParameter(f"anim-fps must be between 10 and 30, got {anim_fps}.")
        clean_anim_fps = anim_fps

    return (
        clean_theme,
        clean_accent,
        clean_provider,
        clean_speed,
        clean_captions,
        clean_music,
        clean_anim_fps,
    )


def _apply_settings_overrides(
    settings: Settings,
    provider: str | None = None,
    model: str | None = None,
    voice_speed: float | None = None,
    anim_fps: int | None = None,
    fast: bool = False,
) -> Settings:
    """Create a new Settings instance with overridden fields using dataclasses.replace."""
    overrides: dict[str, Any] = {}
    if provider is not None:
        overrides["llm_provider"] = provider
    if model is not None:
        overrides["llm_model"] = model.strip()
    if voice_speed is not None:
        overrides["voice_speed"] = voice_speed
    if anim_fps is not None:
        overrides["anim_fps"] = anim_fps
    if fast:
        overrides["anim_quality"] = "fast"
        if anim_fps is None:
            overrides["anim_fps"] = 15

    if overrides:
        return replace(settings, **overrides)
    return settings


@app.command()
def make(
    url: Annotated[
        str,
        typer.Argument(
            help="Public GitHub repository URL (e.g. https://github.com/owner/repo).",
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Custom destination path for the rendered MP4 video.",
        ),
    ] = None,
    theme: Annotated[
        str,
        typer.Option(
            "--theme",
            help="Slide visual theme ('dark' or 'light').",
        ),
    ] = "dark",
    accent: Annotated[
        str,
        typer.Option(
            "--accent",
            help="Hex color code for accent highlights.",
        ),
    ] = "#6366f1",
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="LLM provider ('ollama' or 'openai').",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="LLM model name (e.g. 'qwen2.5:7b' or 'gpt-4o-mini').",
        ),
    ] = None,
    voice_speed: Annotated[
        float | None,
        typer.Option(
            "--voice-speed",
            help="Piper voice narration speed factor (0.5 to 1.5).",
        ),
    ] = None,
    script_only: Annotated[
        bool,
        typer.Option(
            "--script-only",
            help="Stop after script generation without rendering media.",
        ),
    ] = False,
    reuse: Annotated[
        bool,
        typer.Option(
            "--reuse",
            help="Reuse existing repo.json and scenes.json if available.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force re-execution of all stages, clearing caches.",
        ),
    ] = False,
    keep_temp: Annotated[
        bool,
        typer.Option(
            "--keep-temp",
            help="Retain intermediate rendering artifacts.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Enable verbose debug logging.",
        ),
    ] = False,
    no_media: Annotated[
        bool,
        typer.Option(
            "--no-media",
            help="Skip downloading and displaying repository screenshots and clips.",
        ),
    ] = False,
    no_animation: Annotated[
        bool,
        typer.Option(
            "--no-animation",
            help="Render animated scenes as static snapshots instead of video clips.",
        ),
    ] = False,
    no_transitions: Annotated[
        bool,
        typer.Option(
            "--no-transitions",
            help="Disable crossfade transitions between scenes.",
        ),
    ] = False,
    captions: Annotated[
        str,
        typer.Option(
            "--captions",
            help="Subtitle caption mode ('off', 'srt', or 'burn').",
        ),
    ] = "srt",
    music: Annotated[
        Path | None,
        typer.Option(
            "--music",
            help="Path to background music audio file.",
        ),
    ] = None,
    fast: Annotated[
        bool,
        typer.Option(
            "--fast",
            help="Render animation faster at 15 fps and encode with veryfast preset.",
        ),
    ] = False,
    anim_fps: Annotated[
        int | None,
        typer.Option(
            "--anim-fps",
            help="Animation rendering frame rate (10 to 30).",
        ),
    ] = None,
) -> None:
    """Generate a narrated demo video from a public GitHub repository URL."""
    setup_logging(verbose=verbose)

    (
        clean_theme,
        clean_accent,
        clean_provider,
        clean_speed,
        clean_captions,
        clean_music,
        clean_anim_fps,
    ) = _validate_options(
        theme=theme,
        accent=accent,
        provider=provider,
        voice_speed=voice_speed,
        captions=captions,
        music=music,
        anim_fps=anim_fps,
    )

    base_settings = get_settings()
    settings = _apply_settings_overrides(
        base_settings,
        provider=clean_provider,
        model=model,
        voice_speed=clean_speed,
        anim_fps=clean_anim_fps,
        fast=fast,
    )

    tracker = _StageProgressTracker()
    pipeline_options = PipelineOptions(
        theme=clean_theme,
        accent=clean_accent,
        script_only=script_only,
        reuse=reuse,
        force=force,
        keep_temp=keep_temp,
        no_media=no_media,
        no_animation=no_animation,
        no_transitions=no_transitions,
        captions=clean_captions,
        music=clean_music,
        fast=fast,
        anim_fps=clean_anim_fps,
        on_stage=tracker.on_stage,
        on_scene=tracker.on_scene,
    )

    t_start = time.monotonic()
    try:
        result = run_pipeline(
            url=url,
            out=out,
            settings=settings,
            options=pipeline_options,
        )
        tracker.finish()
        total_time = time.monotonic() - t_start
        _print_summary(result, total_time, no_media=no_media)
        final_path = result.video if result.video is not None else result.scenes_json
        typer.echo(str(final_path))
    except Repo2VideoError as exc:
        if verbose:
            logger.exception("Pipeline failed with expected error:")
        typer.secho(f"Error: {exc.message}", fg=typer.colors.RED, err=True)
        if exc.hint:
            typer.secho(f"Hint: {exc.hint}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.echo("Cancelled.", err=True)
        if out is not None and out.is_file() and out.stat().st_size == 0:
            out.unlink(missing_ok=True)
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as exc:
        crash_log = settings.output_dir / "crash.log"
        try:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            tb = traceback.format_exc()
            crash_log.write_text(tb, encoding="utf-8")
        except OSError:
            pass
        typer.secho(f"Unexpected error: {exc}", fg=typer.colors.RED, err=True)
        typer.secho(
            f"Traceback written to {crash_log}. Please open an issue.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1) from exc


@app.command()
def render(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to an existing scenes.json file or its work folder.",
        ),
    ],
    label: Annotated[
        str | None,
        typer.Option(
            "--label",
            help="Repository label formatted as 'owner/name'.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Custom destination path for the rendered MP4 video.",
        ),
    ] = None,
    theme: Annotated[
        str,
        typer.Option(
            "--theme",
            help="Slide visual theme ('dark' or 'light').",
        ),
    ] = "dark",
    accent: Annotated[
        str,
        typer.Option(
            "--accent",
            help="Hex color code for accent highlights.",
        ),
    ] = "#6366f1",
    voice_speed: Annotated[
        float | None,
        typer.Option(
            "--voice-speed",
            help="Piper voice narration speed factor (0.5 to 1.5).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force re-rendering all slides and audio.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Enable verbose debug logging.",
        ),
    ] = False,
    no_media: Annotated[
        bool,
        typer.Option(
            "--no-media",
            help="Skip downloading and displaying repository screenshots and clips.",
        ),
    ] = False,
    no_animation: Annotated[
        bool,
        typer.Option(
            "--no-animation",
            help="Render animated scenes as static snapshots instead of video clips.",
        ),
    ] = False,
    no_transitions: Annotated[
        bool,
        typer.Option(
            "--no-transitions",
            help="Disable crossfade transitions between scenes.",
        ),
    ] = False,
    captions: Annotated[
        str,
        typer.Option(
            "--captions",
            help="Subtitle caption mode ('off', 'srt', or 'burn').",
        ),
    ] = "srt",
    music: Annotated[
        Path | None,
        typer.Option(
            "--music",
            help="Path to background music audio file.",
        ),
    ] = None,
    fast: Annotated[
        bool,
        typer.Option(
            "--fast",
            help="Render animation faster at 15 fps and encode with veryfast preset.",
        ),
    ] = False,
    anim_fps: Annotated[
        int | None,
        typer.Option(
            "--anim-fps",
            help="Animation rendering frame rate (10 to 30).",
        ),
    ] = None,
) -> None:
    """Render slides, voice, and video from an existing or hand-edited scenes.json script."""
    setup_logging(verbose=verbose)

    (
        clean_theme,
        clean_accent,
        _,
        clean_speed,
        clean_captions,
        clean_music,
        clean_anim_fps,
    ) = _validate_options(
        theme=theme,
        accent=accent,
        provider=None,
        voice_speed=voice_speed,
        captions=captions,
        music=music,
        anim_fps=anim_fps,
    )

    base_settings = get_settings()
    settings = _apply_settings_overrides(
        base_settings,
        voice_speed=clean_speed,
        anim_fps=clean_anim_fps,
        fast=fast,
    )

    resolved_path = path.resolve()
    scenes_file = resolved_path / "scenes.json" if resolved_path.is_dir() else resolved_path
    work_dir = scenes_file.parent

    repo_label: str = label.strip() if label else ""
    if not repo_label:
        repo_json = work_dir / "repo.json"
        if repo_json.is_file():
            try:
                repo_data = RepoData.load(repo_json)
                repo_label = f"{repo_data.owner}/{repo_data.name}"
            except (Repo2VideoError, ValueError, OSError):
                repo_label = ""
        if not repo_label and scenes_file.is_file():
            try:
                script_data = ScriptData.load(scenes_file)
                repo_label = script_data.title
            except (Repo2VideoError, ValueError, OSError):
                repo_label = work_dir.name
        if not repo_label:
            repo_label = work_dir.name

    tracker = _StageProgressTracker()
    pipeline_options = PipelineOptions(
        theme=clean_theme,
        accent=clean_accent,
        script_only=False,
        reuse=False,
        force=force,
        keep_temp=False,
        no_media=no_media,
        no_animation=no_animation,
        no_transitions=no_transitions,
        captions=clean_captions,
        music=clean_music,
        fast=fast,
        anim_fps=clean_anim_fps,
        on_stage=tracker.on_stage,
        on_scene=tracker.on_scene,
    )

    t_start = time.monotonic()
    try:
        result = render_only(
            scenes_json=scenes_file,
            repo_label=repo_label,
            out=out,
            settings=settings,
            options=pipeline_options,
        )
        tracker.finish()
        total_time = time.monotonic() - t_start
        _print_summary(result, total_time, no_media=no_media)
        typer.echo(str(result.video))
    except Repo2VideoError as exc:
        if verbose:
            logger.exception("Pipeline failed with expected error:")
        typer.secho(f"Error: {exc.message}", fg=typer.colors.RED, err=True)
        if exc.hint:
            typer.secho(f"Hint: {exc.hint}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.echo("Cancelled.", err=True)
        if out is not None and out.is_file() and out.stat().st_size == 0:
            out.unlink(missing_ok=True)
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as exc:
        crash_log = settings.output_dir / "crash.log"
        try:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            tb = traceback.format_exc()
            crash_log.write_text(tb, encoding="utf-8")
        except OSError:
            pass
        typer.secho(f"Unexpected error: {exc}", fg=typer.colors.RED, err=True)
        typer.secho(
            f"Traceback written to {crash_log}. Please open an issue.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1) from exc


@app.command()
def doctor(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Enable verbose debug logging.",
        ),
    ] = False,
) -> None:
    """Print a checklist of required local dependencies and their readiness status."""
    setup_logging(verbose=verbose)
    settings = get_settings()
    checks = check_doctor(settings)

    all_ok = True
    for c in checks:
        if c.name == "GITHUB_TOKEN":
            typer.echo(f"GITHUB_TOKEN: {c.status}")
        elif c.status == "OK":
            typer.secho(f"{c.name}: OK", fg=typer.colors.GREEN)
        elif c.status == "WARN":
            msg = f"{c.name}: WARN"
            if c.hint:
                msg += f" - Hint: {c.hint}"
            typer.secho(msg, fg=typer.colors.YELLOW)
        else:
            if c.required:
                all_ok = False
            msg = f"{c.name}: FAIL"
            if c.hint:
                msg += f" - Hint: {c.hint}"
            typer.secho(msg, fg=typer.colors.RED)

    if not all_ok:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)
