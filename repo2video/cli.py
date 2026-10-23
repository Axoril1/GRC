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

from repo2video import __version__
from repo2video.config import Settings, get_settings
from repo2video.errors import Repo2VideoError
from repo2video.logging_setup import setup_logging
from repo2video.models import RepoData, ScriptData
from repo2video.pipeline import (
    PipelineOptions,
    PipelineResult,
    check_doctor,
    render_only,
)
from repo2video.pipeline import run as run_pipeline

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="repo2video",
    help="Turn a public GitHub repository into a narrated demo video.",
    no_args_is_help=True,
)

STAGE_DESCRIPTIONS: dict[str, str] = {
    "analyze": "Analyzing repository",
    "script": "Writing the script",
    "slides": "Rendering slide images",
    "voice": "Synthesizing voice narration",
    "render": "Compositing video with FFmpeg",
}

HEX_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(f"repo2video {__version__}")
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
    """repo2video CLI: Turn GitHub repositories into narrated demo videos."""


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

    def finish(self) -> None:
        """Report completion timing for the final stage."""
        if self.stage_start is not None and self.current_stage is not None:
            elapsed = time.monotonic() - self.stage_start
            typer.echo(f"Done in {elapsed:.1f}s", err=True)
            self.current_stage = None
            self.stage_start = None


def _print_summary(result: PipelineResult, total_time: float) -> None:
    """Print a structured execution summary to standard error."""
    lines = ["\nSummary:"]
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


def _validate_options(
    theme: str,
    accent: str,
    provider: str | None,
    voice_speed: float | None,
) -> tuple[str, str, str | None, float | None]:
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

    return clean_theme, clean_accent, clean_provider, clean_speed


def _apply_settings_overrides(
    settings: Settings,
    provider: str | None = None,
    model: str | None = None,
    voice_speed: float | None = None,
) -> Settings:
    """Create a new Settings instance with overridden fields using dataclasses.replace."""
    overrides: dict[str, Any] = {}
    if provider is not None:
        overrides["llm_provider"] = provider
    if model is not None:
        overrides["llm_model"] = model.strip()
    if voice_speed is not None:
        overrides["voice_speed"] = voice_speed

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
            help="LLM model name (e.g. 'mistral:latest' or 'gpt-4o-mini').",
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
) -> None:
    """Generate a narrated demo video from a public GitHub repository URL."""
    setup_logging(verbose=verbose)

    clean_theme, clean_accent, clean_provider, clean_speed = _validate_options(
        theme=theme,
        accent=accent,
        provider=provider,
        voice_speed=voice_speed,
    )

    base_settings = get_settings()
    settings = _apply_settings_overrides(
        base_settings,
        provider=clean_provider,
        model=model,
        voice_speed=clean_speed,
    )

    tracker = _StageProgressTracker()
    pipeline_options = PipelineOptions(
        theme=clean_theme,
        accent=clean_accent,
        script_only=script_only,
        reuse=reuse,
        force=force,
        keep_temp=keep_temp,
        on_stage=tracker.on_stage,
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
        _print_summary(result, total_time)
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
) -> None:
    """Render slides, voice, and video from an existing or hand-edited scenes.json script."""
    setup_logging(verbose=verbose)

    clean_theme, clean_accent, _, clean_speed = _validate_options(
        theme=theme,
        accent=accent,
        provider=None,
        voice_speed=voice_speed,
    )

    base_settings = get_settings()
    settings = _apply_settings_overrides(
        base_settings,
        voice_speed=clean_speed,
    )

    # Determine scenes_file and fallback repo_label
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
        on_stage=tracker.on_stage,
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
        _print_summary(result, total_time)
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
        elif c.ok:
            typer.secho(f"{c.name}: OK", fg=typer.colors.GREEN)
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
