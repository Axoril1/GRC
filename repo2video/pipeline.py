"""Pipeline orchestrator that executes stages sequentially from URL analysis to final video rendering."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repo2video import __version__
from repo2video.analyzer import analyze, parse_repo_url
from repo2video.config import Settings, get_settings
from repo2video.errors import Repo2VideoError
from repo2video.llm import get_client
from repo2video.models import RepoData, ScriptData
from repo2video.render import SceneAssets, check_ffmpeg, probe_video, render_video
from repo2video.scenes import check_chromium, render_scenes
from repo2video.scriptwriter import write_script
from repo2video.voice import PiperEngine, synthesize_all

logger = logging.getLogger(__name__)


def _noop_stage(name: str, number: int, total: int) -> None:
    """Default no-op progress callback."""


@dataclass
class PipelineOptions:
    """Options governing pipeline execution behavior.

    Attributes:
        theme: Slide visual theme ('dark' or 'light').
        accent: Hex color code for accent highlights.
        script_only: Stop after script generation without rendering media.
        reuse: Reuse existing repo.json and scenes.json if present.
        force: Force re-execution of all stages, clearing intermediate caches.
        keep_temp: Retain intermediate rendering files.
        on_stage: Progress callback invoked at the start of each stage.
    """

    theme: str = "dark"
    accent: str = "#6366f1"
    script_only: bool = False
    reuse: bool = False
    force: bool = False
    keep_temp: bool = False
    on_stage: Callable[[str, int, int], None] = field(default=_noop_stage)


@dataclass
class PipelineResult:
    """Output artifacts and metrics produced by the pipeline.

    Attributes:
        work_dir: Directory containing intermediate and output files.
        repo_json: Path to the generated or reused repo.json file.
        scenes_json: Path to the generated or reused scenes.json file.
        video: Path to the finished MP4 video (None if script_only).
        captions: Path to the companion SRT subtitle file (None if script_only).
        duration_seconds: Total duration of the rendered video in seconds.
        timings: Execution time in seconds for each executed stage.
    """

    work_dir: Path
    repo_json: Path
    scenes_json: Path
    video: Path | None
    captions: Path | None
    duration_seconds: float | None
    timings: dict[str, float]


@dataclass
class DoctorCheck:
    """Result of an individual environment dependency check.

    Attributes:
        name: Name of the checked component.
        ok: Whether the check passed.
        status: Short status label ('OK', 'FAIL', 'set', 'not set').
        hint: Optional guidance on how to fix a failing check.
        required: Whether the component is mandatory for video generation.
    """

    name: str
    ok: bool
    status: str
    hint: str | None = None
    required: bool = True


def _sanitize_name(name: str) -> str:
    """Sanitize a repository owner or name for use in a directory path.

    Replaces characters other than letters, digits, hyphen, underscore, and dot
    with an underscore, and converts to lowercase.

    Args:
        name: Raw identifier string.

    Returns:
        Sanitized lowercase string.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name).lower()


def _run_preflight(
    settings: Settings,
    *,
    script_only: bool,
    check_llm: bool = True,
) -> None:
    """Run preflight health checks for LLM, Piper, and FFmpeg before pipeline execution.

    Collects all failures together and raises a single comprehensive Repo2VideoError
    if any check fails.

    Args:
        settings: Application settings.
        script_only: If True, only LLM readiness is validated.
        check_llm: If True, verifies LLM server and model readiness.

    Raises:
        Repo2VideoError: If any of the preflight checks fail.
    """
    failures: list[tuple[str, str | None]] = []

    if check_llm:
        try:
            client = get_client(settings)
            client.check_ready()
        except Exception as exc:  # noqa: BLE001
            msg = getattr(exc, "message", str(exc))
            hint = getattr(exc, "hint", None)
            failures.append((msg, hint))

    if not script_only:
        try:
            piper = PiperEngine(settings=settings)
            piper.check_ready()
        except Exception as exc:  # noqa: BLE001
            msg = getattr(exc, "message", str(exc))
            hint = getattr(exc, "hint", None)
            failures.append((msg, hint))

        try:
            check_ffmpeg()
        except Exception as exc:  # noqa: BLE001
            msg = getattr(exc, "message", str(exc))
            hint = getattr(exc, "hint", None)
            failures.append((msg, hint))

    if failures:
        lines: list[str] = [f"Preflight checks failed with {len(failures)} error(s):"]
        hints: list[str] = []
        for i, (prob, hint) in enumerate(failures, start=1):
            lines.append(f"{i}. {prob}")
            if hint:
                lines.append(f"   Hint: {hint}")
                hints.append(hint)
        combined_message = "\n".join(lines)
        combined_hint = " | ".join(hints) if hints else "Resolve preflight issues."
        raise Repo2VideoError(combined_message, hint=combined_hint)


def check_doctor(settings: Settings | None = None) -> list[DoctorCheck]:
    """Inspect environment prerequisites and return a list of DoctorCheck results.

    Checks:
        1. Python version (>= 3.11)
        2. FFmpeg and required encoders (libx264, aac)
        3. Playwright Chromium
        4. Piper TTS and voice model file
        5. LLM server and model readiness
        6. GITHUB_TOKEN environment variable status (set / not set)

    Args:
        settings: Application settings. Defaults to get_settings().

    Returns:
        List of DoctorCheck objects.
    """
    resolved_settings = settings if settings is not None else get_settings()
    checks: list[DoctorCheck] = []

    # 1. Python version
    py_ok = sys.version_info >= (3, 11)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(
        DoctorCheck(
            name=f"Python version ({py_ver})",
            ok=py_ok,
            status="OK" if py_ok else "FAIL",
            hint=None if py_ok else "Python 3.11 or newer is required.",
            required=True,
        )
    )

    # 2. FFmpeg and encoders
    try:
        check_ffmpeg()
        checks.append(
            DoctorCheck(
                name="FFmpeg and encoders",
                ok=True,
                status="OK",
                required=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        hint = getattr(exc, "hint", "Install FFmpeg with libx264 and aac encoders.")
        checks.append(
            DoctorCheck(
                name="FFmpeg and encoders",
                ok=False,
                status="FAIL",
                hint=hint,
                required=True,
            )
        )

    # 3. Playwright Chromium
    try:
        check_chromium()
        checks.append(
            DoctorCheck(
                name="Playwright Chromium",
                ok=True,
                status="OK",
                required=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        hint = getattr(exc, "hint", "Run: playwright install chromium")
        checks.append(
            DoctorCheck(
                name="Playwright Chromium",
                ok=False,
                status="FAIL",
                hint=hint,
                required=True,
            )
        )

    # 4. Piper and voice file
    try:
        piper = PiperEngine(settings=resolved_settings)
        piper.check_ready()
        checks.append(
            DoctorCheck(
                name="Piper and voice file",
                ok=True,
                status="OK",
                required=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        hint = getattr(exc, "hint", "Ensure piper-tts is installed and voice model exists.")
        checks.append(
            DoctorCheck(
                name="Piper and voice file",
                ok=False,
                status="FAIL",
                hint=hint,
                required=True,
            )
        )

    # 5. LLM server and model
    provider = resolved_settings.llm_provider
    model = resolved_settings.llm_model
    try:
        client = get_client(resolved_settings)
        client.check_ready()
        checks.append(
            DoctorCheck(
                name=f"LLM server and model ({provider}: {model})",
                ok=True,
                status="OK",
                required=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        hint = getattr(
            exc,
            "hint",
            f"Ensure {provider} is running and model '{model}' is installed.",
        )
        checks.append(
            DoctorCheck(
                name=f"LLM server and model ({provider}: {model})",
                ok=False,
                status="FAIL",
                hint=hint,
                required=True,
            )
        )

    # 6. GITHUB_TOKEN
    token_val = os.environ.get("GITHUB_TOKEN") or resolved_settings.github_token
    token_set = bool(token_val and token_val.strip())
    checks.append(
        DoctorCheck(
            name="GITHUB_TOKEN",
            ok=True,
            status="set" if token_set else "not set",
            hint=None
            if token_set
            else "Set GITHUB_TOKEN to raise API rate limit from 60 to 5000/hr.",
            required=False,
        )
    )

    return checks


def _pair_scene_assets(
    script_data: ScriptData,
    slide_paths: list[Path],
    audios: list[Any],
) -> list[SceneAssets]:
    """Pair slide PNG images with corresponding narration audio into SceneAssets.

    Args:
        script_data: Validated ScriptData instance.
        slide_paths: List of rendered slide PNG file paths.
        audios: List of SceneAudio objects from the voice stage.

    Returns:
        List of paired SceneAssets in scene order.

    Raises:
        Repo2VideoError: If counts or scene IDs mismatch.
    """
    if len(slide_paths) != len(audios) or len(slide_paths) != len(script_data.scenes):
        raise Repo2VideoError(
            f"Mismatch in scene assets count: {len(slide_paths)} slides, "
            f"{len(audios)} audio tracks, {len(script_data.scenes)} scenes.",
            hint="Rerun with --force to regenerate all slides and audio.",
        )

    slide_by_id: dict[int, Path] = {}
    for p in slide_paths:
        m = re.search(r"scene_(\d+)", p.stem)
        if m:
            slide_by_id[int(m.group(1))] = p
        else:
            raise Repo2VideoError(
                f"Cannot determine scene id from slide filename: {p.name}.",
                hint="Expected filename format: scene_XX.png.",
            )

    audio_by_id = {a.scene_id: a for a in audios}

    scene_assets: list[SceneAssets] = []
    for sc in script_data.scenes:
        if sc.id not in slide_by_id or sc.id not in audio_by_id:
            raise Repo2VideoError(
                f"Mismatch in scene assets: missing slide or audio for scene {sc.id}.",
                hint="Rerun with --force to regenerate all intermediate media.",
            )
        scene_assets.append(
            SceneAssets(
                scene_id=sc.id,
                image=slide_by_id[sc.id],
                audio=audio_by_id[sc.id].path,
                duration_seconds=audio_by_id[sc.id].duration_seconds,
                narration=sc.narration,
            )
        )

    return scene_assets


def _write_run_json(
    path: Path,
    *,
    url: str,
    settings: Settings,
    options: PipelineOptions,
    script_data: ScriptData | None,
    timings: dict[str, float],
    repo_json: Path,
    scenes_json: Path,
    video_path: Path | None,
    captions_path: Path | None,
) -> None:
    """Write run.json metadata file containing execution metrics without secrets.

    Args:
        path: Path to the target run.json file.
        url: Analyzed repository URL.
        settings: Application settings.
        options: Pipeline options.
        script_data: ScriptData instance if generated.
        timings: Dictionary mapping stage names to seconds.
        repo_json: Path to repo.json.
        scenes_json: Path to scenes.json.
        video_path: Path to the final video file, or None.
        captions_path: Path to the final captions file, or None.
    """
    scene_count = len(script_data.scenes) if script_data is not None else 0
    total_words = (
        sum(len(s.narration.split()) for s in script_data.scenes) if script_data is not None else 0
    )

    outputs: dict[str, str | None] = {
        "repo_json": repo_json.name if repo_json.exists() else None,
        "scenes_json": scenes_json.name if scenes_json.exists() else None,
        "video": video_path.name if video_path is not None and video_path.exists() else None,
        "captions": (
            captions_path.name if captions_path is not None and captions_path.exists() else None
        ),
    }

    meta: dict[str, Any] = {
        "tool_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "url": url,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "voice_model": settings.piper_voice_path.name,
        "theme": options.theme,
        "scene_count": scene_count,
        "total_narration_words": total_words,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "outputs": outputs,
    }

    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run(
    url: str,
    out: Path | None = None,
    settings: Settings | None = None,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    """Execute the full video generation pipeline from GitHub URL to finished MP4.

    Args:
        url: Public GitHub repository URL.
        out: Optional destination path for the rendered MP4 file.
        settings: Application settings. Defaults to get_settings().
        options: Pipeline execution options. Defaults to PipelineOptions().

    Returns:
        PipelineResult containing paths to generated assets, timings, and metrics.

    Raises:
        Repo2VideoError: If preflight checks or any pipeline stage fails.
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_options = options if options is not None else PipelineOptions()

    owner, repo_name = parse_repo_url(url)
    folder_name = f"{_sanitize_name(owner)}__{_sanitize_name(repo_name)}"
    work_dir = resolved_settings.output_dir / folder_name
    work_dir.mkdir(parents=True, exist_ok=True)

    repo_json = work_dir / "repo.json"
    scenes_json = work_dir / "scenes.json"
    slides_dir = work_dir / "slides"
    audio_dir = work_dir / "audio"
    debug_dir = work_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    run_json = work_dir / "run.json"

    # Preflight check before starting any stage
    _run_preflight(resolved_settings, script_only=resolved_options.script_only)

    # Force flag: clean intermediate media directories and ignore cache
    if resolved_options.force:
        shutil.rmtree(slides_dir, ignore_errors=True)
        shutil.rmtree(audio_dir, ignore_errors=True)

    stage_names = (
        ["analyze", "script"]
        if resolved_options.script_only
        else ["analyze", "script", "slides", "voice", "render"]
    )
    total_stages = len(stage_names)
    timings: dict[str, float] = {}

    # Stage 1: analyze
    resolved_options.on_stage("analyze", 1, total_stages)
    repo_data: RepoData
    if resolved_options.reuse and not resolved_options.force and repo_json.is_file():
        try:
            repo_data = RepoData.load(repo_json)
            logger.info("Reusing existing repo.json from %s", repo_json)
            timings["analyze"] = 0.0
        except (Repo2VideoError, ValueError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load existing repo.json (%s); re-analyzing.", exc)
            t0 = time.monotonic()
            repo_data = analyze(url, settings=resolved_settings)
            repo_data.save(repo_json)
            timings["analyze"] = time.monotonic() - t0
    else:
        t0 = time.monotonic()
        repo_data = analyze(url, settings=resolved_settings)
        repo_data.save(repo_json)
        timings["analyze"] = time.monotonic() - t0

    repo_label = f"{repo_data.owner}/{repo_data.name}"

    # Stage 2: script
    resolved_options.on_stage("script", 2, total_stages)
    script_data: ScriptData
    if resolved_options.reuse and not resolved_options.force and scenes_json.is_file():
        try:
            script_data = ScriptData.load(scenes_json)
            logger.info("Reusing existing scenes.json from %s", scenes_json)
            timings["script"] = 0.0
        except (Repo2VideoError, ValueError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load existing scenes.json (%s); re-generating.", exc)
            t0 = time.monotonic()
            client = get_client(resolved_settings)
            script_data = write_script(repo_data, client, settings=resolved_settings)
            script_data.save(scenes_json)
            timings["script"] = time.monotonic() - t0
    else:
        t0 = time.monotonic()
        client = get_client(resolved_settings)
        script_data = write_script(repo_data, client, settings=resolved_settings)
        script_data.save(scenes_json)
        timings["script"] = time.monotonic() - t0

    if resolved_options.script_only:
        _write_run_json(
            run_json,
            url=url,
            settings=resolved_settings,
            options=resolved_options,
            script_data=script_data,
            timings=timings,
            repo_json=repo_json,
            scenes_json=scenes_json,
            video_path=None,
            captions_path=None,
        )
        return PipelineResult(
            work_dir=work_dir,
            repo_json=repo_json,
            scenes_json=scenes_json,
            video=None,
            captions=None,
            duration_seconds=None,
            timings=timings,
        )

    # Stage 3: slides
    resolved_options.on_stage("slides", 3, total_stages)
    slides_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    slide_paths = render_scenes(
        script_data,
        slides_dir,
        repo_label=repo_label,
        theme=resolved_options.theme,
        accent=resolved_options.accent,
        settings=resolved_settings,
    )
    timings["slides"] = time.monotonic() - t0

    # Stage 4: voice
    resolved_options.on_stage("voice", 4, total_stages)
    audio_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    audios = synthesize_all(
        script_data,
        audio_dir,
        settings=resolved_settings,
    )
    timings["voice"] = time.monotonic() - t0

    # Pair slides and audio into SceneAssets
    scene_assets = _pair_scene_assets(script_data, slide_paths, audios)

    # Stage 5: render
    resolved_options.on_stage("render", 5, total_stages)
    target_video = Path(out).resolve() if out is not None else work_dir / "demo.mp4"
    target_captions = target_video.with_suffix(".srt")

    t0 = time.monotonic()
    render_video(
        scene_assets,
        target_video,
        settings=resolved_settings,
        keep_temp=resolved_options.keep_temp,
    )
    timings["render"] = time.monotonic() - t0

    probe_info = probe_video(target_video)
    duration_seconds = probe_info["duration_seconds"]

    _write_run_json(
        run_json,
        url=url,
        settings=resolved_settings,
        options=resolved_options,
        script_data=script_data,
        timings=timings,
        repo_json=repo_json,
        scenes_json=scenes_json,
        video_path=target_video,
        captions_path=target_captions,
    )

    return PipelineResult(
        work_dir=work_dir,
        repo_json=repo_json,
        scenes_json=scenes_json,
        video=target_video,
        captions=target_captions,
        duration_seconds=duration_seconds,
        timings=timings,
    )


def render_only(
    scenes_json: Path,
    repo_label: str,
    out: Path | None = None,
    settings: Settings | None = None,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    """Render slides, voice, and video starting from an existing scenes.json script file.

    Allows users to edit the script manually before producing media.

    Args:
        scenes_json: Path to the existing scenes.json file (or work directory).
        repo_label: Repository label (e.g. 'owner/name').
        out: Optional destination path for the rendered MP4 video.
        settings: Application settings. Defaults to get_settings().
        options: Pipeline execution options. Defaults to PipelineOptions().

    Returns:
        PipelineResult containing paths to generated assets and metrics.

    Raises:
        Repo2VideoError: If scenes.json is missing or rendering fails.
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_options = options if options is not None else PipelineOptions()

    path = Path(scenes_json).resolve()
    if path.is_dir():
        candidate = path / "scenes.json"
        if candidate.is_file():
            path = candidate
        else:
            raise Repo2VideoError(
                f"Scenes file not found at directory {path}.",
                hint="Directory must contain scenes.json.",
            )

    if not path.is_file():
        raise Repo2VideoError(
            f"Scenes file not found at {path}.",
            hint="Ensure the path points to an existing scenes.json file.",
        )

    script_data = ScriptData.load(path)
    work_dir = path.parent
    repo_json = work_dir / "repo.json"
    slides_dir = work_dir / "slides"
    audio_dir = work_dir / "audio"
    run_json = work_dir / "run.json"

    # Preflight check Piper and FFmpeg (LLM check skipped for render_only)
    _run_preflight(
        resolved_settings,
        script_only=False,
        check_llm=False,
    )

    if resolved_options.force:
        shutil.rmtree(slides_dir, ignore_errors=True)
        shutil.rmtree(audio_dir, ignore_errors=True)

    stage_names = ["slides", "voice", "render"]
    total_stages = len(stage_names)
    timings: dict[str, float] = {}

    # Stage 1 (of 3): slides
    resolved_options.on_stage("slides", 1, total_stages)
    slides_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    slide_paths = render_scenes(
        script_data,
        slides_dir,
        repo_label=repo_label,
        theme=resolved_options.theme,
        accent=resolved_options.accent,
        settings=resolved_settings,
    )
    timings["slides"] = time.monotonic() - t0

    # Stage 2 (of 3): voice
    resolved_options.on_stage("voice", 2, total_stages)
    audio_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    audios = synthesize_all(
        script_data,
        audio_dir,
        settings=resolved_settings,
    )
    timings["voice"] = time.monotonic() - t0

    # Pair assets
    scene_assets = _pair_scene_assets(script_data, slide_paths, audios)

    # Stage 3 (of 3): render
    resolved_options.on_stage("render", 3, total_stages)
    target_video = Path(out).resolve() if out is not None else work_dir / "demo.mp4"
    target_captions = target_video.with_suffix(".srt")

    t0 = time.monotonic()
    render_video(
        scene_assets,
        target_video,
        settings=resolved_settings,
        keep_temp=resolved_options.keep_temp,
    )
    timings["render"] = time.monotonic() - t0

    probe_info = probe_video(target_video)
    duration_seconds = probe_info["duration_seconds"]

    _write_run_json(
        run_json,
        url="",
        settings=resolved_settings,
        options=resolved_options,
        script_data=script_data,
        timings=timings,
        repo_json=repo_json,
        scenes_json=path,
        video_path=target_video,
        captions_path=target_captions,
    )

    return PipelineResult(
        work_dir=work_dir,
        repo_json=repo_json,
        scenes_json=path,
        video=target_video,
        captions=target_captions,
        duration_seconds=duration_seconds,
        timings=timings,
    )
