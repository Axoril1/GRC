"""Pipeline orchestrator that executes stages sequentially from URL analysis to final video rendering."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from GRC import __version__
from GRC import scenes as _scenes
from GRC.analyzer import analyze, parse_repo_url
from GRC.anim import AnimationRenderer
from GRC.config import Settings, get_settings
from GRC.errors import Repo2VideoError, SceneError
from GRC.llm import get_client
from GRC.media import collect_media, credit_line
from GRC.models import KeyFile, MediaAsset, RepoData, ScriptData, TerminalSession
from GRC.render import (
    RenderOptions,
    SceneAssets,
    SceneVisual,
    capabilities,
    check_ffmpeg,
    probe_video,
    render_video,
    scene_video_lengths,
)
from GRC.scene_arch import build_arch_html
from GRC.scene_code import build_code_html
from GRC.scene_media import build_clip_frame_html, build_media_html, media_frame_rect
from GRC.scene_stats import build_stats_html
from GRC.scene_terminal import build_terminal_html
from GRC.scenes import (
    build_html,
    check_chromium,
    screenshot_html,
)
from GRC.scriptwriter import resolve_references, write_script
from GRC.voice import PiperEngine, SceneAudio, synthesize_all

# Backward compatibility alias for test monkeypatching
render_scenes = _scenes.render_scenes

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
        no_media: Skip collecting media assets from the repository.
        no_animation: Render static snapshots instead of animated video clips.
        no_transitions: Disable crossfade transitions between scenes.
        captions: Subtitle output mode ('off', 'srt', or 'burn').
        music: Optional path to background music audio file.
        fast: Use fast quality presets and frame rates.
        anim_fps: Optional override for animation frame rate (10 to 30).
        on_stage: Progress callback invoked at the start of each stage.
        on_scene: Progress callback invoked as each scene visual finishes rendering.
    """

    theme: str = "dark"
    accent: str = "#6366f1"
    script_only: bool = False
    reuse: bool = False
    force: bool = False
    keep_temp: bool = False
    no_media: bool = False
    no_animation: bool = False
    no_transitions: bool = False
    captions: str = "srt"
    music: Path | None = None
    fast: bool = False
    anim_fps: int | None = None
    on_stage: Callable[[str, int, int], None] = field(default=_noop_stage)
    on_scene: Callable[[int, int, str], None] | None = None


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
        scene_counts: Count of scenes per scene type.
        fallbacks: List of scenes that fell back to static slides with reasons.
        media_count: Number of downloaded media assets used.
        credit_line: Media attribution credit line, if used.
    """

    work_dir: Path
    repo_json: Path
    scenes_json: Path
    video: Path | None
    captions: Path | None
    duration_seconds: float | None
    timings: dict[str, float]
    scene_counts: dict[str, int] = field(default_factory=dict)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    media_count: int = 0
    credit_line: str | None = None


@dataclass
class DoctorCheck:
    """Result of an individual environment dependency check.

    Attributes:
        name: Name of the checked component.
        ok: Whether the check passed.
        status: Short status label ('OK', 'FAIL', 'WARN', 'set', 'not set').
        hint: Optional guidance on how to fix a failing check.
        required: Whether the component is mandatory for video generation.
    """

    name: str
    ok: bool
    status: str
    hint: str | None = None
    required: bool = True


def _sanitize_name(name: str) -> str:
    """Sanitize a repository owner or name for use in a directory path."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name).lower()


def _run_preflight(
    settings: Settings,
    *,
    script_only: bool,
    check_llm: bool = True,
) -> None:
    """Run preflight health checks for LLM, Piper, and FFmpeg before pipeline execution."""
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
        2. Pillow image library
        3. Pygments syntax highlighter
        4. FFmpeg binaries and required encoders (libx264, aac)
        5. FFmpeg filters (overlay, xfade)
        6. Playwright Chromium launches
        7. Piper TTS and voice model file
        8. LLM server and model readiness
        9. GITHUB_TOKEN environment variable status (set / not set)
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

    # 2. Pillow image library
    try:
        import PIL  # noqa: F401

        checks.append(
            DoctorCheck(
                name="Pillow image library",
                ok=True,
                status="OK",
                required=True,
            )
        )
    except ImportError:
        checks.append(
            DoctorCheck(
                name="Pillow image library",
                ok=False,
                status="FAIL",
                hint="Install Pillow: pip install pillow",
                required=True,
            )
        )

    # 3. Pygments syntax highlighter
    try:
        import pygments  # noqa: F401

        checks.append(
            DoctorCheck(
                name="Pygments syntax highlighter",
                ok=True,
                status="OK",
                required=True,
            )
        )
    except ImportError:
        checks.append(
            DoctorCheck(
                name="Pygments syntax highlighter",
                ok=False,
                status="FAIL",
                hint="Install Pygments: pip install pygments",
                required=True,
            )
        )

    # 4. FFmpeg and encoders
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

    # 5. FFmpeg filters
    try:
        cap = capabilities()
        if cap.has_overlay:
            checks.append(
                DoctorCheck(
                    name="FFmpeg overlay filter",
                    ok=True,
                    status="OK",
                    required=True,
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="FFmpeg overlay filter",
                    ok=False,
                    status="FAIL",
                    hint="Install FFmpeg build with overlay filter support.",
                    required=True,
                )
            )

        if cap.has_xfade:
            checks.append(
                DoctorCheck(
                    name="FFmpeg xfade filter",
                    ok=True,
                    status="OK",
                    required=False,
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="FFmpeg xfade filter",
                    ok=True,
                    status="WARN",
                    hint="FFmpeg lacks xfade filter; transitions will degrade to direct cuts.",
                    required=False,
                )
            )
    except Exception as exc:  # noqa: BLE001
        hint = getattr(exc, "hint", str(exc))
        checks.append(
            DoctorCheck(
                name="FFmpeg filters",
                ok=False,
                status="FAIL",
                hint=hint,
                required=True,
            )
        )

    # 6. Playwright Chromium launches
    try:
        check_chromium()
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("about:blank", timeout=20000)
            browser.close()
        checks.append(
            DoctorCheck(
                name="Playwright Chromium launches",
                ok=True,
                status="OK",
                required=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        hint = getattr(exc, "hint", "Run: playwright install chromium")
        checks.append(
            DoctorCheck(
                name="Playwright Chromium launches",
                ok=False,
                status="FAIL",
                hint=hint,
                required=True,
            )
        )

    # 7. Piper and voice file
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

    # 8. LLM server and model
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

    # 9. GITHUB_TOKEN
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
    """Pair slide PNG images with corresponding narration audio into SceneAssets (v1 compatibility)."""
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


def _render_visuals(
    script_data: ScriptData,
    repo_data: RepoData,
    media_assets: list[MediaAsset],
    work_dir: Path,
    audios: list[SceneAudio],
    settings: Settings,
    options: PipelineOptions,
    *,
    render_options: RenderOptions,
    on_scene: Callable[[int, int, str], None] | None = None,
) -> tuple[list[SceneVisual], list[dict[str, Any]], str | None]:
    """Execute the Visuals stage rendering images, clips, and overlays for each scene.

    Computes exact animation lengths from narration lengths and options,
    opens one AnimationRenderer instance, and dispatches per scene type.
    Degrades failing scenes into plain static slides with warnings.

    Returns:
        Tuple of (scene_visuals, fallbacks, credit_line).
    """
    audio_by_id = {a.scene_id: a for a in audios}
    ordered_audios = [audio_by_id[s.id] for s in script_data.scenes]
    narration_seconds = [a.duration_seconds for a in ordered_audios]

    video_lens = scene_video_lengths(narration_seconds, settings, render_options)

    media_by_id: dict[str, MediaAsset] = {a.id: a for a in media_assets}
    key_file_by_id: dict[str, KeyFile] = {f.id: f for f in repo_data.key_files}
    terminal_by_id: dict[str, TerminalSession] = {t.id: t for t in repo_data.terminals}

    repo_label = (
        f"{repo_data.owner}/{repo_data.name}"
        if repo_data.owner and repo_data.name
        else repo_data.name
    )

    # Credit line handling
    used_media_ids = {s.media_id for s in script_data.scenes if s.type == "media" and s.media_id}
    used_media = [media_by_id[m_id] for m_id in used_media_ids if m_id in media_by_id]
    c_line = credit_line(used_media, repo_label) if used_media else None

    if c_line:
        for s in script_data.scenes:
            if s.type == "outro" and len(s.bullets) < 4:
                if c_line not in s.bullets:
                    s.bullets.append(c_line)
                break

    slides_dir = work_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    fallbacks: list[dict[str, Any]] = []
    scene_visuals: list[SceneVisual] = []
    total_scenes = len(script_data.scenes)

    with AnimationRenderer(settings=settings) as renderer:
        for idx, scene in enumerate(script_data.scenes):
            v_len = video_lens[idx]
            scene_visual: SceneVisual | None = None

            # 1. Plain static slide types
            if scene.type in ("title", "problem", "feature", "howto", "outro"):
                slide_png = slides_dir / f"scene_{scene.id:02d}.png"
                sidecar = slides_dir / f"scene_{scene.id:02d}.png.sha256"
                html_doc = build_html(
                    scene,
                    index=idx + 1,
                    total=total_scenes,
                    repo_label=repo_label,
                    theme=options.theme,
                    accent=options.accent,
                )
                html_hash = hashlib.sha256(html_doc.encode("utf-8")).hexdigest()
                if not options.force and slide_png.is_file() and sidecar.is_file():
                    try:
                        if sidecar.read_text(encoding="utf-8").strip() == html_hash:
                            scene_visual = SceneVisual(kind="image", image=slide_png)
                    except OSError:
                        pass

                if scene_visual is None:
                    screenshot_html(
                        html_doc,
                        slide_png,
                        settings=settings,
                        page=getattr(renderer, "_page", None),
                    )
                    sidecar.write_text(html_hash, encoding="utf-8")
                    scene_visual = SceneVisual(kind="image", image=slide_png)

            # 2. Media scenes
            elif scene.type == "media":
                try:
                    if not scene.media_id or scene.media_id not in media_by_id:
                        raise SceneError(f"Media asset '{scene.media_id}' not found.")
                    asset = media_by_id[scene.media_id]
                    asset_path = Path(asset.path)
                    if not asset_path.is_file():
                        raise SceneError(f"Media file not found at {asset_path}.")

                    if asset.kind == "image":
                        html_doc = build_media_html(
                            scene,
                            asset,
                            theme=options.theme,
                            accent=options.accent,
                            repo_label=repo_label,
                            index=idx + 1,
                            total=total_scenes,
                            duration=v_len,
                            settings=settings,
                        )
                        if not options.no_animation:
                            out_clip = clips_dir / f"scene_{scene.id:02d}.mp4"
                            renderer.render(html_doc, duration=v_len, out_path=out_clip)
                            scene_visual = SceneVisual(kind="video", video=out_clip)
                        else:
                            t_capture = 0.85 * v_len
                            frame_bytes = renderer.capture_frame(html_doc, t=t_capture)
                            out_png = slides_dir / f"scene_{scene.id:02d}.png"
                            out_png.write_bytes(frame_bytes)
                            scene_visual = SceneVisual(kind="image", image=out_png)

                    elif asset.kind == "clip":
                        frame_html = build_clip_frame_html(
                            scene,
                            asset,
                            theme=options.theme,
                            accent=options.accent,
                            repo_label=repo_label,
                            index=idx + 1,
                            total=total_scenes,
                            duration=v_len,
                            settings=settings,
                        )
                        frame_png = slides_dir / f"scene_{scene.id:02d}_frame.png"
                        screenshot_html(
                            frame_html,
                            frame_png,
                            settings=settings,
                            page=getattr(renderer, "_page", None),
                        )
                        rect = media_frame_rect(settings)

                        if not options.no_animation:
                            scene_visual = SceneVisual(
                                kind="overlay",
                                image=frame_png,
                                video=asset_path,
                                rect=rect,
                            )
                        else:
                            clip_f0 = slides_dir / f"scene_{scene.id:02d}_f0.png"
                            subprocess.run(
                                [
                                    "ffmpeg",
                                    "-y",
                                    "-ss",
                                    "0",
                                    "-i",
                                    str(asset_path),
                                    "-vframes",
                                    "1",
                                    str(clip_f0),
                                ],
                                check=True,
                                capture_output=True,
                            )
                            bg_img = Image.open(frame_png).convert("RGBA")
                            clip_img = Image.open(clip_f0).convert("RGBA")
                            clip_img.thumbnail((rect.w, rect.h), Image.Resampling.LANCZOS)
                            pos_x = rect.x + (rect.w - clip_img.width) // 2
                            pos_y = rect.y + (rect.h - clip_img.height) // 2
                            bg_img.paste(clip_img, (pos_x, pos_y), clip_img)
                            out_png = slides_dir / f"scene_{scene.id:02d}.png"
                            bg_img.convert("RGB").save(out_png)
                            scene_visual = SceneVisual(kind="image", image=out_png)
                    else:
                        raise SceneError(f"Unknown media asset kind: {asset.kind}")

                except Exception as exc:  # noqa: BLE001
                    reason = getattr(exc, "message", str(exc))
                    logger.warning(
                        "Scene %d (%s) failed: %s; falling back to static slide.",
                        scene.id,
                        scene.type,
                        reason,
                    )
                    fallbacks.append(
                        {
                            "scene_id": scene.id,
                            "type": scene.type,
                            "reason": reason,
                        }
                    )
                    try:
                        fallback_html = build_html(
                            scene,
                            index=idx + 1,
                            total=total_scenes,
                            repo_label=repo_label,
                            theme=options.theme,
                            accent=options.accent,
                        )
                        fallback_png = slides_dir / f"scene_{scene.id:02d}.png"
                        screenshot_html(
                            fallback_html,
                            fallback_png,
                            settings=settings,
                            page=getattr(renderer, "_page", None),
                        )
                        scene_visual = SceneVisual(kind="image", image=fallback_png)
                    except Exception as fallback_exc:
                        raise SceneError(
                            f"Static fallback slide failed for scene {scene.id}: {fallback_exc}",
                            hint="Ensure Playwright Chromium is properly installed.",
                        ) from fallback_exc

            # 3. Animated scene types (code, terminal, architecture, stats)
            elif scene.type in ("code", "terminal", "architecture", "stats"):
                try:
                    if scene.type == "code":
                        if not scene.file_id or scene.file_id not in key_file_by_id:
                            raise SceneError(f"Key file '{scene.file_id}' not found.")
                        kf = key_file_by_id[scene.file_id]
                        html_doc = build_code_html(
                            scene,
                            kf,
                            theme=options.theme,
                            accent=options.accent,
                            repo_label=repo_label,
                            index=idx + 1,
                            total=total_scenes,
                            duration=v_len,
                            settings=settings,
                        )
                    elif scene.type == "terminal":
                        if not scene.terminal_id or scene.terminal_id not in terminal_by_id:
                            raise SceneError(f"Terminal session '{scene.terminal_id}' not found.")
                        sess = terminal_by_id[scene.terminal_id]
                        html_doc = build_terminal_html(
                            scene,
                            sess,
                            theme=options.theme,
                            accent=options.accent,
                            repo_label=repo_label,
                            index=idx + 1,
                            total=total_scenes,
                            duration=v_len,
                            settings=settings,
                        )
                    elif scene.type == "architecture":
                        html_doc = build_arch_html(
                            scene,
                            scene.architecture,
                            theme=options.theme,
                            accent=options.accent,
                            repo_label=repo_label,
                            index=idx + 1,
                            total=total_scenes,
                            duration=v_len,
                            settings=settings,
                        )
                    elif scene.type == "stats":
                        html_doc = build_stats_html(
                            scene,
                            repo_data,
                            theme=options.theme,
                            accent=options.accent,
                            repo_label=repo_label,
                            index=idx + 1,
                            total=total_scenes,
                            duration=v_len,
                            settings=settings,
                        )
                    else:
                        raise SceneError(f"Unhandled animated scene type '{scene.type}'")

                    if not options.no_animation:
                        out_clip = clips_dir / f"scene_{scene.id:02d}.mp4"
                        renderer.render(html_doc, duration=v_len, out_path=out_clip)
                        scene_visual = SceneVisual(kind="video", video=out_clip)
                    else:
                        t_capture = 0.85 * v_len
                        frame_bytes = renderer.capture_frame(html_doc, t=t_capture)
                        out_png = slides_dir / f"scene_{scene.id:02d}.png"
                        out_png.write_bytes(frame_bytes)
                        scene_visual = SceneVisual(kind="image", image=out_png)

                except Exception as exc:  # noqa: BLE001
                    reason = getattr(exc, "message", str(exc))
                    logger.warning(
                        "Scene %d (%s) failed: %s; falling back to static slide.",
                        scene.id,
                        scene.type,
                        reason,
                    )
                    fallbacks.append(
                        {
                            "scene_id": scene.id,
                            "type": scene.type,
                            "reason": reason,
                        }
                    )
                    try:
                        fallback_html = build_html(
                            scene,
                            index=idx + 1,
                            total=total_scenes,
                            repo_label=repo_label,
                            theme=options.theme,
                            accent=options.accent,
                        )
                        fallback_png = slides_dir / f"scene_{scene.id:02d}.png"
                        screenshot_html(
                            fallback_html,
                            fallback_png,
                            settings=settings,
                            page=getattr(renderer, "_page", None),
                        )
                        scene_visual = SceneVisual(kind="image", image=fallback_png)
                    except Exception as fallback_exc:
                        raise SceneError(
                            f"Static fallback slide failed for scene {scene.id}: {fallback_exc}",
                            hint="Ensure Playwright Chromium is properly installed.",
                        ) from fallback_exc

            # 4. Unknown/unexpected scene type fallback
            else:
                fallback_html = build_html(
                    scene,
                    index=idx + 1,
                    total=total_scenes,
                    repo_label=repo_label,
                    theme=options.theme,
                    accent=options.accent,
                )
                fallback_png = slides_dir / f"scene_{scene.id:02d}.png"
                screenshot_html(
                    fallback_html,
                    fallback_png,
                    settings=settings,
                    page=getattr(renderer, "_page", None),
                )
                scene_visual = SceneVisual(kind="image", image=fallback_png)

            if scene_visual is None:
                raise SceneError(f"Could not produce visual for scene {scene.id}")

            scene_visuals.append(scene_visual)
            if on_scene is not None:
                on_scene(idx + 1, total_scenes, scene.type)

    return scene_visuals, fallbacks, c_line


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
    media_assets: list[MediaAsset] | None = None,
    fallbacks: list[dict[str, Any]] | None = None,
    credit_line_str: str | None = None,
) -> None:
    """Write run.json metadata file containing execution metrics without secrets."""
    scene_count = len(script_data.scenes) if script_data is not None else 0
    total_words = (
        sum(len(s.narration.split()) for s in script_data.scenes) if script_data is not None else 0
    )
    scene_types = dict(Counter(s.type for s in script_data.scenes)) if script_data else {}

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
        "scene_types": scene_types,
        "fallbacks": fallbacks or [],
        "media": {
            "count": len(media_assets or []),
            "origins": sorted({a.origin for a in (media_assets or [])}),
        },
        "animation": {
            "fps": settings.anim_fps,
            "quality": settings.anim_quality,
            "enabled": not options.no_animation,
        },
        "flags": {
            "no_media": options.no_media,
            "no_animation": options.no_animation,
            "no_transitions": options.no_transitions,
            "captions": options.captions,
            "fast": options.fast,
            "music": bool(options.music),
        },
        "credit_line": credit_line_str,
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
    """Execute the full video generation pipeline from GitHub URL to finished MP4."""
    resolved_settings = settings if settings is not None else get_settings()
    resolved_options = options if options is not None else PipelineOptions()

    # Apply option overrides to settings
    effective_settings = resolved_settings
    if resolved_options.fast:
        effective_settings = replace(
            effective_settings,
            anim_quality="fast",
            anim_fps=15 if resolved_options.anim_fps is None else resolved_options.anim_fps,
        )
    elif resolved_options.anim_fps is not None:
        effective_settings = replace(effective_settings, anim_fps=resolved_options.anim_fps)

    owner, repo_name = parse_repo_url(url)
    folder_name = f"{_sanitize_name(owner)}__{_sanitize_name(repo_name)}"
    work_dir = effective_settings.output_dir / folder_name
    work_dir.mkdir(parents=True, exist_ok=True)

    repo_json = work_dir / "repo.json"
    scenes_json = work_dir / "scenes.json"
    audio_dir = work_dir / "audio"
    slides_dir = work_dir / "slides"
    clips_dir = work_dir / "clips"
    debug_dir = work_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    run_json = work_dir / "run.json"

    # Preflight check before starting any stage
    _run_preflight(effective_settings, script_only=resolved_options.script_only)

    if resolved_options.force:
        shutil.rmtree(slides_dir, ignore_errors=True)
        shutil.rmtree(clips_dir, ignore_errors=True)
        shutil.rmtree(audio_dir, ignore_errors=True)

    stage_names = (
        ["analyze", "media", "script"]
        if resolved_options.script_only
        else ["analyze", "media", "script", "voice", "visuals", "render"]
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
            repo_data = analyze(url, settings=effective_settings)
            repo_data.save(repo_json)
            timings["analyze"] = time.monotonic() - t0
    else:
        t0 = time.monotonic()
        repo_data = analyze(url, settings=effective_settings)
        repo_data.save(repo_json)
        timings["analyze"] = time.monotonic() - t0

    # Stage 2: media
    resolved_options.on_stage("media", 2, total_stages)
    media_assets: list[MediaAsset] = []
    if resolved_options.no_media:
        logger.info("Skipping media collection stage (--no-media).")
        timings["media"] = 0.0
    else:
        t0 = time.monotonic()
        try:
            media_assets = collect_media(
                repo_data,
                work_dir,
                effective_settings,
                force=resolved_options.force,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Media collection failed (%s); continuing without media.", exc)
            media_assets = []
        timings["media"] = time.monotonic() - t0

    # Stage 3: script
    resolved_options.on_stage("script", 3, total_stages)
    script_data: ScriptData
    if resolved_options.reuse and not resolved_options.force and scenes_json.is_file():
        try:
            script_data = ScriptData.load(scenes_json)
            logger.info("Reusing existing scenes.json from %s", scenes_json)
            script_data = resolve_references(script_data, repo_data, media_assets=media_assets)
            script_data.save(scenes_json)
            timings["script"] = 0.0
        except (Repo2VideoError, ValueError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load existing scenes.json (%s); re-generating.", exc)
            t0 = time.monotonic()
            client = get_client(effective_settings)
            try:
                script_data = write_script(
                    repo_data,
                    client,
                    settings=effective_settings,
                    media_assets=media_assets,
                )
            except TypeError:
                script_data = write_script(repo_data, client, settings=effective_settings)
            script_data = resolve_references(script_data, repo_data, media_assets=media_assets)
            script_data.save(scenes_json)
            timings["script"] = time.monotonic() - t0
    else:
        t0 = time.monotonic()
        client = get_client(effective_settings)
        try:
            script_data = write_script(
                repo_data,
                client,
                settings=effective_settings,
                media_assets=media_assets,
            )
        except TypeError:
            script_data = write_script(repo_data, client, settings=effective_settings)
        script_data = resolve_references(script_data, repo_data, media_assets=media_assets)
        script_data.save(scenes_json)
        timings["script"] = time.monotonic() - t0

    if resolved_options.script_only:
        _write_run_json(
            run_json,
            url=url,
            settings=effective_settings,
            options=resolved_options,
            script_data=script_data,
            timings=timings,
            repo_json=repo_json,
            scenes_json=scenes_json,
            video_path=None,
            captions_path=None,
            media_assets=media_assets,
            fallbacks=[],
        )
        return PipelineResult(
            work_dir=work_dir,
            repo_json=repo_json,
            scenes_json=scenes_json,
            video=None,
            captions=None,
            duration_seconds=None,
            timings=timings,
            scene_counts=dict(Counter(s.type for s in script_data.scenes)),
            fallbacks=[],
            media_count=len(media_assets),
        )

    # Stage 4: voice
    resolved_options.on_stage("voice", 4, total_stages)
    audio_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    audios = synthesize_all(
        script_data,
        audio_dir,
        settings=effective_settings,
    )
    timings["voice"] = time.monotonic() - t0

    # Build RenderOptions
    render_opts = RenderOptions(
        transitions=not resolved_options.no_transitions,
        transition_seconds=0.4,
        captions=resolved_options.captions,
        music_path=resolved_options.music,
        music_volume=0.10,
        crf=24 if resolved_options.fast else 20,
        preset="veryfast" if resolved_options.fast else "medium",
        theme=resolved_options.theme,
        keep_temp=resolved_options.keep_temp,
    )

    # Stage 5: visuals
    resolved_options.on_stage("visuals", 5, total_stages)
    t0 = time.monotonic()
    scene_visuals, fallbacks, credit_line_str = _render_visuals(
        script_data=script_data,
        repo_data=repo_data,
        media_assets=media_assets,
        work_dir=work_dir,
        audios=audios,
        settings=effective_settings,
        options=resolved_options,
        render_options=render_opts,
        on_scene=resolved_options.on_scene,
    )
    timings["visuals"] = time.monotonic() - t0

    # Pair visuals and audio into SceneAssets
    audio_by_id = {a.scene_id: a for a in audios}
    scene_assets: list[SceneAssets] = []
    for sc, vis in zip(script_data.scenes, scene_visuals):
        aud = audio_by_id[sc.id]
        scene_assets.append(
            SceneAssets(
                scene_id=sc.id,
                audio=aud.path,
                narration_seconds=aud.duration_seconds,
                narration=sc.narration,
                visual=vis,
            )
        )

    # Stage 6: render
    resolved_options.on_stage("render", 6, total_stages)
    target_video = Path(out).resolve() if out is not None else work_dir / "demo.mp4"
    target_captions = (
        target_video.with_suffix(".srt") if resolved_options.captions != "off" else None
    )

    t0 = time.monotonic()
    try:
        render_video(
            scene_assets,
            target_video,
            settings=effective_settings,
            options=render_opts,
            keep_temp=resolved_options.keep_temp,
        )
    except TypeError:
        render_video(
            scene_assets,
            target_video,
            settings=effective_settings,
            keep_temp=resolved_options.keep_temp,
        )
    timings["render"] = time.monotonic() - t0

    probe_info = probe_video(target_video)
    duration_seconds = probe_info["duration_seconds"]

    _write_run_json(
        run_json,
        url=url,
        settings=effective_settings,
        options=resolved_options,
        script_data=script_data,
        timings=timings,
        repo_json=repo_json,
        scenes_json=scenes_json,
        video_path=target_video,
        captions_path=target_captions,
        media_assets=media_assets,
        fallbacks=fallbacks,
        credit_line_str=credit_line_str,
    )

    return PipelineResult(
        work_dir=work_dir,
        repo_json=repo_json,
        scenes_json=scenes_json,
        video=target_video,
        captions=target_captions,
        duration_seconds=duration_seconds,
        timings=timings,
        scene_counts=dict(Counter(s.type for s in script_data.scenes)),
        fallbacks=fallbacks,
        media_count=len(media_assets),
        credit_line=credit_line_str,
    )


def render_only(
    scenes_json: Path,
    repo_label: str,
    out: Path | None = None,
    settings: Settings | None = None,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    """Render slides, voice, and video starting from an existing scenes.json script file."""
    resolved_settings = settings if settings is not None else get_settings()
    resolved_options = options if options is not None else PipelineOptions()

    effective_settings = resolved_settings
    if resolved_options.fast:
        effective_settings = replace(
            effective_settings,
            anim_quality="fast",
            anim_fps=15 if resolved_options.anim_fps is None else resolved_options.anim_fps,
        )
    elif resolved_options.anim_fps is not None:
        effective_settings = replace(effective_settings, anim_fps=resolved_options.anim_fps)

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
    audio_dir = work_dir / "audio"
    slides_dir = work_dir / "slides"
    clips_dir = work_dir / "clips"
    run_json = work_dir / "run.json"

    # Load or mock repo_data
    repo_data: RepoData
    if repo_json.is_file():
        try:
            repo_data = RepoData.load(repo_json)
        except Exception:  # noqa: BLE001
            owner_part, _, repo_part = repo_label.partition("/")
            repo_data = RepoData(
                owner=owner_part or "unknown",
                name=repo_part or repo_label or "unknown",
                readme="",
            )
    else:
        owner_part, _, repo_part = repo_label.partition("/")
        repo_data = RepoData(
            owner=owner_part or "unknown",
            name=repo_part or repo_label or "unknown",
            readme="",
        )

    # Discover any downloaded media assets
    media_dir = work_dir / "media"
    media_assets: list[MediaAsset] = []
    if media_dir.is_dir():
        for f in media_dir.glob("asset_*"):
            if f.is_file() and not f.name.endswith(".json"):
                kind = (
                    "clip"
                    if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov", ".gif")
                    else "image"
                )
                media_assets.append(
                    MediaAsset(
                        id=f.stem,
                        kind=kind,
                        path=f,
                        width=1920,
                        height=1080,
                        origin="readme",
                        source_url="",
                    )
                )

    # Always resolve references on loaded script
    script_data = resolve_references(script_data, repo_data, media_assets=media_assets)
    script_data.save(path)

    # Preflight check Piper and FFmpeg
    _run_preflight(
        effective_settings,
        script_only=False,
        check_llm=False,
    )

    if resolved_options.force:
        shutil.rmtree(slides_dir, ignore_errors=True)
        shutil.rmtree(clips_dir, ignore_errors=True)
        shutil.rmtree(audio_dir, ignore_errors=True)

    stage_names = ["voice", "visuals", "render"]
    total_stages = len(stage_names)
    timings: dict[str, float] = {}

    # Stage 1: voice
    resolved_options.on_stage("voice", 1, total_stages)
    audio_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    audios = synthesize_all(
        script_data,
        audio_dir,
        settings=effective_settings,
    )
    timings["voice"] = time.monotonic() - t0

    render_opts = RenderOptions(
        transitions=not resolved_options.no_transitions,
        transition_seconds=0.4,
        captions=resolved_options.captions,
        music_path=resolved_options.music,
        music_volume=0.10,
        crf=24 if resolved_options.fast else 20,
        preset="veryfast" if resolved_options.fast else "medium",
        theme=resolved_options.theme,
        keep_temp=resolved_options.keep_temp,
    )

    # Stage 2: visuals
    resolved_options.on_stage("visuals", 2, total_stages)
    t0 = time.monotonic()
    scene_visuals, fallbacks, credit_line_str = _render_visuals(
        script_data=script_data,
        repo_data=repo_data,
        media_assets=media_assets,
        work_dir=work_dir,
        audios=audios,
        settings=effective_settings,
        options=resolved_options,
        render_options=render_opts,
        on_scene=resolved_options.on_scene,
    )
    timings["visuals"] = time.monotonic() - t0

    audio_by_id = {a.scene_id: a for a in audios}
    scene_assets: list[SceneAssets] = []
    for sc, vis in zip(script_data.scenes, scene_visuals):
        aud = audio_by_id[sc.id]
        scene_assets.append(
            SceneAssets(
                scene_id=sc.id,
                audio=aud.path,
                narration_seconds=aud.duration_seconds,
                narration=sc.narration,
                visual=vis,
            )
        )

    # Stage 3: render
    resolved_options.on_stage("render", 3, total_stages)
    target_video = Path(out).resolve() if out is not None else work_dir / "demo.mp4"
    target_captions = (
        target_video.with_suffix(".srt") if resolved_options.captions != "off" else None
    )

    t0 = time.monotonic()
    try:
        render_video(
            scene_assets,
            target_video,
            settings=effective_settings,
            options=render_opts,
            keep_temp=resolved_options.keep_temp,
        )
    except TypeError:
        render_video(
            scene_assets,
            target_video,
            settings=effective_settings,
            keep_temp=resolved_options.keep_temp,
        )
    timings["render"] = time.monotonic() - t0

    probe_info = probe_video(target_video)
    duration_seconds = probe_info["duration_seconds"]

    _write_run_json(
        run_json,
        url="",
        settings=effective_settings,
        options=resolved_options,
        script_data=script_data,
        timings=timings,
        repo_json=repo_json,
        scenes_json=path,
        video_path=target_video,
        captions_path=target_captions,
        media_assets=media_assets,
        fallbacks=fallbacks,
        credit_line_str=credit_line_str,
    )

    return PipelineResult(
        work_dir=work_dir,
        repo_json=repo_json,
        scenes_json=path,
        video=target_video,
        captions=target_captions,
        duration_seconds=duration_seconds,
        timings=timings,
        scene_counts=dict(Counter(s.type for s in script_data.scenes)),
        fallbacks=fallbacks,
        media_count=len(media_assets),
        credit_line=credit_line_str,
    )
