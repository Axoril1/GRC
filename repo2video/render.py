"""Video rendering stage that composites visual slides and audio tracks into an MP4 video using FFmpeg."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repo2video.config import Settings, get_settings
from repo2video.errors import RenderError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneAssets:
    """Assets required to render a single video scene.

    Attributes:
        scene_id: Numeric identifier of the scene (1-based).
        image: Path to the scene slide PNG image.
        audio: Path to the scene narration WAV audio.
        duration_seconds: Duration of the audio track in seconds.
        narration: Narration text for the scene (used for subtitle generation).
    """

    scene_id: int
    image: Path
    audio: Path
    duration_seconds: float
    narration: str

    def __post_init__(self) -> None:
        """Ensure image and audio are Path instances."""
        if not isinstance(self.image, Path):
            object.__setattr__(self, "image", Path(self.image))
        if not isinstance(self.audio, Path):
            object.__setattr__(self, "audio", Path(self.audio))


def check_ffmpeg() -> str:
    """Verify that ffmpeg and ffprobe are available and support required encoders.

    Checks PATH for both binaries, queries available encoders to ensure
    libx264 and aac are present, and returns the first line of ffmpeg -version.

    Returns:
        First line of output from ffmpeg -version.

    Raises:
        RenderError: If ffmpeg or ffprobe are missing, or required encoders are absent.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    if not ffmpeg_bin or not ffprobe_bin:
        raise RenderError(
            "FFmpeg is not installed or not on PATH.",
            hint=(
                "Install FFmpeg: Windows (winget install ffmpeg), "
                "macOS (brew install ffmpeg), "
                "Ubuntu or Debian (sudo apt install ffmpeg)."
            ),
        )

    # Verify libx264 and aac encoders
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        raise RenderError(
            "FFmpeg is not installed or not on PATH.",
            hint=(
                "Install FFmpeg: Windows (winget install ffmpeg), "
                "macOS (brew install ffmpeg), "
                "Ubuntu or Debian (sudo apt install ffmpeg)."
            ),
        ) from exc

    if res.returncode != 0:
        raise RenderError(
            "Failed to query FFmpeg encoders.",
            hint="Ensure FFmpeg runs properly from your command line.",
        )

    encoders_output = res.stdout
    has_libx264 = "libx264" in encoders_output
    has_aac = bool(
        re.search(r"A[A-Z.]{5}\s+aac\b", encoders_output)
        or " aac " in encoders_output
        or "\naac " in encoders_output
    )

    if not has_libx264 or not has_aac:
        missing: list[str] = []
        if not has_libx264:
            missing.append("libx264")
        if not has_aac:
            missing.append("aac")
        missing_str = " and ".join(missing)
        raise RenderError(
            f"FFmpeg build is missing required encoder(s): {missing_str}.",
            hint="Install a full FFmpeg build with libx264 and aac support enabled.",
        )

    # Retrieve version line
    try:
        ver_res = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        raise RenderError(
            "Failed to query FFmpeg version.",
            hint="Ensure FFmpeg is installed and accessible.",
        ) from exc

    if ver_res.returncode != 0 or not ver_res.stdout:
        raise RenderError(
            "Failed to obtain FFmpeg version.",
            hint="Ensure FFmpeg runs properly from your command line.",
        )

    return ver_res.stdout.splitlines()[0].strip()


def build_segment_command(assets: SceneAssets, out_path: Path, settings: Settings) -> list[str]:
    """Build the FFmpeg command argument list to render a single scene segment.

    Builds each segment as a Matroska file (.mkv), not an MP4, so that audio
    is stored without AAC encoder delay.

    Args:
        assets: Scene assets including slide image, audio path, and duration.
        out_path: Output path for the rendered .mkv segment.
        settings: Application settings with video dimensions, fps, and pause.

    Returns:
        List of command arguments ready for subprocess execution.
    """
    total_duration = assets.duration_seconds + settings.scene_pause_seconds
    total_dur_str = f"{total_duration:.3f}"
    pause_str = f"{settings.scene_pause_seconds:.3f}"

    # -t makes the length exact and removes any chance of an endless looped image;
    # apad adds the pause as silence; the audio is resampled to 44100 Hz stereo so
    # every segment has identical parameters, which the concat step requires.
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        str(settings.video_fps),
        "-i",
        str(assets.image),
        "-i",
        str(assets.audio),
        "-af",
        f"apad=pad_dur={pause_str}",
        "-t",
        total_dur_str,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(settings.video_fps),
        "-vf",
        f"scale={settings.video_width}:{settings.video_height}:flags=lanczos,setsar=1",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(out_path),
    ]


def build_concat_list(paths: list[Path]) -> str:
    """Build FFmpeg concat demuxer file contents for a sequence of segments.

    Each line specifies a segment file using forward slashes and single quotes,
    escaping any single quotes within the path using '\\''.

    Args:
        paths: Ordered list of segment file paths.

    Returns:
        Newline-terminated string formatted for FFmpeg's concat demuxer.
    """
    lines: list[str] = []
    for p in paths:
        posix_str = Path(p).resolve().as_posix()
        escaped = posix_str.replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + ("\n" if lines else "")


def make_segment(assets: SceneAssets, out_path: Path, settings: Settings) -> Path:
    """Render a single scene segment (.mkv) from slide image and narration audio.

    Args:
        assets: Scene assets with image, audio, and duration.
        out_path: Target path for the output .mkv file.
        settings: Application settings.

    Returns:
        Path to the rendered segment file.

    Raises:
        RenderError: If input files are missing or FFmpeg encoding fails.
    """
    out_path = Path(out_path)
    if not assets.image.is_file():
        raise RenderError(
            f"Scene {assets.scene_id} image not found at {assets.image}.",
            hint="Ensure scene builder generated all slide images.",
        )
    if not assets.audio.is_file():
        raise RenderError(
            f"Scene {assets.scene_id} audio not found at {assets.audio}.",
            hint="Ensure voice stage generated all narration audio files.",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_segment_command(assets, out_path, settings)

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(
            f"FFmpeg segment generation for scene {assets.scene_id} timed out after 300 seconds.",
            hint="Try shorter narration or a smaller video resolution.",
        ) from exc
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        raise RenderError(
            f"Failed to execute FFmpeg for scene {assets.scene_id}: {exc}",
            hint="Ensure FFmpeg is installed and accessible on PATH.",
        ) from exc

    if res.returncode != 0:
        stderr_str = res.stderr.strip()
        err_tail = stderr_str[-600:] if len(stderr_str) > 600 else stderr_str
        raise RenderError(
            f"FFmpeg failed rendering scene {assets.scene_id} (exit code {res.returncode}): {err_tail}",
            hint="Check input image and audio validity.",
        )

    if not out_path.is_file():
        raise RenderError(
            f"FFmpeg segment output was not created for scene {assets.scene_id} at {out_path}.",
            hint="Check disk space and write permissions.",
        )

    return out_path


def probe_video(path: Path) -> dict[str, Any]:
    """Probe a video file using ffprobe and return stream and format metadata.

    Args:
        path: Path to the media file to inspect.

    Returns:
        Dictionary containing 'duration_seconds' (float), 'width' (int),
        'height' (int), and 'has_audio' (bool).

    Raises:
        RenderError: If probing fails, output is invalid JSON, or video stream is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise RenderError(
            f"File not found for probing at {path}.",
            hint="Ensure the media file was created before probing.",
        )

    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        raise RenderError(
            f"Failed to run ffprobe on {path}: {exc}",
            hint="Ensure ffprobe is installed and on PATH.",
        ) from exc

    if res.returncode != 0:
        stderr_tail = res.stderr.strip()[-600:]
        raise RenderError(
            f"ffprobe failed for {path} (exit code {res.returncode}): {stderr_tail}",
            hint="Verify the file is a valid video file.",
        )

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise RenderError(
            f"ffprobe returned invalid JSON for {path}: {exc}",
            hint="Check that ffprobe produces valid JSON output.",
        ) from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    width = 0
    height = 0
    video_found = False
    has_audio = False
    video_duration: float | None = None

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not video_found:
            video_found = True
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            if "duration" in stream:
                try:
                    video_duration = float(stream["duration"])
                except (ValueError, TypeError):
                    pass
        elif codec_type == "audio":
            has_audio = True

    duration = 0.0
    if "duration" in fmt:
        try:
            duration = float(fmt["duration"])
        except (ValueError, TypeError):
            pass
    elif video_duration is not None:
        duration = video_duration

    if not video_found:
        raise RenderError(
            f"No video stream found in {path}.",
            hint="Check that video encoding succeeded.",
        )

    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "has_audio": has_audio,
    }


def format_srt_timestamp(total_seconds: float) -> str:
    """Format floating-point seconds into an SRT timestamp HH:MM:SS,mmm.

    Args:
        total_seconds: Time in seconds (non-negative).

    Returns:
        Formatted timestamp string 'HH:MM:SS,mmm'.
    """
    if total_seconds < 0:
        total_seconds = 0.0
    millis = round(total_seconds * 1000)
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(assets: list[SceneAssets], out_path: Path, settings: Settings | None = None) -> Path:
    """Create an SRT subtitle file aligned with scene narrations and pauses.

    Each scene's narration is wrapped into lines of at most 42 characters at
    word boundaries, grouped into cues of at most 2 lines, and distributed
    proportionally across the scene's narration duration.

    Args:
        assets: Ordered list of scene assets.
        out_path: Target path for the output .srt file.
        settings: Application settings. Defaults to get_settings().

    Returns:
        Path to the written .srt file.

    Raises:
        RenderError: If writing the SRT file fails.
    """
    resolved_settings = settings if settings is not None else get_settings()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cues_blocks: list[str] = []
    cue_index = 1
    current_scene_start = 0.0

    for asset in assets:
        scene_pause = resolved_settings.scene_pause_seconds
        narration_clean = " ".join(asset.narration.split())

        if narration_clean:
            # Wrap narration into lines of at most 42 characters at word boundaries
            lines = textwrap.wrap(
                narration_clean,
                width=42,
                break_long_words=True,
                break_on_hyphens=False,
            )

            # Group lines into cues with at most 2 lines each
            cues_text: list[str] = []
            for i in range(0, len(lines), 2):
                cues_text.append("\n".join(lines[i : i + 2]))

            total_chars = sum(len(c) for c in cues_text)
            if total_chars == 0:
                total_chars = 1

            scene_dur = max(0.0, asset.duration_seconds)
            cum_chars = 0

            for cue_str in cues_text:
                start_offset = scene_dur * (cum_chars / total_chars)
                cum_chars += len(cue_str)
                end_offset = scene_dur * (cum_chars / total_chars)

                t_start = current_scene_start + start_offset
                t_end = current_scene_start + end_offset

                start_ts = format_srt_timestamp(t_start)
                end_ts = format_srt_timestamp(t_end)

                cue_block = f"{cue_index}\n{start_ts} --> {end_ts}\n{cue_str}"
                cues_blocks.append(cue_block)
                cue_index += 1

        # Advance scene start time by scene narration duration plus pause
        current_scene_start += asset.duration_seconds + scene_pause

    srt_content = "\n\n".join(cues_blocks) + ("\n" if cues_blocks else "")
    try:
        out_path.write_text(srt_content, encoding="utf-8")
    except OSError as exc:
        raise RenderError(
            f"Failed to write subtitles file {out_path}: {exc}",
            hint="Check disk write permissions and directory existence.",
        ) from exc

    return out_path


def render_video(
    assets: list[SceneAssets],
    out_path: Path,
    settings: Settings | None = None,
    *,
    keep_temp: bool = False,
) -> Path:
    """Render a full MP4 demo video and companion SRT subtitles from scene assets.

    Composes each scene slide and audio into a temporary lossless .mkv segment,
    concatenates all segments without re-encoding video, encodes audio to AAC,
    validates output stream properties, and generates an SRT file.

    Args:
        assets: Ordered list of SceneAssets for all scenes in the video.
        out_path: Target path for the final MP4 video.
        settings: Application settings. Defaults to get_settings().
        keep_temp: Whether to retain intermediate segment files in _work folder.

    Returns:
        Path to the finished MP4 video file.

    Raises:
        RenderError: If rendering, concatenation, validation, or subtitles fail.
    """
    resolved_settings = settings if settings is not None else get_settings()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not assets:
        raise RenderError(
            "No scene assets provided to render_video.",
            hint="Provide at least one SceneAssets item to render.",
        )

    # 1. Pre-flight check FFmpeg and input file existence
    check_ffmpeg()
    for a in assets:
        if not a.image.is_file():
            raise RenderError(
                f"Scene {a.scene_id} image not found at {a.image}.",
                hint="Ensure scene builder generated all slide images.",
            )
        if not a.audio.is_file():
            raise RenderError(
                f"Scene {a.scene_id} audio not found at {a.audio}.",
                hint="Ensure voice stage generated all narration audio files.",
            )

    work_dir = out_path.parent / f"{out_path.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    segment_paths: list[Path] = []

    try:
        # 2. Render each segment
        for a in assets:
            seg_path = work_dir / f"segment_{a.scene_id:02d}.mkv"
            t0 = time.monotonic()
            make_segment(a, seg_path, resolved_settings)
            elapsed = time.monotonic() - t0
            segment_paths.append(seg_path)
            logger.info(
                "Scene %d rendered in %.2fs (duration: %.2fs)",
                a.scene_id,
                elapsed,
                a.duration_seconds,
            )

        # 3. Build concat list and concatenate
        concat_file = work_dir / "concat_list.txt"
        concat_text = build_concat_list(segment_paths)
        concat_file.write_text(concat_text, encoding="utf-8")

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        logger.debug("Executing video concatenation: %s", concat_cmd)

        try:
            concat_res = subprocess.run(
                concat_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderError(
                "FFmpeg video concatenation timed out after 600 seconds.",
                hint="Check system disk I/O performance and video length.",
            ) from exc
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            raise RenderError(
                f"FFmpeg concatenation failed to execute: {exc}",
                hint="Ensure FFmpeg is installed and accessible on PATH.",
            ) from exc

        if concat_res.returncode != 0:
            stderr_str = concat_res.stderr.strip()
            err_tail = stderr_str[-600:] if len(stderr_str) > 600 else stderr_str
            raise RenderError(
                f"FFmpeg video concatenation failed (exit code {concat_res.returncode}): {err_tail}",
                hint="Ensure all segments were rendered with identical parameters.",
            )

        # 4. Generate companion subtitles
        srt_path = out_path.with_suffix(".srt")
        write_srt(assets, srt_path, settings=resolved_settings)

        # 5. Validate rendered video
        info = probe_video(out_path)
        if not info["has_audio"]:
            raise RenderError(
                f"Rendered video {out_path} has no audio stream.",
                hint="Ensure input audio files were valid and AAC encoding succeeded.",
            )
        if (
            info["width"] != resolved_settings.video_width
            or info["height"] != resolved_settings.video_height
        ):
            raise RenderError(
                f"Rendered video dimensions {info['width']}x{info['height']} do not match "
                f"configured dimensions {resolved_settings.video_width}x{resolved_settings.video_height}.",
                hint="Check video scaling filter in segment generation.",
            )

        expected_duration = sum(
            a.duration_seconds + resolved_settings.scene_pause_seconds for a in assets
        )
        diff = abs(info["duration_seconds"] - expected_duration)
        if diff > 0.75:
            raise RenderError(
                f"Rendered video duration {info['duration_seconds']:.2f}s differs from expected "
                f"{expected_duration:.2f}s by {diff:.2f}s (allowed tolerance is 0.75s).",
                hint="Check scene segment durations and audio padding.",
            )

        file_size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Final video rendered: %s (%.2f MB, %.2fs duration)",
            out_path,
            file_size_mb,
            info["duration_seconds"],
        )

    finally:
        if not keep_temp and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    return out_path


def main() -> None:
    """CLI entrypoint to check FFmpeg version and readiness."""
    parser = argparse.ArgumentParser(
        description="repo2video renderer: compose slides and audio into an MP4 video."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether FFmpeg and required encoders are installed and ready.",
    )
    args = parser.parse_args()

    if args.check:
        try:
            ver = check_ffmpeg()
            print(f"FFmpeg is ready: {ver}")
            sys.exit(0)
        except RenderError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
