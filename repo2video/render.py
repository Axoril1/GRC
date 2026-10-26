"""Video rendering stage that composites visual slides, clips, overlays, and audio tracks into an MP4 video using FFmpeg."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repo2video.captions import (
    Cue,
    build_cues,
    format_srt_timestamp,
    render_caption_images,
)
from repo2video.captions import write_srt as captions_write_srt
from repo2video.config import Settings, get_settings
from repo2video.errors import RenderError
from repo2video.scene_media import Rect, media_frame_rect

logger = logging.getLogger(__name__)

__all__ = [
    "Capabilities",
    "RenderOptions",
    "SceneAssets",
    "SceneVisual",
    "Timeline",
    "build_caption_graph",
    "build_concat_list",
    "build_segment_command",
    "build_xfade_graph",
    "capabilities",
    "check_ffmpeg",
    "compute_timeline",
    "format_srt_timestamp",
    "make_segment",
    "probe_video",
    "render_video",
    "scene_video_lengths",
    "write_srt",
]

ALLOWED_MUSIC_EXTENSIONS: frozenset[str] = frozenset([".mp3", ".wav", ".ogg", ".m4a", ".flac"])


@dataclass(frozen=True)
class SceneVisual:
    """Visual content specification for a single video scene.

    Attributes:
        kind: Visual category ('image', 'video', or 'overlay').
        image: Path to slide PNG image (for image), or background PNG (for overlay).
        video: Path to prepared silent video clip (for video or overlay).
        rect: Bounding box inside background image for overlay placement.
    """

    kind: str
    image: Path | None = None
    video: Path | None = None
    rect: Rect | None = None

    def __post_init__(self) -> None:
        """Coerce image and video to Path instances when provided."""
        if self.image is not None and not isinstance(self.image, Path):
            object.__setattr__(self, "image", Path(self.image))
        if self.video is not None and not isinstance(self.video, Path):
            object.__setattr__(self, "video", Path(self.video))


@dataclass(frozen=True)
class SceneAssets:
    """Assets required to render a single video scene.

    Compatible with both v1 (image + duration_seconds) and v2 (visual + narration_seconds).

    Attributes:
        scene_id: Numeric identifier of the scene (1-based).
        audio: Path to the scene narration WAV audio.
        narration_seconds: Duration of the audio track in seconds.
        narration: Narration text for the scene (used for subtitle generation).
        visual: Visual assets definition (SceneVisual instance).
        image: Slide image path (v1 compatibility).
        duration_seconds: Audio duration in seconds (v1 compatibility).
    """

    scene_id: int
    audio: Path
    narration_seconds: float = 0.0
    narration: str = ""
    visual: SceneVisual | None = None
    image: Path | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        """Coerce paths and synchronize v1 and v2 duration and visual fields."""
        if self.audio is not None and not isinstance(self.audio, Path):
            object.__setattr__(self, "audio", Path(self.audio))

        # Synchronize duration_seconds and narration_seconds
        dur = (
            self.narration_seconds if self.narration_seconds > 0 else (self.duration_seconds or 0.0)
        )
        object.__setattr__(self, "narration_seconds", float(dur))
        object.__setattr__(self, "duration_seconds", float(dur))

        # Synchronize visual and image
        if self.visual is None:
            if self.image is not None and not isinstance(self.image, Path):
                object.__setattr__(self, "image", Path(self.image))
        else:
            if self.image is None and self.visual.image is not None:
                object.__setattr__(self, "image", self.visual.image)


@dataclass
class RenderOptions:
    """Configurable rendering parameters for the final video output.

    Attributes:
        transitions: Whether to render crossfades between scenes.
        transition_seconds: Duration of crossfade transitions in seconds.
        captions: Caption rendering mode ('off', 'srt', or 'burn').
        music_path: Optional path to background music audio file.
        music_volume: Background music relative volume (0.0 to 1.0).
        crf: Constant Rate Factor for x264 video encoding (default: 20).
        preset: FFmpeg x264 encoding speed preset (default: 'medium').
        theme: Color theme for burned caption pills ('dark' or 'light').
        keep_temp: Retain intermediate segments in _work folder on completion.
    """

    transitions: bool = True
    transition_seconds: float = 0.4
    captions: str = "srt"
    music_path: Path | None = None
    music_volume: float = 0.10
    crf: int = 20
    preset: str = "medium"
    theme: str = "dark"
    keep_temp: bool = False

    def __post_init__(self) -> None:
        """Coerce music_path to Path if provided."""
        if self.music_path is not None and not isinstance(self.music_path, Path):
            self.music_path = Path(self.music_path)


@dataclass(frozen=True)
class Capabilities:
    """Cached FFmpeg binary features, encoders, and filter graph capabilities.

    Attributes:
        version: First line of output from `ffmpeg -version`.
        has_libx264: Whether the libx264 H.264 video encoder is available.
        has_aac: Whether the AAC audio encoder is available.
        has_xfade: Whether the xfade video crossfade filter is available.
        has_overlay: Whether the overlay video filter is available.
        has_amix_normalize: Whether amix supports the normalize parameter.
    """

    version: str
    has_libx264: bool
    has_aac: bool
    has_xfade: bool
    has_overlay: bool
    has_amix_normalize: bool = True


@dataclass(frozen=True)
class Timeline:
    """Global video and audio timeline synchronization plan.

    Attributes:
        lead: Silence lead duration before narration for each scene.
        audio_len: Total length of each scene's audio track (lead + narration + pause).
        video_len: Length of each scene's video segment (audio_len + T for overlapping scenes).
        audio_start: Timeline start time of each scene's audio track.
        narration_start: Timeline start time of actual narration speech for each scene.
        offsets: Timeline timestamp offsets for xfade transitions between adjacent scenes.
        total: Total runtime of the final video in seconds.
    """

    lead: list[float]
    audio_len: list[float]
    video_len: list[float]
    audio_start: list[float]
    narration_start: list[float]
    offsets: list[float]
    total: float


# Cached binary capabilities and runner identity
_CACHED_CAPABILITIES: Capabilities | None = None
_CACHED_RUNNER_ID: tuple[int, int] | None = None


def capabilities() -> Capabilities:
    """Query and cache FFmpeg binary capabilities, encoders, and filters.

    Returns:
        Capabilities dataclass instance.

    Raises:
        RenderError: If ffmpeg or ffprobe binaries cannot be found on PATH.
    """
    global _CACHED_CAPABILITIES, _CACHED_RUNNER_ID
    current_runner_id = (id(subprocess.run), id(shutil.which))
    if _CACHED_CAPABILITIES is not None and _CACHED_RUNNER_ID == current_runner_id:
        return _CACHED_CAPABILITIES

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

    # 1. Version
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
            "Failed to execute FFmpeg to query version.",
            hint="Ensure FFmpeg is executable and accessible.",
        ) from exc

    if ver_res.returncode != 0 or not ver_res.stdout:
        raise RenderError(
            "Failed to obtain FFmpeg version.",
            hint="Ensure FFmpeg runs properly from your command line.",
        )
    version_line = ver_res.stdout.splitlines()[0].strip()

    # 2. Encoders
    try:
        enc_res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        encoders_output = enc_res.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        encoders_output = ""

    has_libx264 = "libx264" in encoders_output
    has_aac = bool(
        re.search(r"A[A-Z.]{5}\s+aac\b", encoders_output)
        or " aac " in encoders_output
        or "\naac " in encoders_output
    )

    # 3. Filters
    try:
        filt_res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        filters_output = filt_res.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        filters_output = ""

    has_xfade = bool(re.search(r"\bxfade\b", filters_output))
    has_overlay = bool(re.search(r"\boverlay\b", filters_output))

    # 4. Amix filter options
    try:
        amix_res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "filter=amix"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        has_amix_normalize = "normalize" in amix_res.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        has_amix_normalize = False

    _CACHED_CAPABILITIES = Capabilities(
        version=version_line,
        has_libx264=has_libx264,
        has_aac=has_aac,
        has_xfade=has_xfade,
        has_overlay=has_overlay,
        has_amix_normalize=has_amix_normalize,
    )
    _CACHED_RUNNER_ID = current_runner_id
    return _CACHED_CAPABILITIES


def check_ffmpeg() -> str:
    """Verify that ffmpeg and ffprobe are available and support required encoders.

    Checks PATH for both binaries, queries available encoders to ensure
    libx264 and aac are present, and returns the first line of ffmpeg -version.

    Returns:
        First line of output from ffmpeg -version.

    Raises:
        RenderError: If ffmpeg or ffprobe are missing, or required encoders are absent.
    """
    cap = capabilities()
    if not cap.has_libx264 or not cap.has_aac:
        missing: list[str] = []
        if not cap.has_libx264:
            missing.append("libx264")
        if not cap.has_aac:
            missing.append("aac")
        missing_str = " and ".join(missing)
        raise RenderError(
            f"FFmpeg build is missing required encoder(s): {missing_str}.",
            hint="Install a full FFmpeg build with libx264 and aac support enabled.",
        )
    return cap.version


def compute_timeline(
    narration_seconds: list[float],
    pause: float,
    T: float,
    transitions: bool,
) -> Timeline:
    """Compute the master audio and video timeline for scene rendering.

    Definitions for scene i of n (0-based):
    - lead_i = T for i > 0 when transitions are on and n > 1, otherwise 0.
    - L_i = lead_i + narration_i + pause (length of scene audio track).
    - v_i = L_i + T for every scene except the last when transitions are on,
      otherwise v_i = L_i.
    - S_i = sum of L_k for k < i (timeline start of scene i audio).
    - narration_start_i = S_i + lead_i.
    - xfade offset between scene i and scene i+1 = S_{i+1} = sum of L_k for k <= i.
    - total = sum of all L_i.

    Args:
        narration_seconds: Real audio narration duration in seconds for each scene.
        pause: Silence pause duration after narration speech in seconds.
        T: Crossfade transition duration in seconds.
        transitions: Whether crossfade transitions are active.

    Returns:
        Timeline containing lead, audio_len, video_len, offsets, and total.
    """
    n = len(narration_seconds)
    if n == 0:
        return Timeline(
            lead=[],
            audio_len=[],
            video_len=[],
            audio_start=[],
            narration_start=[],
            offsets=[],
            total=0.0,
        )

    # Crossfades require transitions enabled and at least 2 scenes
    actual_trans = transitions and (n > 1)

    lead: list[float] = [round(T if (actual_trans and i > 0) else 0.0, 3) for i in range(n)]
    audio_len: list[float] = [round(lead[i] + narration_seconds[i] + pause, 3) for i in range(n)]
    video_len: list[float] = [
        round(audio_len[i] + (T if (actual_trans and i < n - 1) else 0.0), 3) for i in range(n)
    ]

    audio_start: list[float] = []
    cum_s = 0.0
    for length in audio_len:
        audio_start.append(round(cum_s, 3))
        cum_s += length

    narration_start: list[float] = [round(audio_start[i] + lead[i], 3) for i in range(n)]

    offsets: list[float] = [round(audio_start[i + 1], 3) for i in range(n - 1)]
    total = round(sum(audio_len), 3)

    return Timeline(
        lead=lead,
        audio_len=audio_len,
        video_len=video_len,
        audio_start=audio_start,
        narration_start=narration_start,
        offsets=offsets,
        total=total,
    )


def scene_video_lengths(
    narration_seconds: list[float],
    settings: Settings,
    options: RenderOptions | None = None,
) -> list[float]:
    """Calculate exact silent video lengths required for each scene before rendering.

    Args:
        narration_seconds: List of narration speech durations for each scene.
        settings: Application settings.
        options: Rendering options.

    Returns:
        List of duration floats v_i for each scene.
    """
    use_trans = options.transitions if options is not None else True
    t_dur = options.transition_seconds if options is not None else 0.4
    tl = compute_timeline(
        narration_seconds=narration_seconds,
        pause=settings.scene_pause_seconds,
        T=t_dur,
        transitions=use_trans,
    )
    return tl.video_len


def build_xfade_graph(n: int, offsets: list[float], T: float, fps: int) -> str:
    """Build the filter_complex graph string chaining xfade filters across n scenes.

    Args:
        n: Number of input video segments.
        offsets: Transition start offset timestamps for each pair.
        T: Duration of the crossfade transition in seconds.
        fps: Target video framerate.

    Returns:
        Complete FFmpeg filter_complex string with output label [x{n-1}].
    """
    if n <= 0:
        return ""
    if n == 1:
        return f"[0:v]settb=AVTB,setpts=PTS-STARTPTS,fps={fps},format=yuv420p[x0]"

    # Normalise each input segment
    norm_lines: list[str] = []
    for i in range(n):
        norm_lines.append(f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={fps},format=yuv420p[s{i}]")

    # Chain xfades
    fade_lines: list[str] = []
    for i in range(n - 1):
        prev_lbl = "[s0]" if i == 0 else f"[x{i}]"
        next_lbl = f"[s{i + 1}]"
        out_lbl = f"[x{i + 1}]"
        offset_val = offsets[i]
        fade_lines.append(
            f"{prev_lbl}{next_lbl}xfade=transition=fade:duration={T:.3f}:offset={offset_val:.3f}{out_lbl}"
        )

    return ";".join(norm_lines + fade_lines)


def build_caption_graph(
    cues: list[Cue],
    video_label: str,
    start_input_index: int = 1,
) -> tuple[str, str]:
    """Build the filter_complex overlay chain placing transparent caption pills on video.

    Args:
        cues: List of subtitle cues with timestamps.
        video_label: Current input video label in the filtergraph (e.g. '0:v' or '[x2]').
        start_input_index: First FFmpeg input index corresponding to the cue PNGs.

    Returns:
        Tuple of (filter_graph_string, final_video_label).
    """
    clean_in = video_label.strip("[]")
    if not cues:
        return ("", f"[{clean_in}]")

    current_label = f"[{clean_in}]"
    segments: list[str] = []

    for i, cue in enumerate(cues):
        cue_in = f"[{start_input_index + i}:v]"
        next_label = f"[cap_{i + 1}]"
        filt = (
            f"{current_label}{cue_in}overlay=x=(W-w)/2:y=H-h-70:"
            f"enable='between(t,{cue.start:.3f},{cue.end:.3f})':eof_action=pass{next_label}"
        )
        segments.append(filt)
        current_label = next_label

    return (";".join(segments), current_label)


def probe_audio_duration(path: Path) -> float:
    """Probe the exact duration of an audio file using ffprobe.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds as a float.

    Raises:
        RenderError: If probing fails.
    """
    path = Path(path)
    if not path.is_file():
        raise RenderError(
            f"Audio file not found at {path}.",
            hint="Ensure audio file exists before probing.",
        )
    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
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
        raise RenderError(
            f"ffprobe failed for {path}: {res.stderr.strip()[-600:]}",
            hint="Ensure file is valid audio.",
        )

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise RenderError(
            f"ffprobe returned invalid JSON for {path}: {exc}",
            hint="Check ffprobe installation.",
        ) from exc

    fmt = data.get("format", {})
    if "duration" in fmt:
        try:
            return float(fmt["duration"])
        except (ValueError, TypeError):
            pass

    for s in data.get("streams", []):
        if s.get("codec_type") == "audio" and "duration" in s:
            try:
                return float(s["duration"])
            except (ValueError, TypeError):
                pass
    return 0.0


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


def build_segment_command(
    assets: SceneAssets,
    out_path: Path,
    settings: Settings,
    *,
    video_length: float | None = None,
    options: RenderOptions | None = None,
) -> list[str]:
    """Build the FFmpeg command argument list to render a single scene segment.

    Supports v1 image+audio segments and v2 silent video segments (image, video, overlay).

    Args:
        assets: Scene assets including visual and audio paths.
        out_path: Output path for the rendered .mkv segment.
        settings: Application settings with canvas dimensions and fps.
        video_length: Target segment duration in seconds.
        options: Render options with encoding presets.

    Returns:
        List of command arguments ready for subprocess execution.
    """
    preset = options.preset if options is not None else "medium"
    crf = str(options.crf if options is not None else 20)
    fps_str = str(settings.video_fps)
    w = settings.video_width
    h = settings.video_height

    # v1 backward-compatibility mode: no visual specified and no explicit video_length
    if assets.visual is None and video_length is None:
        total_duration = assets.duration_seconds + settings.scene_pause_seconds
        total_dur_str = f"{total_duration:.3f}"
        pause_str = f"{settings.scene_pause_seconds:.3f}"
        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            fps_str,
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
            preset,
            "-crf",
            crf,
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-r",
            fps_str,
            "-vf",
            f"scale={w}:{h}:flags=lanczos,setsar=1",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(out_path),
        ]

    # v2 silent video segment mode (Step A)
    v_i = (
        video_length
        if video_length is not None
        else (assets.narration_seconds + settings.scene_pause_seconds)
    )
    dur_str = f"{v_i:.3f}"
    vis = assets.visual or SceneVisual(kind="image", image=assets.image)

    if vis.kind == "image":
        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            fps_str,
            "-i",
            str(vis.image),
            "-t",
            dur_str,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-r",
            fps_str,
            "-vf",
            f"scale={w}:{h}:flags=lanczos,setsar=1",
            "-an",
            str(out_path),
        ]

    if vis.kind == "video":
        if vis.video is None:
            raise RenderError(f"Scene {assets.scene_id} video visual is missing clip path.")
        clip_dur = 0.0
        try:
            info = probe_video(vis.video)
            clip_dur = info["duration_seconds"]
        except (RenderError, OSError):
            clip_dur = 0.0

        # If clip is shorter than v_i by more than 0.1s, hold the last frame
        if clip_dur > 0 and (v_i - clip_dur > 0.1):
            pad_dur = v_i - clip_dur
            v_filter = f"tpad=stop_mode=clone:stop_duration={pad_dur:.3f},scale={w}:{h}:flags=lanczos,setsar=1"
        else:
            v_filter = f"scale={w}:{h}:flags=lanczos,setsar=1"

        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(vis.video),
            "-t",
            dur_str,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-r",
            fps_str,
            "-vf",
            v_filter,
            "-an",
            str(out_path),
        ]

    if vis.kind == "overlay":
        if vis.image is None or vis.video is None:
            raise RenderError(
                f"Scene {assets.scene_id} overlay visual requires both background image and video clip."
            )
        rect = vis.rect or media_frame_rect(settings)
        clip_dur = 0.0
        try:
            info = probe_video(vis.video)
            clip_dur = info["duration_seconds"]
        except (RenderError, OSError):
            clip_dur = 0.0

        # The demo clip is looped with -stream_loop -1 when shorter than v_i because short
        # GIF-style demos look natural looped, and it is cut with -t v_i so long clips never
        # run past the scene.
        loop_args = ["-stream_loop", "-1"] if (clip_dur > 0 and clip_dur < v_i) else []

        ov_filter = (
            f"[1:v]scale={rect.w}:{rect.h}:force_original_aspect_ratio=decrease,"
            f"pad={rect.w}:{rect.h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps_str}[ov];"
            f"[0:v][ov]overlay=x={rect.x}:y={rect.y}[v]"
        )

        return (
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-framerate",
                fps_str,
                "-i",
                str(vis.image),
            ]
            + loop_args
            + [
                "-i",
                str(vis.video),
                "-filter_complex",
                ov_filter,
                "-map",
                "[v]",
                "-t",
                dur_str,
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                crf,
                "-pix_fmt",
                "yuv420p",
                "-r",
                fps_str,
                "-an",
                str(out_path),
            ]
        )

    raise RenderError(f"Unknown visual kind '{vis.kind}'.")


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


def make_segment(
    assets: SceneAssets,
    out_path: Path,
    settings: Settings,
    *,
    video_length: float | None = None,
    options: RenderOptions | None = None,
) -> Path:
    """Render a single scene segment (.mkv) from visual and audio assets.

    Args:
        assets: Scene assets.
        out_path: Target path for the output .mkv file.
        settings: Application settings.
        video_length: Optional target video length in seconds.
        options: Optional render options.

    Returns:
        Path to the rendered segment file.

    Raises:
        RenderError: If input files are missing or FFmpeg encoding fails.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate input assets
    vis = assets.visual or SceneVisual(kind="image", image=assets.image)
    if vis.image is not None and not vis.image.is_file():
        raise RenderError(
            f"Scene {assets.scene_id} image not found at {vis.image}.",
            hint="Ensure scene builder generated all slide images.",
        )
    if vis.video is not None and not vis.video.is_file():
        raise RenderError(
            f"Scene {assets.scene_id} video clip not found at {vis.video}.",
            hint="Ensure animation engine rendered all video clips.",
        )
    if assets.visual is None and video_length is None and not assets.audio.is_file():
        raise RenderError(
            f"Scene {assets.scene_id} audio not found at {assets.audio}.",
            hint="Ensure voice stage generated all narration audio files.",
        )

    cmd = build_segment_command(
        assets,
        out_path,
        settings,
        video_length=video_length,
        options=options,
    )

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
            hint="Check input image and video clip validity.",
        )

    if not out_path.is_file():
        raise RenderError(
            f"FFmpeg segment output was not created for scene {assets.scene_id} at {out_path}.",
            hint="Check disk space and write permissions.",
        )

    return out_path


def write_srt(
    assets: list[SceneAssets],
    out_path: Path,
    settings: Settings | None = None,
    options: RenderOptions | None = None,
) -> Path:
    """Create an SRT subtitle file aligned with scene narrations and pauses.

    Uses compute_timeline to derive narration start and duration timestamps.

    Args:
        assets: Ordered list of scene assets.
        out_path: Target path for the output .srt file.
        settings: Application settings. Defaults to get_settings().
        options: Rendering options.

    Returns:
        Path to the written .srt file.

    Raises:
        RenderError: If writing the SRT file fails.
    """
    resolved_settings = settings if settings is not None else get_settings()
    use_trans = (options.transitions and (len(assets) > 1)) if options is not None else False
    t_dur = options.transition_seconds if options is not None else 0.4

    timeline = compute_timeline(
        narration_seconds=[a.narration_seconds for a in assets],
        pause=resolved_settings.scene_pause_seconds,
        T=t_dur,
        transitions=use_trans,
    )

    cues = build_cues(
        narrations=[a.narration for a in assets],
        starts=timeline.narration_start,
        durations=[a.narration_seconds for a in assets],
    )

    return captions_write_srt(cues, out_path)


def render_video(
    assets: list[SceneAssets],
    out_path: Path,
    settings: Settings | None = None,
    options: RenderOptions | None = None,
    *,
    keep_temp: bool = False,
) -> Path:
    """Render a full MP4 demo video and companion subtitles from scene assets.

    Steps:
    A. Render silent video segments for each scene.
    B. Process and concatenate narration audio with lead silence and pauses.
    C. Join video with crossfade transitions (or concat demuxer if disabled).
    D. Burn caption overlays if requested.
    E. Mix optional background music and encode final MP4 with libx264 and aac.
    F. Validate stream properties and duration with ffprobe.

    Args:
        assets: Ordered list of SceneAssets for all scenes.
        out_path: Target path for the final MP4 video.
        settings: Application settings. Defaults to get_settings().
        options: RenderOptions parameters.
        keep_temp: Whether to retain intermediate segment files in _work folder.

    Returns:
        Path to the finished MP4 video file.

    Raises:
        RenderError: If rendering, concatenation, validation, or subtitles fail.
    """
    resolved_settings = settings if settings is not None else get_settings()
    is_v1_call = (options is None) and all(a.visual is None for a in assets)
    opts = options if options is not None else RenderOptions(transitions=not is_v1_call)
    if keep_temp:
        opts.keep_temp = True

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not assets:
        raise RenderError(
            "No scene assets provided to render_video.",
            hint="Provide at least one SceneAssets item to render.",
        )

    # Validate music file if provided
    if opts.music_path is not None:
        m_path = Path(opts.music_path)
        if m_path.suffix.lower() not in ALLOWED_MUSIC_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_MUSIC_EXTENSIONS))
            raise RenderError(
                f"Unsupported music file extension '{m_path.suffix}'.",
                hint=f"Allowed music extensions are: {allowed}.",
            )
        if not m_path.is_file():
            raise RenderError(
                f"Background music file not found at {m_path}.",
                hint="Check the music file path and ensure it exists.",
            )

    # Pre-flight check FFmpeg and capabilities
    check_ffmpeg()
    cap = capabilities()

    # Verify input assets exist
    for a in assets:
        vis = a.visual or SceneVisual(kind="image", image=a.image)
        if vis.image is not None and not vis.image.is_file():
            raise RenderError(
                f"Scene {a.scene_id} image not found at {vis.image}.",
                hint="Ensure scene builder generated all slide images.",
            )
        if vis.video is not None and not vis.video.is_file():
            raise RenderError(
                f"Scene {a.scene_id} video clip not found at {vis.video}.",
                hint="Ensure animation engine rendered all video clips.",
            )
        if not a.audio.is_file():
            raise RenderError(
                f"Scene {a.scene_id} audio not found at {a.audio}.",
                hint="Ensure voice stage generated all narration audio files.",
            )

    # Capability degradation: xfade missing -> disable transitions
    use_transitions = opts.transitions and (len(assets) > 1)
    if use_transitions and not cap.has_xfade:
        logger.warning(
            "Transitions requested but FFmpeg build lacks xfade filter; continuing with transitions disabled."
        )
        use_transitions = False

    # Compute timeline
    timeline = compute_timeline(
        narration_seconds=[a.narration_seconds for a in assets],
        pause=resolved_settings.scene_pause_seconds,
        T=opts.transition_seconds,
        transitions=use_transitions,
    )

    work_dir = out_path.parent / f"{out_path.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    segment_paths: list[Path] = []
    audio_paths: list[Path] = []

    try:
        # Step A: Render one silent video segment per scene
        for i, a in enumerate(assets):
            seg_path = work_dir / f"segment_{a.scene_id:02d}.mkv"
            t0 = time.monotonic()
            make_segment(
                a,
                seg_path,
                resolved_settings,
                video_length=timeline.video_len[i],
                options=opts,
            )
            elapsed = time.monotonic() - t0
            segment_paths.append(seg_path)
            logger.info(
                "Scene %d video segment rendered in %.2fs (length: %.2fs)",
                a.scene_id,
                elapsed,
                timeline.video_len[i],
            )

        # Step B: Audio processing
        pause_str = f"{resolved_settings.scene_pause_seconds:.3f}"
        for i, a in enumerate(assets):
            scene_audio_path = work_dir / f"scene_audio_{a.scene_id:02d}.wav"
            lead_ms = round(timeline.lead[i] * 1000)

            # Build audio filter: adelay if lead > 0, apad for pause, resample to 44100Hz stereo
            if lead_ms > 0:
                af_str = f"adelay={lead_ms}:all=1,apad=pad_dur={pause_str}"
            else:
                af_str = f"apad=pad_dur={pause_str}"

            aud_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(a.audio),
                "-af",
                af_str,
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                "-t",
                f"{timeline.audio_len[i]:.3f}",
                str(scene_audio_path),
            ]
            try:
                aud_res = subprocess.run(
                    aud_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300,
                )
            except subprocess.TimeoutExpired as exc:
                raise RenderError(
                    f"Audio processing timed out for scene {a.scene_id}.",
                    hint="Check system disk I/O.",
                ) from exc

            if aud_res.returncode != 0:
                err_tail = aud_res.stderr.strip()[-600:]
                raise RenderError(
                    f"FFmpeg audio processing failed for scene {a.scene_id}: {err_tail}",
                    hint="Check input audio validity.",
                )
            audio_paths.append(scene_audio_path)

        # Concatenate audio segments with concat demuxer into narration.wav
        concat_audio_file = work_dir / "concat_audio.txt"
        concat_audio_file.write_text(build_concat_list(audio_paths), encoding="utf-8")
        narration_wav = work_dir / "narration.wav"

        concat_aud_cmd = [
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
            str(concat_audio_file),
            "-c",
            "copy",
            str(narration_wav),
        ]
        res_aud_concat = subprocess.run(
            concat_aud_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if res_aud_concat.returncode != 0:
            raise RenderError(
                f"FFmpeg audio concatenation failed: {res_aud_concat.stderr.strip()[-600:]}",
                hint="Check intermediate audio files.",
            )

        # Validate narration.wav duration is within 0.05s of timeline total
        actual_audio_dur = probe_audio_duration(narration_wav)
        if abs(actual_audio_dur - timeline.total) > 0.05:
            logger.warning(
                "Narration audio length (%.2fs) differs from timeline total (%.2fs) by %.3fs",
                actual_audio_dur,
                timeline.total,
                abs(actual_audio_dur - timeline.total),
            )

        # Build Subtitles / Cues
        cues = build_cues(
            narrations=[a.narration for a in assets],
            starts=timeline.narration_start,
            durations=[a.narration_seconds for a in assets],
        )

        if opts.captions != "off":
            srt_path = out_path.with_suffix(".srt")
            captions_write_srt(cues, srt_path)

        caption_pngs: list[Path] = []
        if opts.captions == "burn" and cues:
            caption_pngs = render_caption_images(
                cues,
                work_dir,
                settings=resolved_settings,
                theme=opts.theme,
            )

        # Steps C, D, E: Final Video Encode
        n = len(segment_paths)
        if use_transitions:
            # Transitions ON: inputs 0..n-1 are video segments, n is narration.wav
            input_args: list[str] = []
            for sp in segment_paths:
                input_args.extend(["-i", str(sp)])
            input_args.extend(["-i", str(narration_wav)])
            narration_idx = n
            next_input_idx = n + 1

            music_idx: int | None = None
            if opts.music_path is not None:
                input_args.extend(["-stream_loop", "-1", "-i", str(opts.music_path)])
                music_idx = next_input_idx
                next_input_idx += 1

            caption_start_idx = next_input_idx
            for cpng in caption_pngs:
                input_args.extend(["-loop", "1", "-i", str(cpng)])

            # Build filters
            xfade_filter = build_xfade_graph(
                n,
                timeline.offsets,
                opts.transition_seconds,
                resolved_settings.video_fps,
            )
            v_curr = f"x{n - 1}"

            cap_filter, v_final = (
                build_caption_graph(
                    cues,
                    v_curr,
                    start_input_index=caption_start_idx,
                )
                if caption_pngs
                else ("", f"[{v_curr}]")
            )

            # Audio filter
            audio_filter = ""
            audio_map = f"{narration_idx}:a"
            if music_idx is not None:
                fade_start = max(0.0, timeline.total - 2.0)
                fade_dur = min(2.0, timeline.total)
                norm_flag = ":normalize=0" if cap.has_amix_normalize else ",volume=2"
                audio_filter = (
                    f"[{music_idx}:a]volume={opts.music_volume:.3f},"
                    f"afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f},"
                    f"atrim=0:{timeline.total:.3f}[bgm];"
                    f"[{narration_idx}:a][bgm]amix=inputs=2:duration=first:dropout_transition=0{norm_flag}[aout]"
                )
                audio_map = "[aout]"

            filter_parts = [xfade_filter]
            if cap_filter:
                filter_parts.append(cap_filter)
            if audio_filter:
                filter_parts.append(audio_filter)
            final_filter_str = ";".join(filter_parts)

            final_cmd = (
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                ]
                + input_args
                + [
                    "-filter_complex",
                    final_filter_str,
                    "-map",
                    v_final,
                    "-map",
                    audio_map,
                    "-c:v",
                    "libx264",
                    "-preset",
                    opts.preset,
                    "-crf",
                    str(opts.crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-t",
                    f"{timeline.total:.3f}",
                    str(out_path),
                ]
            )

        else:
            # Transitions OFF or single scene
            concat_video_file = work_dir / "concat_video.txt"
            concat_video_file.write_text(build_concat_list(segment_paths), encoding="utf-8")

            if not caption_pngs and opts.music_path is None:
                # Direct stream copy with concat demuxer
                final_cmd = [
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
                    str(concat_video_file),
                    "-i",
                    str(narration_wav),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-t",
                    f"{timeline.total:.3f}",
                    str(out_path),
                ]
            else:
                # Re-encode required for burning captions and/or music mixing
                input_args = [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_video_file),
                    "-i",
                    str(narration_wav),
                ]
                narration_idx = 1
                next_input_idx = 2

                music_idx = None
                if opts.music_path is not None:
                    input_args.extend(["-stream_loop", "-1", "-i", str(opts.music_path)])
                    music_idx = next_input_idx
                    next_input_idx += 1

                caption_start_idx = next_input_idx
                for cpng in caption_pngs:
                    input_args.extend(["-loop", "1", "-i", str(cpng)])

                cap_filter, v_final = (
                    build_caption_graph(
                        cues,
                        "0:v",
                        start_input_index=caption_start_idx,
                    )
                    if caption_pngs
                    else ("", "0:v")
                )

                audio_filter = ""
                audio_map = f"{narration_idx}:a"
                if music_idx is not None:
                    fade_start = max(0.0, timeline.total - 2.0)
                    fade_dur = min(2.0, timeline.total)
                    norm_flag = ":normalize=0" if cap.has_amix_normalize else ",volume=2"
                    audio_filter = (
                        f"[{music_idx}:a]volume={opts.music_volume:.3f},"
                        f"afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f},"
                        f"atrim=0:{timeline.total:.3f}[bgm];"
                        f"[{narration_idx}:a][bgm]amix=inputs=2:duration=first:dropout_transition=0{norm_flag}[aout]"
                    )
                    audio_map = "[aout]"

                filter_parts = []
                if cap_filter:
                    filter_parts.append(cap_filter)
                if audio_filter:
                    filter_parts.append(audio_filter)
                final_filter_str = ";".join(filter_parts)

                final_cmd = (
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                    ]
                    + input_args
                    + (["-filter_complex", final_filter_str] if final_filter_str else [])
                    + [
                        "-map",
                        v_final,
                        "-map",
                        audio_map,
                        "-c:v",
                        "libx264",
                        "-preset",
                        opts.preset,
                        "-crf",
                        str(opts.crf),
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-movflags",
                        "+faststart",
                        "-t",
                        f"{timeline.total:.3f}",
                        str(out_path),
                    ]
                )

        try:
            final_res = subprocess.run(
                final_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderError(
                "Final video encoding timed out after 900 seconds.",
                hint="Check system performance and video resolution.",
            ) from exc

        if final_res.returncode != 0:
            err_tail = final_res.stderr.strip()[-600:]
            raise RenderError(
                f"FFmpeg final video encoding failed (exit code {final_res.returncode}): {err_tail}",
                hint="Check filtergraph and input stream validity.",
            )

        # Step F: Validate rendered video
        info = probe_video(out_path)
        if not info["has_audio"]:
            raise RenderError(
                f"Rendered video {out_path} has no audio stream.",
                hint="Ensure input audio was valid and AAC encoding succeeded.",
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

        diff = abs(info["duration_seconds"] - timeline.total)
        if diff > 0.75:
            raise RenderError(
                f"Rendered video duration {info['duration_seconds']:.2f}s differs from expected "
                f"{timeline.total:.2f}s by {diff:.2f}s (allowed tolerance is 0.75s).",
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
        if not opts.keep_temp and work_dir.exists():
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
