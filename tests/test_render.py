"""Unit and integration tests for the repo2video video rendering stage."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import wave
import zlib
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from repo2video.config import Settings
from repo2video.errors import RenderError
from repo2video.render import (
    SceneAssets,
    build_concat_list,
    build_segment_command,
    check_ffmpeg,
    format_srt_timestamp,
    make_segment,
    probe_video,
    render_video,
    write_srt,
)

HAS_FFMPEG = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))
skip_if_no_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG, reason="FFmpeg and ffprobe binaries are required on PATH"
)


def make_tiny_png(
    path: Path,
    width: int = 16,
    height: int = 16,
    color: tuple[int, int, int] = (100, 150, 240),
) -> Path:
    """Create a minimal valid RGB PNG file using zlib and struct (no external libraries).

    Args:
        path: File path to save the PNG image.
        width: Image width in pixels.
        height: Image height in pixels.
        color: RGB tuple (0-255).

    Returns:
        The created file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    raw_scanline = b"\x00" + bytes(color) * width
    raw_data = raw_scanline * height
    compressed = zlib.compress(raw_data)
    idat = make_chunk(b"IDAT", compressed)
    iend = make_chunk(b"IEND", b"")

    path.write_bytes(sig + ihdr + idat + iend)
    return path


def make_dummy_wav(path: Path, duration_seconds: float, rate: int = 22050) -> Path:
    """Create a valid PCM WAV audio file containing silence.

    Args:
        path: File path to save the WAV file.
        duration_seconds: Audio duration in seconds.
        rate: Sample rate in Hz.

    Returns:
        The created file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(rate * duration_seconds)
    data = b"\x00" * (num_frames * 2)  # 16-bit mono PCM
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data)
    return path


# ==============================================================================
# 1. Tests with no FFmpeg required (pure functions & mocked calls)
# ==============================================================================


def test_scene_assets_dataclass(tmp_path: Path) -> None:
    """Verify SceneAssets is frozen and coerces paths properly."""
    img = tmp_path / "slide.png"
    aud = tmp_path / "voice.wav"
    assets = SceneAssets(
        scene_id=1,
        image=img,
        audio=aud,
        duration_seconds=5.25,
        narration="Hello world.",
    )
    assert assets.scene_id == 1
    assert assets.image == img
    assert assets.audio == aud
    assert assets.duration_seconds == 5.25
    assert assets.narration == "Hello world."

    # String path coercion
    assets_from_str = SceneAssets(
        scene_id=2,
        image=str(img),  # type: ignore[arg-type]
        audio=str(aud),  # type: ignore[arg-type]
        duration_seconds=3.0,
        narration="Test",
    )
    assert isinstance(assets_from_str.image, Path)
    assert isinstance(assets_from_str.audio, Path)

    # Immutability
    with pytest.raises(FrozenInstanceError):
        assets.scene_id = 99  # type: ignore[misc]


def test_build_segment_command_flags_and_rounding(tmp_path: Path) -> None:
    """Verify build_segment_command constructs all flags, rounds -t, and handles spaces."""
    img_with_space = tmp_path / "my folder" / "slide 01.png"
    aud_with_space = tmp_path / "my folder" / "voice 01.wav"
    out_mkv = tmp_path / "output segments" / "scene 01.mkv"

    assets = SceneAssets(
        scene_id=1,
        image=img_with_space,
        audio=aud_with_space,
        duration_seconds=2.3456,
        narration="Checking timing rounding.",
    )
    settings = Settings(
        video_width=1280,
        video_height=720,
        video_fps=25,
        scene_pause_seconds=0.6,
    )

    cmd = build_segment_command(assets, out_mkv, settings)

    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd
    assert "-hide_banner" in cmd
    assert cmd[cmd.index("-loglevel") + 1] == "error"
    assert cmd[cmd.index("-loop") + 1] == "1"
    assert cmd[cmd.index("-framerate") + 1] == "25"
    assert cmd[cmd.index("-i") + 1] == str(img_with_space)
    # Check that paths with spaces remain single argument elements
    assert str(img_with_space) in cmd
    assert str(aud_with_space) in cmd
    assert str(out_mkv) in cmd

    # Check pause padding and total duration rounding (2.3456 + 0.6 = 2.9456 -> 2.946)
    assert cmd[cmd.index("-af") + 1] == "apad=pad_dur=0.600"
    assert cmd[cmd.index("-t") + 1] == "2.946"

    # Check codecs and video filters
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-preset") + 1] == "medium"
    assert cmd[cmd.index("-crf") + 1] == "20"
    assert cmd[cmd.index("-tune") + 1] == "stillimage"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-r") + 1] == "25"
    assert cmd[cmd.index("-vf") + 1] == "scale=1280:720:flags=lanczos,setsar=1"
    assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
    assert cmd[cmd.index("-ar") + 1] == "44100"
    assert cmd[cmd.index("-ac") + 1] == "2"


def test_build_concat_list_escaping_and_order(tmp_path: Path) -> None:
    """Verify build_concat_list outputs forward slashes, quote escaping, and correct ordering."""
    p1 = tmp_path / "segment_01.mkv"
    p2 = tmp_path / "O'Reilly" / "segment_02.mkv"
    p3 = tmp_path / "segment 03.mkv"

    content = build_concat_list([p1, p2, p3])
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    assert len(lines) == 3
    # Check forward slashes
    assert p1.resolve().as_posix() in lines[0]
    # Check single quote escaping: ' replaced with '\''
    assert r"O'\''Reilly" in lines[1]
    # Check paths with spaces
    assert p3.resolve().as_posix() in lines[2]
    # Check format: file 'path'
    assert lines[0].startswith("file '") and lines[0].endswith("'")
    assert lines[1].startswith("file '") and lines[1].endswith("'")
    assert lines[2].startswith("file '") and lines[2].endswith("'")


def test_write_srt_cue_timing_and_splitting(tmp_path: Path) -> None:
    """Verify write_srt generates valid cue timestamps, 42-char lines, and non-ASCII text."""
    assets = [
        SceneAssets(
            scene_id=1,
            image=tmp_path / "img1.png",
            audio=tmp_path / "aud1.wav",
            duration_seconds=3.0,
            narration="Welcome to repo2video. This short sentence fits easily.",
        ),
        SceneAssets(
            scene_id=2,
            image=tmp_path / "img2.png",
            audio=tmp_path / "aud2.wav",
            duration_seconds=6.0,
            narration=(
                "This is a longer narration that will definitely exceed forty-two characters "
                "per line and need to be wrapped into multiple cues with proportional timing."
            ),
        ),
        SceneAssets(
            scene_id=3,
            image=tmp_path / "img3.png",
            audio=tmp_path / "aud3.wav",
            duration_seconds=2.5,
            narration="Développé avec succès! Visitez le site web officiel dès maintenant.",
        ),
    ]

    settings = Settings(scene_pause_seconds=0.5)
    srt_file = tmp_path / "subtitles.srt"
    write_srt(assets, srt_file, settings=settings)

    assert srt_file.is_file()
    content = srt_file.read_text(encoding="utf-8")
    assert "Développé avec succès!" in content

    blocks = content.strip().split("\n\n")
    # Multiple cues should be created
    assert len(blocks) >= 3

    # Check numbering starts at 1
    assert blocks[0].startswith("1\n")
    assert blocks[1].startswith("2\n")

    # Check line lengths: each line of subtitle text must be <= 42 chars
    for block in blocks:
        lines = block.splitlines()
        assert len(lines) >= 3  # id, timestamp, and 1 or 2 text lines
        text_lines = lines[2:]
        assert len(text_lines) <= 2
        for tline in text_lines:
            assert len(tline) <= 42

    # Check scene 1 starts at 00:00:00,000
    assert "00:00:00,000 -->" in blocks[0]

    # Scene 2 must start after scene 1 duration (3.0s) + pause (0.5s) = 3.5s -> 00:00:03,500
    # Find cue corresponding to scene 2 start
    assert any("00:00:03,500 -->" in b for b in blocks)

    # Format timestamp helper checks
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(65.123) == "00:01:05,123"
    assert format_srt_timestamp(3600.0) == "01:00:00,000"


def test_check_ffmpeg_missing_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify check_ffmpeg raises RenderError with platform installation hints when binaries missing."""

    def fake_which(name: str) -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", fake_which)

    with pytest.raises(RenderError) as exc_info:
        check_ffmpeg()
    assert "FFmpeg is not installed or not on PATH" in exc_info.value.message
    hint = exc_info.value.hint or ""
    assert "winget install ffmpeg" in hint
    assert "brew install ffmpeg" in hint
    assert "sudo apt install ffmpeg" in hint


def test_check_ffmpeg_missing_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify check_ffmpeg raises RenderError when libx264 or aac are absent."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "-encoders" in cmd:
            # Missing libx264
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="A....D aac AAC audio\n")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ffmpeg version 6.0")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RenderError) as exc_info:
        check_ffmpeg()
    assert "missing required encoder(s): libx264" in exc_info.value.message


def test_make_segment_missing_input_files(tmp_path: Path) -> None:
    """Verify make_segment raises RenderError naming the path and scene id before calling FFmpeg."""
    missing_img = tmp_path / "missing_slide.png"
    valid_aud = tmp_path / "voice.wav"
    valid_aud.touch()

    assets = SceneAssets(
        scene_id=4,
        image=missing_img,
        audio=valid_aud,
        duration_seconds=2.0,
        narration="Testing missing image.",
    )
    settings = Settings()

    with pytest.raises(RenderError) as exc_info:
        make_segment(assets, tmp_path / "out.mkv", settings)
    assert "Scene 4 image not found" in exc_info.value.message
    assert str(missing_img) in exc_info.value.message

    # Missing audio file
    valid_img = tmp_path / "slide.png"
    valid_img.touch()
    missing_aud = tmp_path / "missing_voice.wav"
    assets_missing_aud = SceneAssets(
        scene_id=7,
        image=valid_img,
        audio=missing_aud,
        duration_seconds=2.0,
        narration="Testing missing audio.",
    )

    with pytest.raises(RenderError) as exc_info:
        make_segment(assets_missing_aud, tmp_path / "out.mkv", settings)
    assert "Scene 7 audio not found" in exc_info.value.message
    assert str(missing_aud) in exc_info.value.message


def test_make_segment_ffmpeg_failure_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify make_segment includes scene id and last 600 chars of stderr on failure."""
    img = tmp_path / "slide.png"
    img.touch()
    aud = tmp_path / "voice.wav"
    aud.touch()
    out_mkv = tmp_path / "out.mkv"

    err_text = "Detailed FFmpeg failure: " + ("e" * 700) + "end_of_err"

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=err_text)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assets = SceneAssets(
        scene_id=3,
        image=img,
        audio=aud,
        duration_seconds=3.0,
        narration="Testing failure report.",
    )

    with pytest.raises(RenderError) as exc_info:
        make_segment(assets, out_mkv, Settings())

    assert "scene 3" in exc_info.value.message
    assert "end_of_err" in exc_info.value.message
    assert len(exc_info.value.message) < 800


def test_probe_video_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify probe_video parses json streams and format properly."""
    dummy_file = tmp_path / "mock.mp4"
    dummy_file.touch()

    ffprobe_json = {
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "duration": "12.500"},
            {"codec_type": "audio", "duration": "12.500"},
        ],
        "format": {"duration": "12.500", "size": "4500000"},
    }

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout=json.dumps(ffprobe_json), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    info = probe_video(dummy_file)
    assert info["duration_seconds"] == 12.5
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["has_audio"] is True


# ==============================================================================
# 2. Integration tests with real FFmpeg execution
# ==============================================================================


@skip_if_no_ffmpeg
@pytest.mark.ffmpeg
def test_render_video_real_ffmpeg(tmp_path: Path) -> None:
    """Verify render_video produces valid MP4, matching duration, audio stream, and SRT."""
    settings = Settings(
        video_width=640,
        video_height=360,
        video_fps=24,
        scene_pause_seconds=0.5,
    )

    # 3 scenes with 1.0s, 1.5s, 2.0s audio at 22050 Hz mono
    durations = [1.0, 1.5, 2.0]
    assets: list[SceneAssets] = []

    for i, dur in enumerate(durations, start=1):
        png_path = make_tiny_png(tmp_path / f"slide_{i}.png", width=64, height=64)
        wav_path = make_dummy_wav(tmp_path / f"voice_{i}.wav", duration_seconds=dur, rate=22050)
        assets.append(
            SceneAssets(
                scene_id=i,
                image=png_path,
                audio=wav_path,
                duration_seconds=dur,
                narration=f"Scene {i} narration text for real FFmpeg integration test.",
            )
        )

    out_mp4 = tmp_path / "output_video.mp4"
    rendered_path = render_video(assets, out_mp4, settings=settings, keep_temp=False)

    assert rendered_path == out_mp4
    assert out_mp4.is_file()
    assert out_mp4.stat().st_size > 5000  # Non-trivial MP4 file size

    # Probe the generated MP4
    info = probe_video(out_mp4)
    assert info["has_audio"] is True
    assert info["width"] == 640
    assert info["height"] == 360

    # Expected duration: 1.0 + 1.5 + 2.0 + 3 * 0.5 = 6.0 seconds
    expected_dur = 4.5 + (3 * 0.5)
    assert abs(info["duration_seconds"] - expected_dur) <= 0.75

    # Check companion SRT file
    srt_path = out_mp4.with_suffix(".srt")
    assert srt_path.is_file()
    srt_text = srt_path.read_text(encoding="utf-8")
    assert "Scene 1 narration" in srt_text
    assert "Scene 3 narration" in srt_text

    # Temporary work folder should be deleted
    work_folder = tmp_path / f"{out_mp4.name}_work"
    assert not work_folder.exists()


@skip_if_no_ffmpeg
@pytest.mark.ffmpeg
def test_render_video_folder_with_spaces(tmp_path: Path) -> None:
    """Verify render_video handles directories with spaces and non-ASCII characters."""
    folder_with_space = tmp_path / "Project Demo Video 2026 (Étude)"
    folder_with_space.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        video_width=640,
        video_height=360,
        video_fps=24,
        scene_pause_seconds=0.4,
    )

    png_path = make_tiny_png(folder_with_space / "slide 1.png")
    wav_path = make_dummy_wav(folder_with_space / "audio 1.wav", duration_seconds=1.0)

    assets = [
        SceneAssets(
            scene_id=1,
            image=png_path,
            audio=wav_path,
            duration_seconds=1.0,
            narration="Narration with path spaces test.",
        )
    ]

    out_mp4 = folder_with_space / "demo final.mp4"
    rendered = render_video(assets, out_mp4, settings=settings)

    assert rendered.is_file()
    info = probe_video(out_mp4)
    assert info["has_audio"] is True
    assert info["width"] == 640
    assert info["height"] == 360
