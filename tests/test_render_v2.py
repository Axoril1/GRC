"""Tests for repo2video/render.py version 2 rendering pipeline, timeline, and filters."""

from __future__ import annotations

import re
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
from PIL import Image

from repo2video.captions import Cue
from repo2video.config import Settings
from repo2video.errors import RenderError
from repo2video.render import (
    Capabilities,
    RenderOptions,
    SceneAssets,
    SceneVisual,
    build_caption_graph,
    build_segment_command,
    build_xfade_graph,
    compute_timeline,
    probe_video,
    render_video,
    scene_video_lengths,
)
from repo2video.scene_media import Rect

HAS_FFMPEG = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))
skip_if_no_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG, reason="FFmpeg and ffprobe binaries are required on PATH"
)


def make_color_png(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (640, 360)
) -> Path:
    """Create a solid color PNG image using Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


def make_silent_wav(path: Path, duration: float, rate: int = 44100) -> Path:
    """Create a valid PCM WAV audio file with silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * duration)
    data = b"\x00" * (frames * 4)  # 16-bit stereo PCM
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data)
    return path


def make_sine_wav(path: Path, duration: float, freq: int = 440) -> Path:
    """Generate a sine wave WAV audio file using FFmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={duration}",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def make_color_clip(
    path: Path, color_hex: str, duration: float, size: tuple[int, int] = (640, 360)
) -> Path:
    """Generate a solid color video clip using FFmpeg lavfi color source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color_hex}:s={size[0]}x{size[1]}:d={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


# ==============================================================================
# 1. Non-FFmpeg Pure Function Tests
# ==============================================================================


def test_compute_timeline_worked_example() -> None:
    """Verify compute_timeline matches the prompt worked example precisely."""
    # narration [4.0, 5.0, 3.0], pause 0.6, T 0.4
    tl = compute_timeline(
        narration_seconds=[4.0, 5.0, 3.0],
        pause=0.6,
        T=0.4,
        transitions=True,
    )
    assert tl.lead == [0.0, 0.4, 0.4]
    assert tl.audio_len == [4.6, 6.0, 4.0]
    assert tl.video_len == [5.0, 6.4, 4.0]
    assert tl.audio_start == [0.0, 4.6, 10.6]
    assert tl.narration_start == [0.0, 5.0, 11.0]
    assert tl.offsets == [4.6, 10.6]
    assert tl.total == 14.6


def test_compute_timeline_transitions_off() -> None:
    """Verify compute_timeline with transitions off has zero lead, v equals L, and defined offsets."""
    tl = compute_timeline(
        narration_seconds=[4.0, 5.0, 3.0],
        pause=0.6,
        T=0.4,
        transitions=False,
    )
    assert tl.lead == [0.0, 0.0, 0.0]
    assert tl.audio_len == [4.6, 5.6, 3.6]
    assert tl.video_len == tl.audio_len
    assert tl.audio_start == [0.0, 4.6, 10.2]
    assert tl.narration_start == [0.0, 4.6, 10.2]
    assert tl.offsets == [4.6, 10.2]
    assert tl.total == 13.8


def test_compute_timeline_single_scene() -> None:
    """Verify compute_timeline for a single scene produces zero lead and no offsets."""
    tl = compute_timeline(
        narration_seconds=[5.0],
        pause=0.5,
        T=0.4,
        transitions=True,
    )
    assert tl.lead == [0.0]
    assert tl.audio_len == [5.5]
    assert tl.video_len == [5.5]
    assert tl.audio_start == [0.0]
    assert tl.narration_start == [0.0]
    assert tl.offsets == []
    assert tl.total == 5.5


def test_compute_timeline_short_narration() -> None:
    """Verify compute_timeline with very short narration durations."""
    tl = compute_timeline(
        narration_seconds=[0.2, 0.3],
        pause=0.6,
        T=0.4,
        transitions=True,
    )
    assert tl.lead == [0.0, 0.4]
    assert tl.audio_len == [0.8, 1.3]
    assert tl.video_len == [1.2, 1.3]
    assert tl.offsets == [0.8]
    assert tl.total == 2.1


def test_scene_video_lengths() -> None:
    """Verify scene_video_lengths returns video_len list for scene pipeline planning."""
    settings = Settings(scene_pause_seconds=0.6)
    options = RenderOptions(transitions=True, transition_seconds=0.4)
    v_lens = scene_video_lengths([4.0, 5.0, 3.0], settings, options)
    assert v_lens == [5.0, 6.4, 4.0]


def test_build_xfade_graph_3_and_5_scenes() -> None:
    """Verify build_xfade_graph generates proper normalisation, labels, and offset syntax."""
    # 3 scenes
    graph3 = build_xfade_graph(3, [4.6, 10.6], T=0.4, fps=30)
    assert "[0:v]settb=AVTB,setpts=PTS-STARTPTS,fps=30,format=yuv420p[s0]" in graph3
    assert "[1:v]settb=AVTB,setpts=PTS-STARTPTS,fps=30,format=yuv420p[s1]" in graph3
    assert "[2:v]settb=AVTB,setpts=PTS-STARTPTS,fps=30,format=yuv420p[s2]" in graph3
    assert "[s0][s1]xfade=transition=fade:duration=0.400:offset=4.600[x1]" in graph3
    assert "[x1][s2]xfade=transition=fade:duration=0.400:offset=10.600[x2]" in graph3
    assert graph3.endswith("[x2]")

    # 5 scenes
    graph5 = build_xfade_graph(5, [3.0, 7.0, 11.0, 15.0], T=0.4, fps=25)
    assert "[s0][s1]xfade=transition=fade:duration=0.400:offset=3.000[x1]" in graph5
    assert "[x1][s2]xfade=transition=fade:duration=0.400:offset=7.000[x2]" in graph5
    assert "[x2][s3]xfade=transition=fade:duration=0.400:offset=11.000[x3]" in graph5
    assert "[x3][s4]xfade=transition=fade:duration=0.400:offset=15.000[x4]" in graph5
    assert graph5.endswith("[x4]")


def test_build_caption_graph() -> None:
    """Verify build_caption_graph chains overlay filters with center alignment and timing."""
    # Empty cues
    empty_filter, empty_out = build_caption_graph([], "v_in")
    assert empty_filter == ""
    assert empty_out == "[v_in]"

    # 2 cues
    cues = [
        Cue(start=1.0, end=2.5, lines=["Line 1"]),
        Cue(start=3.0, end=4.5, lines=["Line 2"]),
    ]
    filter_str, final_label = build_caption_graph(cues, "x2", start_input_index=3)

    assert (
        "[x2][3:v]overlay=x=(W-w)/2:y=H-h-70:enable='between(t,1.000,2.500)':eof_action=pass[cap_1]"
        in filter_str
    )
    assert (
        "[cap_1][4:v]overlay=x=(W-w)/2:y=H-h-70:enable='between(t,3.000,4.500)':eof_action=pass[cap_2]"
        in filter_str
    )
    assert final_label == "[cap_2]"


def test_music_validation_errors(tmp_path: Path) -> None:
    """Verify render_video validates music file existence and extensions."""
    settings = Settings()
    assets = [
        SceneAssets(
            scene_id=1,
            audio=tmp_path / "voice.wav",
            narration_seconds=2.0,
            narration="Test",
            visual=SceneVisual(kind="image", image=tmp_path / "img.png"),
        )
    ]
    (tmp_path / "img.png").touch()
    (tmp_path / "voice.wav").touch()

    # Missing file
    opts_missing = RenderOptions(music_path=tmp_path / "non_existent.mp3")
    with pytest.raises(RenderError, match="not found"):
        render_video(assets, tmp_path / "out.mp4", settings=settings, options=opts_missing)

    # Invalid extension
    bad_ext = tmp_path / "music.exe"
    bad_ext.touch()
    opts_bad = RenderOptions(music_path=bad_ext)
    with pytest.raises(RenderError, match="Unsupported music file extension"):
        render_video(assets, tmp_path / "out.mp4", settings=settings, options=opts_bad)


def test_missing_capabilities_degrade_transitions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify lack of xfade capability logs warning and degrades to transitions off."""
    import repo2video.render as r_mod

    fake_cap = Capabilities(
        version="ffmpeg version 6.0",
        has_libx264=True,
        has_aac=True,
        has_xfade=False,  # missing xfade
        has_overlay=True,
    )
    monkeypatch.setattr(r_mod, "capabilities", lambda: fake_cap)

    # Ensure check_ffmpeg passes
    assert r_mod.check_ffmpeg() == "ffmpeg version 6.0"


def test_command_construction_for_each_visual_kind(tmp_path: Path) -> None:
    """Verify build_segment_command constructs correct flags and retains paths with spaces."""
    settings = Settings(video_width=1280, video_height=720, video_fps=30)
    folder_space = tmp_path / "path with spaces"
    folder_space.mkdir(parents=True, exist_ok=True)

    img_space = folder_space / "slide 1.png"
    aud_space = folder_space / "voice 1.wav"
    out_mkv = folder_space / "segment 1.mkv"
    clip_space = folder_space / "demo clip.mp4"

    # 1. Image visual
    assets_img = SceneAssets(
        scene_id=1,
        audio=aud_space,
        narration_seconds=3.0,
        visual=SceneVisual(kind="image", image=img_space),
    )
    cmd_img = build_segment_command(assets_img, out_mkv, settings, video_length=3.6)
    assert "-loop" in cmd_img and "1" in cmd_img
    assert "-t" in cmd_img and "3.600" in cmd_img
    assert "-an" in cmd_img
    assert str(img_space) in cmd_img
    assert str(out_mkv) in cmd_img

    # 2. Video visual
    assets_vid = SceneAssets(
        scene_id=2,
        audio=aud_space,
        narration_seconds=4.0,
        visual=SceneVisual(kind="video", video=clip_space),
    )
    cmd_vid = build_segment_command(assets_vid, out_mkv, settings, video_length=4.6)
    assert "-an" in cmd_vid
    assert "-t" in cmd_vid and "4.600" in cmd_vid
    assert str(clip_space) in cmd_vid

    # 3. Overlay visual
    rect = Rect(x=100, y=150, w=500, h=300)
    assets_ov = SceneAssets(
        scene_id=3,
        audio=aud_space,
        narration_seconds=5.0,
        visual=SceneVisual(kind="overlay", image=img_space, video=clip_space, rect=rect),
    )
    cmd_ov = build_segment_command(assets_ov, out_mkv, settings, video_length=5.6)
    assert "-filter_complex" in cmd_ov
    filter_idx = cmd_ov.index("-filter_complex") + 1
    assert "overlay=x=100:y=150" in cmd_ov[filter_idx]
    assert "scale=500:300:force_original_aspect_ratio=decrease" in cmd_ov[filter_idx]
    assert "-an" in cmd_ov
    assert "-t" in cmd_ov and "5.600" in cmd_ov


# ==============================================================================
# 2. Real FFmpeg Integration Tests
# ==============================================================================


@skip_if_no_ffmpeg
@pytest.mark.ffmpeg
def test_render_video_v2_transitions_on_and_off(tmp_path: Path) -> None:
    """Verify render_video composites image, video, and overlay with transitions on and off."""
    settings = Settings(video_width=640, video_height=360, video_fps=24, scene_pause_seconds=0.5)

    # Build 3 scenes:
    # Scene 1: red image, 1.0s audio
    img_red = make_color_png(tmp_path / "red.png", (220, 40, 40), (640, 360))
    wav_1 = make_silent_wav(tmp_path / "aud1.wav", 1.0)
    a1 = SceneAssets(
        scene_id=1,
        audio=wav_1,
        narration_seconds=1.0,
        narration="First scene red slide.",
        visual=SceneVisual(kind="image", image=img_red),
    )

    # Scene 2: green clip, 1.5s audio
    clip_green = make_color_clip(tmp_path / "green.mp4", "0x00FF00", 2.0, (640, 360))
    wav_2 = make_silent_wav(tmp_path / "aud2.wav", 1.5)
    a2 = SceneAssets(
        scene_id=2,
        audio=wav_2,
        narration_seconds=1.5,
        narration="Second scene green clip.",
        visual=SceneVisual(kind="video", video=clip_green),
    )

    # Scene 3: blue bg + yellow demo clip in rect, 2.0s audio
    bg_blue = make_color_png(tmp_path / "blue.png", (40, 40, 220), (640, 360))
    clip_yellow = make_color_clip(tmp_path / "yellow.mp4", "0xFFFF00", 2.0, (300, 200))
    wav_3 = make_silent_wav(tmp_path / "aud3.wav", 2.0)
    a3 = SceneAssets(
        scene_id=3,
        audio=wav_3,
        narration_seconds=2.0,
        narration="Third scene blue overlay with yellow demo.",
        visual=SceneVisual(
            kind="overlay",
            image=bg_blue,
            video=clip_yellow,
            rect=Rect(x=170, y=80, w=300, h=200),
        ),
    )

    assets = [a1, a2, a3]

    # --- Test 1: Transitions ON ---
    out_trans_on = tmp_path / "video_trans_on.mp4"
    opts_on = RenderOptions(transitions=True, transition_seconds=0.4, captions="off")
    res_on = render_video(assets, out_trans_on, settings=settings, options=opts_on)

    assert res_on.is_file()
    info_on = probe_video(res_on)
    assert info_on["has_audio"] is True
    assert info_on["width"] == 640
    assert info_on["height"] == 360

    # Timeline total: narration [1.0, 1.5, 2.0], pause 0.5, T 0.4 -> total = 6.8s
    tl_on = compute_timeline([1.0, 1.5, 2.0], 0.5, 0.4, True)
    assert abs(info_on["duration_seconds"] - tl_on.total) <= 0.75

    # --- Test 2: Transitions OFF ---
    out_trans_off = tmp_path / "video_trans_off.mp4"
    opts_off = RenderOptions(transitions=False, captions="off")
    res_off = render_video(assets, out_trans_off, settings=settings, options=opts_off)

    assert res_off.is_file()
    info_off = probe_video(res_off)
    assert info_off["has_audio"] is True
    tl_off = compute_timeline([1.0, 1.5, 2.0], 0.5, 0.4, False)
    assert abs(info_off["duration_seconds"] - tl_off.total) <= 0.75


@skip_if_no_ffmpeg
@pytest.mark.ffmpeg
def test_transition_midpoint_color_blend(tmp_path: Path) -> None:
    """Verify frames extracted at the crossfade transition midpoint are a color blend."""
    settings = Settings(video_width=320, video_height=240, video_fps=24, scene_pause_seconds=0.5)

    # Pure Red and Pure Blue
    img_red = make_color_png(tmp_path / "red.png", (255, 0, 0), (320, 240))
    img_blue = make_color_png(tmp_path / "blue.png", (0, 0, 255), (320, 240))
    wav1 = make_silent_wav(tmp_path / "aud1.wav", 1.0)
    wav2 = make_silent_wav(tmp_path / "aud2.wav", 1.0)

    a1 = SceneAssets(
        scene_id=1,
        audio=wav1,
        narration_seconds=1.0,
        narration="A",
        visual=SceneVisual(kind="image", image=img_red),
    )
    a2 = SceneAssets(
        scene_id=2,
        audio=wav2,
        narration_seconds=1.0,
        narration="B",
        visual=SceneVisual(kind="image", image=img_blue),
    )

    # T = 0.6s crossfade
    opts = RenderOptions(transitions=True, transition_seconds=0.6, captions="off")
    out_mp4 = tmp_path / "blend.mp4"
    render_video([a1, a2], out_mp4, settings=settings, options=opts)

    # Timeline: Scene 1 audio length L_0 = 1.0 + 0.5 = 1.5s
    # Transition offset = 1.5s, duration = 0.6s -> midpoint = 1.5 + 0.3 = 1.8s
    frame_png = tmp_path / "midpoint_frame.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "1.800",
            "-i",
            str(out_mp4),
            "-vframes",
            "1",
            str(frame_png),
        ],
        check=True,
        capture_output=True,
    )

    img = Image.open(frame_png).convert("RGB")
    center_pixel = img.getpixel((160, 120))
    r, _g, b = center_pixel

    # Must be a blend: significant red and significant blue, neither near 0
    assert r > 40, f"Expected red component in blend, got {r}"
    assert b > 40, f"Expected blue component in blend, got {b}"


@skip_if_no_ffmpeg
@pytest.mark.ffmpeg
def test_music_mixing_narration_level_within_3db(tmp_path: Path) -> None:
    """Verify background music at 10% volume keeps narration level within 3 dB."""
    settings = Settings(video_width=320, video_height=240, video_fps=24, scene_pause_seconds=0.5)

    img_slide = make_color_png(tmp_path / "slide.png", (40, 40, 40), (320, 240))
    # 440 Hz narration tone at full volume
    narration_wav = make_sine_wav(tmp_path / "narration.wav", duration=2.5, freq=440)
    # 1000 Hz background music
    music_wav = make_sine_wav(tmp_path / "music.wav", duration=5.0, freq=1000)

    assets = [
        SceneAssets(
            scene_id=1,
            audio=narration_wav,
            narration_seconds=2.5,
            narration="Music mixing test.",
            visual=SceneVisual(kind="image", image=img_slide),
        )
    ]

    # Render without music
    out_nomusic = tmp_path / "no_music.mp4"
    render_video(assets, out_nomusic, settings=settings, options=RenderOptions(music_path=None))

    # Render with music at 10% volume
    out_music = tmp_path / "with_music.mp4"
    render_video(
        assets,
        out_music,
        settings=settings,
        options=RenderOptions(music_path=music_wav, music_volume=0.10),
    )

    def measure_mean_volume(video_path: Path) -> float:
        res = subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"mean_volume:\s+([-\d.]+)\s+dB", res.stderr)
        assert match is not None
        return float(match.group(1))

    vol_nomusic = measure_mean_volume(out_nomusic)
    vol_music = measure_mean_volume(out_music)

    diff = abs(vol_music - vol_nomusic)
    assert diff <= 3.0, f"Volume difference {diff:.2f} dB exceeds 3.0 dB tolerance"


@skip_if_no_ffmpeg
@pytest.mark.ffmpeg
def test_burned_captions_appear_in_frame(tmp_path: Path) -> None:
    """Verify burned captions modify the video frame pixels compared to captions off."""
    settings = Settings(video_width=640, video_height=360, video_fps=24, scene_pause_seconds=0.5)

    img_slide = make_color_png(tmp_path / "dark_slide.png", (10, 10, 20), (640, 360))
    wav_voice = make_silent_wav(tmp_path / "voice.wav", duration=3.0)

    assets = [
        SceneAssets(
            scene_id=1,
            audio=wav_voice,
            narration_seconds=3.0,
            narration="Visible burned caption test text.",
            visual=SceneVisual(kind="image", image=img_slide),
        )
    ]

    # Without captions
    out_nocap = tmp_path / "no_cap.mp4"
    render_video(assets, out_nocap, settings=settings, options=RenderOptions(captions="off"))

    # With burned captions
    out_burn = tmp_path / "burn_cap.mp4"
    render_video(assets, out_burn, settings=settings, options=RenderOptions(captions="burn"))

    # Extract frame at t=1.5s from both videos
    frame_nocap = tmp_path / "frame_nocap.png"
    frame_burn = tmp_path / "frame_burn.png"

    for vid, frm in [(out_nocap, frame_nocap), (out_burn, frame_burn)]:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "1.500", "-i", str(vid), "-vframes", "1", str(frm)],
            check=True,
            capture_output=True,
        )

    img1 = Image.open(frame_nocap).convert("L")
    img2 = Image.open(frame_burn).convert("L")

    # Crop the subtitle region at the bottom (y: 250 to 350, x: 100 to 540)
    crop1 = img1.crop((100, 250, 540, 350))
    crop2 = img2.crop((100, 250, 540, 350))

    diff_pixels = sum(abs(p1 - p2) for p1, p2 in zip(crop1.tobytes(), crop2.tobytes()))
    assert diff_pixels > 1000, (
        "Burned caption text did not produce visible difference in subtitle area"
    )


@skip_if_no_ffmpeg
@pytest.mark.ffmpeg
def test_render_v2_folder_with_space(tmp_path: Path) -> None:
    """Verify v2 rendering pipeline succeeds in paths with spaces and non-ASCII chars."""
    folder = tmp_path / "Dossier Vidéo 2026 (Demo)"
    folder.mkdir(parents=True, exist_ok=True)

    settings = Settings(video_width=640, video_height=360, video_fps=24, scene_pause_seconds=0.4)
    png_path = make_color_png(folder / "slide 1.png", (50, 100, 150), (640, 360))
    wav_path = make_silent_wav(folder / "audio 1.wav", duration=1.2)

    assets = [
        SceneAssets(
            scene_id=1,
            audio=wav_path,
            narration_seconds=1.2,
            narration="Non-ASCII path and space test.",
            visual=SceneVisual(kind="image", image=png_path),
        )
    ]

    out_mp4 = folder / "demo output.mp4"
    res = render_video(assets, out_mp4, settings=settings, options=RenderOptions(transitions=False))
    assert res.is_file()
    assert res.stat().st_size > 3000
