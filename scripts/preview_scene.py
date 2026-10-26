"""CLI script to render animated scene previews (PNG and MP4) in dark and light themes."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from GRC.anim import AnimationRenderer
from GRC.config import get_settings
from GRC.media import MediaAsset
from GRC.models import (
    ArchitectureSpec,
    EdgeSpec,
    FocusRange,
    KeyFile,
    NodeSpec,
    RepoData,
    Scene,
    TerminalSession,
)
from GRC.scene_arch import build_arch_html
from GRC.scene_code import build_code_html
from GRC.scene_media import build_media_html
from GRC.scene_stats import build_stats_html
from GRC.scene_terminal import build_terminal_html


def make_dummy_media_image(path: Path) -> Path:
    """Generate a high-contrast mock screenshot image for media scene preview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1600, 760), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Grid lines
    for x in range(0, 1600, 80):
        draw.line([(x, 0), (x, 760)], fill=(30, 41, 59), width=1)
    for y in range(0, 760, 80):
        draw.line([(0, y), (1600, y)], fill=(30, 41, 59), width=1)

    # Decorative mock UI panels
    draw.rounded_rectangle(
        [100, 100, 700, 660], radius=16, fill=(30, 41, 59), outline=(59, 130, 246), width=2
    )
    draw.rounded_rectangle(
        [750, 100, 1500, 360], radius=16, fill=(30, 41, 59), outline=(16, 185, 129), width=2
    )
    draw.rounded_rectangle(
        [750, 400, 1500, 660], radius=16, fill=(30, 41, 59), outline=(236, 72, 153), width=2
    )

    img.save(path)
    return path


def generate_preview(
    scene_type: str,
    theme: str,
    t_snapshot: float,
    out_dir: Path,
    render_video: bool = True,
) -> None:
    """Render a scene to PNG snapshot and MP4 video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    duration = 6.0
    accent = "#3b82f6"
    repo_label = "Axoril1/repo2video"

    html: str
    if scene_type == "code":
        # 40-line snippet with HTML tags and focus ranges
        code_lines = [
            "#!/usr/bin/env python3",
            '"""Main pipeline execution module for repository analysis."""',
            "",
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "# Safe script tag inside code: must NOT execute",
            "DANGEROUS_SNIPPET = '<script>alert(\"XSS vulnerable!\")</script>'",
            "SAFE_HTML = '&lt;div class=\"preview\"&gt;Safe&lt;/div&gt;'",
            "",
            "def process_repository(repo_url: str, output_dir: Path) -> bool:",
            '    """Process a public GitHub repository into a demo video."""',
            "    print(f'Starting analysis for {repo_url}...')",
            "    work_dir = output_dir / 'cache'",
            "    work_dir.mkdir(parents=True, exist_ok=True)",
            "",
            "    # Validate inputs and security constraints",
            "    if not repo_url.startswith('https://github.com/'):",
            "        raise ValueError('Only public GitHub repositories are supported.')",
            "",
            "    # Step 1: Collect metadata & key files",
            "    repo_data = analyze(repo_url)",
            "    if not repo_data.key_files:",
            "        return False",
            "",
            "    # Step 2: Generate narration script",
            "    script = write_script(repo_data)",
            "",
            "    # Step 3: Render animation scenes",
            "    for scene in script.scenes:",
            "        render_scene(scene, work_dir)",
            "",
            "    # Step 4: Synthesize TTS and join with FFmpeg",
            "    video_path = output_dir / 'final_demo.mp4'",
            "    synthesize_and_mux(script, video_path)",
            "    return True",
            "",
            "if __name__ == '__main__':",
            "    process_repository('https://github.com/owner/repo', Path('output'))",
        ]
        content = "\n".join(code_lines)
        kf = KeyFile(
            id="f1",
            path="repo2video/pipeline.py",
            language="python",
            content=content,
            total_lines=len(code_lines),
        )
        scene = Scene(
            id=1,
            type="code",
            file_id="f1",
            heading="Pipeline Architecture",
            narration="Here is the core pipeline implementation showing repository analysis and animated scene rendering without executing untrusted code.",
            focus=[FocusRange(start=8, end=14), FocusRange(start=22, end=29)],
        )
        html = build_code_html(
            scene,
            kf,
            theme=theme,
            accent=accent,
            repo_label=repo_label,
            duration=duration,
            settings=settings,
        )

    elif scene_type == "terminal":
        # Terminal session with 30-line output to verify scrolling and design
        commands = [
            "git clone https://github.com/Axoril1/CodeScan_Ai.git",
            "cd CodeScan_Ai && python -m venv .venv && source .venv/bin/activate",
            "pip install -r requirements.txt",
        ]
        output = [
            "Collecting requests>=2.31.0",
            "  Downloading requests-2.31.0-py3-none-any.whl (62 kB)",
            "Collecting pydantic>=2.7.0",
            "  Downloading pydantic-2.7.0-py3-none-any.whl (407 kB)",
            "Collecting playwright>=1.43.0",
            "  Downloading playwright-1.43.0-py3-none-manylinux_2_17_x86_64.whl (36.4 MB)",
            "Collecting piper-tts>=1.2.0",
            "  Downloading piper_tts-1.2.0-py3-none-any.whl (18.2 MB)",
            "Collecting Pillow>=10.2.0",
            "  Downloading pillow-10.2.0-cp311-cp311-manylinux.whl (4.5 MB)",
            "Collecting Pygments>=2.17.0",
            "  Downloading pygments-2.17.2-py3-none-any.whl (1.2 MB)",
            "Installing collected packages: requests, pydantic, playwright, piper-tts, Pillow, Pygments",
            "Running setup.py develop for repo2video",
            "Compiling native extensions...",
            "Building wheel for onnxruntime (pyproject.toml) ... done",
            "Created wheel for onnxruntime: filename=onnxruntime-1.17.1-cp311.whl",
            "Installing collected packages: onnxruntime",
            "Successfully installed onnxruntime-1.17.1",
            "Running post-installation verification check...",
            "Checking Pygments syntax lexers... OK",
            "Checking Playwright headless Chromium browser... OK",
            "Checking Piper ONNX voice synthesis model... OK",
            "Checking FFmpeg libx264 image2pipe encoder... OK",
            "Creating configuration cache in /home/axoril/.repo2video...",
            "Writing local environment template .env.example... OK",
            "All 30 system and package dependencies verified successfully.",
        ]
        session = TerminalSession.model_construct(
            id="t1",
            purpose="install",
            commands=commands,
            output=output,
        )
        scene = Scene(
            id=2,
            type="terminal",
            terminal_id="t1",
            heading="Installation & Setup",
            narration="Watch how quickly repo2video sets up its dependencies in an isolated virtual environment completely offline.",
        )
        html = build_terminal_html(
            scene,
            session,
            theme=theme,
            accent=accent,
            repo_label=repo_label,
            duration=duration,
            settings=settings,
        )

    elif scene_type == "media":
        img_path = out_dir / "preview_media_sample.png"
        make_dummy_media_image(img_path)
        asset = MediaAsset(
            id="m1",
            kind="image",
            path=img_path,
            width=1600,
            height=760,
            duration_seconds=0.0,
            alt="Dashboard UI and Code Scanner Results",
            origin="readme",
            source_url="https://github.com/Axoril1/CodeScan_Ai/raw/main/docs/dashboard.png",
        )
        scene = Scene(
            id=3,
            type="media",
            media_id="m1",
            heading="Application Dashboard",
            narration="Here is a high resolution screenshot of the user interface captured directly from the project README.",
        )
        html = build_media_html(
            scene,
            asset,
            theme=theme,
            accent=accent,
            repo_label=repo_label,
            duration=duration,
            settings=settings,
        )
    elif scene_type == "arch":
        spec = ArchitectureSpec(
            nodes=[
                NodeSpec(id="cli", label="CLI Entrypoint", group="input"),
                NodeSpec(id="analyzer", label="Repo Analyzer", group="core"),
                NodeSpec(id="scriptwriter", label="LLM Writer", group="core"),
                NodeSpec(id="anim", label="Anim Engine", group="core"),
                NodeSpec(id="muxer", label="FFmpeg Muxer", group="output"),
                NodeSpec(id="storage", label="Asset Cache", group="storage"),
            ],
            edges=[
                EdgeSpec(source="cli", target="analyzer", label="Repo URL"),
                EdgeSpec(source="analyzer", target="scriptwriter", label="RepoData"),
                EdgeSpec(source="scriptwriter", target="anim", label="ScriptData"),
                EdgeSpec(source="anim", target="muxer", label="Clips"),
                EdgeSpec(source="analyzer", target="storage", label="media assets"),
                EdgeSpec(source="anim", target="storage", label="reads"),
            ],
        )
        scene = Scene(
            id=4,
            type="architecture",
            heading="Pipeline Architecture",
            narration="Here is an overview of the modular end-to-end processing pipeline from repository URL to final video.",
            architecture=spec,
        )
        html = build_arch_html(
            scene,
            spec,
            theme=theme,
            accent=accent,
            repo_label=repo_label,
            duration=duration,
            settings=settings,
        )

    elif scene_type == "stats":
        repo = RepoData(
            owner="Axoril1",
            name="repo2video",
            description="Turn any public GitHub repository into a polished narrated demo video.",
            stars=1250,
            forks=140,
            open_issues=8,
            license="MIT",
            pushed_at="2026-10-25T18:30:00Z",
            language_percentages={
                "Python": 84.5,
                "JavaScript": 8.5,
                "HTML": 4.5,
                "CSS": 2.5,
            },
            topics=["video-generation", "playwright", "ffmpeg", "piper-tts", "automation"],
        )
        scene = Scene(
            id=5,
            type="stats",
            heading="Project Stats & Languages",
            narration="Here is a snapshot of repository metrics, open source engagement, and technology breakdown across the codebase.",
            stat_keys=["stars", "forks", "updated", "languages"],
        )
        html = build_stats_html(
            scene,
            repo,
            theme=theme,
            accent=accent,
            repo_label=repo_label,
            duration=duration,
            settings=settings,
        )
    else:
        print(f"Unknown scene type: {scene_type}")
        return

    png_path = out_dir / f"preview_{scene_type}_{theme}_t{t_snapshot:.1f}s.png"
    mp4_path = out_dir / f"preview_{scene_type}_{theme}.mp4"

    with AnimationRenderer(settings) as renderer:
        print(f"Rendering PNG snapshot at t={t_snapshot}s to {png_path}...")
        frame_bytes = renderer.capture_frame(html, t_snapshot)
        png_path.write_bytes(frame_bytes)

        if render_video:
            print(f"Rendering {duration}s MP4 video to {mp4_path}...")
            renderer.render(html, duration=duration, out_path=mp4_path)

    print(f"Done! Outputs saved in {out_dir}")


def main() -> None:
    """CLI parser for preview_scene."""
    parser = argparse.ArgumentParser(description="Render animated scene previews.")
    parser.add_argument(
        "--scene-type",
        choices=["code", "terminal", "media", "arch", "stats", "all"],
        default="all",
        help="Scene type to preview (default: all)",
    )
    parser.add_argument(
        "--theme",
        choices=["dark", "light", "both"],
        default="both",
        help="Color theme (default: both)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=2.5,
        help="Timestamp in seconds for the PNG screenshot (default: 2.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/preview"),
        help="Directory to save preview outputs (default: output/preview)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MP4 rendering and only generate PNG snapshots",
    )

    args = parser.parse_args()

    scene_types = (
        ["code", "terminal", "media", "arch", "stats"]
        if args.scene_type == "all"
        else [args.scene_type]
    )
    themes = ["dark", "light"] if args.theme == "both" else [args.theme]

    for st in scene_types:
        for th in themes:
            print(f"\n=== Generating preview for scene '{st}' in {th} theme ===")
            generate_preview(
                scene_type=st,
                theme=th,
                t_snapshot=args.time,
                out_dir=args.output_dir,
                render_video=not args.no_video,
            )


if __name__ == "__main__":
    main()
