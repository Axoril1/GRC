# GRC

Turn any public GitHub repository into a narrated demo video (MP4) in minutes.

`GRC` analyzes repository metadata, code, terminals, and visuals, drafts a narrated script via an LLM, synthesizes voiceover using Piper, generates deterministic animations using Playwright Chromium, and composites the final MP4 with FFmpeg—100% free and open-source, running locally on a standard laptop without GPU, Docker, or paid APIs.

---

## Features

- **Rich Visual Scene Types**: Title, Problem, Feature, How-To, Media screenshots, Syntax-highlighted Code snippets, Animated Terminal replaying README commands, Architecture diagrams, Repo Metrics & Stats, and Outro.
- **Dynamic Animations**: Deterministic frame capture for scrolling code highlights, typed terminal commands, and animated diagrams.
- **Safety First**: Sandboxed execution. Untrusted repositories are never executed, and private networks are strictly blocked.
- **Graceful Degradation**: Every rich visual is optional. If any animation, screenshot, or diagram fails, the scene seamlessly falls back to a clean static slide and the video still renders.
- **Flexible Options**: Custom themes, accent colors, voice speeds, caption modes (off, SRT, burned), background music, and fast rendering mode.

---

## Prerequisites

1. **Python 3.11+**
2. **FFmpeg** and **ffprobe** installed and available on your `PATH` (with `libx264`, `aac`, and optional `xfade`/`overlay` filters):
   - **Linux**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: `winget install ffmpeg`
3. **Playwright Chromium**:
   ```bash
   pip install playwright
   playwright install chromium
   ```
4. **Piper TTS**: Local neural text-to-speech engine.
5. **Local LLM (Ollama)**: By default, uses local Ollama (e.g. `qwen2.5:7b`).

Check your local environment readiness at any time:
```bash
grc doctor
```

---

## Installation

```bash
git clone https://github.com/Axoril1/GRC.git
cd GRC
pip install -e .
```

---

## Quickstart

Generate a video from a GitHub repository:
```bash
grc make https://github.com/owner/repo
```

Fast preview run (15 fps animations, fast encode):
```bash
grc make https://github.com/owner/repo --fast
```

Script-only generation (stops after script stage):
```bash
grc make https://github.com/owner/repo --script-only
```

Re-render from an existing `scenes.json`:
```bash
grc render ./output/owner__repo/scenes.json
```

---

## Command-Line Options

### `make` Options

| Option | Default | Description |
|---|---|---|
| `--out, -o PATH` | Auto | Destination path for the rendered MP4 video. |
| `--fast` | `False` | Renders animations faster at 15 fps and encodes video with CRF 24 and `veryfast` preset. |
| `--anim-fps N` | `24` | Animation rendering frame rate (10 to 30 fps). |
| `--no-media` | `False` | Skip downloading README images or capturing homepage screenshots. |
| `--no-animation` | `False` | Disable video clips; render all animated scenes as static slides captured at 85% timestamp. |
| `--no-transitions` | `False` | Disable crossfade transitions between scenes (uses cut transitions). |
| `--captions MODE` | `srt` | Subtitle caption mode: `off`, `srt` (companion SRT file), or `burn` (burned into MP4). |
| `--music PATH` | `None` | Optional background music audio track (MP3, WAV, OGG, M4A, FLAC). |
| `--theme THEME` | `dark` | Visual color theme (`dark` or `light`). |
| `--accent HEX` | `#6366f1` | Hex color code for badges and highlights. |
| `--provider PROVIDER`| `ollama` | LLM provider (`ollama` or `openai`). |
| `--model MODEL` | `qwen2.5:7b` | Model identifier to use. |
| `--voice-speed SPEED`| `1.0` | Voice narration speed factor (0.5 to 1.5). |
| `--reuse` | `False` | Reuse existing `repo.json` and `scenes.json` in the work directory. |
| `--force` | `False` | Discard cached assets and rerun all stages from scratch. |
| `--keep-temp` | `False` | Preserve intermediate video clips and concatenated segments. |
| `--script-only` | `False` | Exit after script generation without rendering voice or visuals. |
| `--verbose, -V` | `False` | Enable detailed debug logging. |

---

## What the Tool Downloads (Privacy & Security)

Safety and user privacy are fundamental constraints of `GRC`:

### Network Connections
`GRC` only ever connects to the following network endpoints:
1. `api.github.com`: To fetch public repository metadata, directory trees, and README contents.
2. **Configured LLM Address**: Local by default (`http://localhost:11434` for Ollama). If OpenAI is configured, connects to `api.openai.com`.
3. **README Media Hosts**: Public image and video files directly referenced in the repository README.
4. **Project Homepage**: If the repository lists a public homepage URL, Playwright loads the page inside an isolated sandbox with strict route filtering to capture a single preview screenshot.

All other network destinations are blocked. Requests to private, link-local, loopback, or cloud metadata addresses (RFC 1918, RFC 3927, `127.0.0.0/8`, `169.254.169.254`, `::1`) are strictly rejected.

### Code Execution Safety Rule
**Nothing from the analyzed repository is ever executed.**
- Code snippets extracted from the repository are strictly treated as text for syntax highlighting.
- Installation and usage commands found in the README are never invoked in a shell; they are parsed and replayed as text animations inside a mock terminal UI.
- Downloaded images and web pages are treated as untrusted input: images are sanitized and validated with Pillow, and browser contexts have JavaScript permissions and downloads disabled.

### Fallback Guarantee
Every visual element is optional. If an image download fails, an animated builder encounters unexpected syntax, or a diagram fails to layout, the pipeline logs a warning, records the fallback in `run.json`, and seamlessly replaces the scene with a clean static slide. The video generation will still succeed.

---

## Visual Asset Attribution

Images, logos, and video clips downloaded from a repository belong to their respective original authors. When media assets from a repository are used in the video, `GRC` automatically appends an attribution credit line to the outro scene and records the credit in `run.json` (e.g. *"Visuals from the owner/repo README"*).
