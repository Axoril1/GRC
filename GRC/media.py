"""Safe media collection, download, quality filtering, and homepage capture.

Safely downloads README media references (images, gifs, videos) and captures
homepage screenshots for inclusion in generated video scripts.
"""

from __future__ import annotations

import io
import ipaddress
import json
import logging
import socket
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from PIL import Image, ImageStat

from GRC.config import Settings, get_settings
from GRC.models import MediaAsset, MediaRef, RepoData

logger = logging.getLogger(__name__)

__all__ = [
    "MediaAsset",
    "capture_homepage",
    "collect_media",
    "credit_line",
    "download_media",
    "is_public_url",
]

# Network and limits
USER_AGENT = "GRC/0.2"
DOWNLOAD_TIMEOUT_SECONDS = 20
MAX_REDIRECTS = 3

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_GIF_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_VIDEO_BYTES = 40 * 1024 * 1024  # 40 MB

# Image quality filter parameters
MAX_PIXELS = 40_000_000
MIN_IMAGE_WIDTH = 480
MIN_IMAGE_HEIGHT = 270
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 4.0
MIN_STDDEV = 4.0
MAX_LONGEST_SIDE = 2400
THUMB_SIZE = (16, 16)
DUPLICATE_DIFF_THRESHOLD = 3.0

# Clips parameters
MIN_CLIP_DURATION = 1.5
MAX_CLIP_DURATION = 12.0
FFMPEG_TIMEOUT_SECONDS = 120

# Homepage parameters
HOMEPAGE_TIMEOUT_SECONDS = 45
PROHIBITED_HOMEPAGE_DOMAINS = frozenset(
    {
        "github.com",
        "npmjs.com",
        "pypi.org",
        "crates.io",
        "pkg.go.dev",
        "hub.docker.com",
    }
)
BOT_WALL_PHRASES = (
    "just a moment",
    "access denied",
    "captcha",
    "verify you are human",
    "attention required",
    "404",
    "not found",
)

# Ranking keywords
BONUS_KEYWORDS = (
    "screenshot",
    "demo",
    "preview",
    "ui",
    "dashboard",
    "screen",
    "app",
    "interface",
)
PENALTY_KEYWORDS = (
    "logo",
    "icon",
    "banner",
    "badge",
    "sponsor",
    "contributor",
)

CARRIER_GRADE_NAT = ipaddress.ip_network("100.64.0.0/10")


def is_public_url(url: str) -> bool:
    """Return True if the URL points to a publicly routable HTTP/HTTPS endpoint.

    Rejects loopback, private, link-local (169.254.x.x), multicast, reserved,
    carrier-grade NAT (100.64.0.0/10), IPv4-mapped IPv6, and internal hostnames
    (.local, .internal, localhost).

    Note on DNS rebinding:
    DNS records can change between this validation check and the subsequent HTTP
    connection. We mitigate this by validating all resolved IP addresses returned
    by getaddrinfo, validating every hop in manual redirect chains, and restricting
    connections to standard ports (80/443). The residual risk of a remote nameserver
    swapping records during milliseconds between check and socket connect is accepted.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False

    if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
        return False

    try:
        port = parsed.port
    except ValueError:
        return False

    if port not in (None, 80, 443):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    h = hostname.lower().rstrip(".")
    if h == "localhost" or h.endswith((".local", ".internal")):
        return False

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, socket.herror, OSError):
        return False

    if not addr_infos:
        return False

    for res in addr_infos:
        sockaddr = res[4]
        if not sockaddr:
            return False
        addr_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr_str)
        except ValueError:
            return False

        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped

        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
            or (ip.version == 4 and ip in CARRIER_GRADE_NAT)
        ):
            return False

    return True


def detect_file_type(header: bytes) -> str | None:
    """Detect media format based on leading magic bytes.

    Returns one of: 'png', 'jpeg', 'gif', 'webp', 'mp4', 'webm', or None.
    """
    if len(header) >= 8 and header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(header) >= 3 and header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(header) >= 6 and header.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "mp4"
    if len(header) >= 4 and header.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    return None


def download_media(
    ref: MediaRef,
    dest_dir: Path,
    session: requests.Session | None = None,
) -> Path | None:
    """Safely download a media reference into dest_dir, validating SSRF and magic bytes.

    Follows up to 3 manual redirects, checking each hop with is_public_url.
    Aborts if file size exceeds limits (10MB image, 25MB GIF, 40MB video).
    Deletes partial files on abort. Names files by id and detected extension.
    Returns the Path on success, or None on failure (logging a warning).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    sess.cookies.clear()
    headers = {"User-Agent": USER_AGENT}

    current_url = ref.url
    hops = 0
    resp = None

    while hops <= MAX_REDIRECTS:
        if not is_public_url(current_url):
            logger.warning("Rejected non-public URL for %s: %s", ref.id, current_url)
            return None

        try:
            resp = sess.get(
                current_url,
                headers=headers,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                allow_redirects=False,
            )
        except (requests.RequestException, OSError, ValueError) as exc:
            logger.warning("Download request error for %s (%s): %s", ref.id, current_url, exc)
            return None

        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            resp.close()
            if not loc:
                logger.warning("Redirect missing Location header for %s (%s)", ref.id, current_url)
                return None
            current_url = urljoin(current_url, loc)
            hops += 1
            if hops > MAX_REDIRECTS:
                logger.warning("Exceeded %d redirects for %s (%s)", MAX_REDIRECTS, ref.id, ref.url)
                return None
            continue
        break

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "None"
        logger.warning("Download failed for %s (%s): HTTP %s", ref.id, current_url, code)
        if resp is not None:
            resp.close()
        return None

    # Read first chunk to inspect magic bytes
    try:
        chunk_iter = resp.iter_content(chunk_size=8192)
        try:
            first_chunk = next(chunk_iter)
        except StopIteration:
            logger.warning("Empty response body for %s (%s)", ref.id, current_url)
            resp.close()
            return None

        detected_type = detect_file_type(first_chunk)
        if detected_type is None:
            logger.warning("Unrecognized magic bytes for %s (%s)", ref.id, current_url)
            resp.close()
            return None

        # Determine size limit based on detected type
        if detected_type == "gif":
            size_limit = MAX_GIF_BYTES
            ext = ".gif"
        elif detected_type in ("mp4", "webm"):
            size_limit = MAX_VIDEO_BYTES
            ext = f".{detected_type}"
        else:
            size_limit = MAX_IMAGE_BYTES
            ext = ".jpg" if detected_type == "jpeg" else f".{detected_type}"

        # Check Content-Length header if present
        content_len_header = resp.headers.get("Content-Length")
        if content_len_header:
            try:
                if int(content_len_header) > size_limit:
                    logger.warning(
                        "Content-Length %s exceeds limit %s for %s (%s)",
                        content_len_header,
                        size_limit,
                        ref.id,
                        current_url,
                    )
                    resp.close()
                    return None
            except ValueError:
                pass

        dest_path = dest_dir / f"{ref.id}{ext}"
        bytes_written = len(first_chunk)
        if bytes_written > size_limit:
            logger.warning("First chunk exceeded size limit for %s", ref.id)
            resp.close()
            return None

        with open(dest_path, "wb") as f:
            f.write(first_chunk)
            for chunk in chunk_iter:
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > size_limit:
                    logger.warning(
                        "Download exceeded limit %d bytes for %s; deleting partial file",
                        size_limit,
                        ref.id,
                    )
                    resp.close()
                    f.close()
                    dest_path.unlink(missing_ok=True)
                    return None
                f.write(chunk)

        resp.close()
        return dest_path

    except (OSError, requests.RequestException, ValueError) as exc:
        logger.warning("Error streaming media for %s: %s", ref.id, exc)
        resp.close()
        return None


def process_image(
    source_path: Path,
    target_dir: Path,
    asset_id: str,
    accepted_thumbnails: list[list[int]],
) -> tuple[Path, int, int] | None:
    """Validate, filter, and normalize an image using Pillow.

    Applies decompression bomb checks, min dimension (480x270), aspect ratio
    (0.4-4.0), grayscale stddev (>=4.0), and 16x16 thumbnail deduplication.
    Shrinks longest side to <=2400. Saves as PNG (or JPEG quality 90 if >2MB without transparency).
    Returns (saved_path, width, height) or None if rejected.
    """
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
            with Image.open(source_path) as img:
                img.verify()

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
            with Image.open(source_path) as img:
                img.load()
                if getattr(img, "is_animated", False):
                    img.seek(0)

                w, h = img.size
                if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                    logger.warning(
                        "Image %s rejected: too small (%dx%d < %dx%d)",
                        asset_id,
                        w,
                        h,
                        MIN_IMAGE_WIDTH,
                        MIN_IMAGE_HEIGHT,
                    )
                    return None

                aspect = w / h
                if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                    logger.warning(
                        "Image %s rejected: aspect ratio %.2f outside %.1f-%.1f",
                        asset_id,
                        aspect,
                        MIN_ASPECT_RATIO,
                        MAX_ASPECT_RATIO,
                    )
                    return None

                gray = img.convert("L")
                stat = ImageStat.Stat(gray)
                stddev = stat.stddev[0]
                if stddev < MIN_STDDEV:
                    logger.warning(
                        "Image %s rejected: blank or flat image (stddev=%.2f < %.2f)",
                        asset_id,
                        stddev,
                        MIN_STDDEV,
                    )
                    return None

                # 16x16 thumbnail deduplication
                thumb = gray.resize(THUMB_SIZE, Image.Resampling.BILINEAR)
                if hasattr(thumb, "get_flattened_data"):
                    thumb_data = list(thumb.get_flattened_data())
                else:
                    thumb_data = list(thumb.getdata())  # pragma: no cover

                for existing_thumb in accepted_thumbnails:
                    mean_diff = sum(abs(a - b) for a, b in zip(thumb_data, existing_thumb)) / len(
                        thumb_data
                    )
                    if mean_diff < DUPLICATE_DIFF_THRESHOLD:
                        logger.warning(
                            "Image %s rejected: near-duplicate (mean diff=%.2f < %.2f)",
                            asset_id,
                            mean_diff,
                            DUPLICATE_DIFF_THRESHOLD,
                        )
                        return None

                # Transparency check
                has_transparency = False
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    rgba = img.convert("RGBA")
                    alpha = rgba.getchannel("A")
                    min_a, _ = alpha.getextrema()
                    if min_a < 255:
                        has_transparency = True
                        img = rgba
                    else:
                        img = img.convert("RGB")
                else:
                    img = img.convert("RGB")

                # Resize if longest side exceeds limit
                longest = max(w, h)
                if longest > MAX_LONGEST_SIDE:
                    scale = MAX_LONGEST_SIDE / longest
                    new_w = round(w * scale)
                    new_h = round(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    w, h = new_w, new_h

                # Save as PNG or JPEG
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                png_bytes = buf.getvalue()

                if len(png_bytes) > 2 * 1024 * 1024 and not has_transparency:
                    out_path = target_dir / f"{asset_id}.jpg"
                    rgb_img = img.convert("RGB")
                    rgb_img.save(out_path, format="JPEG", quality=90)
                else:
                    out_path = target_dir / f"{asset_id}.png"
                    out_path.write_bytes(png_bytes)

                accepted_thumbnails.append(thumb_data)
                return (out_path, w, h)

    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.warning("Pillow processing failed for %s: %s", asset_id, exc)
        return None


def convert_to_clip(
    source_path: Path,
    target_dir: Path,
    asset_id: str,
) -> tuple[Path, int, int, float] | None:
    """Convert a video or GIF to a normalized MP4 clip using FFmpeg.

    Scales to fit inside 1600x800 with even dimensions, 24fps, yuv420p, max 12 seconds.
    Reads duration, width, height with ffprobe. Rejects clips shorter than 1.5 seconds.
    Returns (mp4_path, width, height, duration) or None.
    """
    clip_path = target_dir / f"{asset_id}.mp4"
    temp_clip = target_dir / f"{asset_id}_temp.mp4" if source_path == clip_path else clip_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-t",
        str(int(MAX_CLIP_DURATION)),
        "-vf",
        "scale=1600:800:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=24,format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(temp_clip),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning(
                "FFmpeg clip conversion failed for %s (exit %d)", asset_id, proc.returncode
            )
            temp_clip.unlink(missing_ok=True)
            return None

        # Probe duration and dimensions with ffprobe
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(temp_clip),
        ]
        probe_proc = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe_proc.returncode != 0:
            logger.warning("ffprobe failed for %s", asset_id)
            temp_clip.unlink(missing_ok=True)
            return None

        data = json.loads(probe_proc.stdout)
        streams = data.get("streams", [])
        if not streams:
            logger.warning("No video streams found in converted clip %s", asset_id)
            temp_clip.unlink(missing_ok=True)
            return None

        width = int(streams[0].get("width", 0))
        height = int(streams[0].get("height", 0))
        dur_val = streams[0].get("duration") or data.get("format", {}).get("duration", "0")
        try:
            duration = float(dur_val)
        except ValueError:
            duration = 0.0

        if duration < MIN_CLIP_DURATION or width <= 0 or height <= 0:
            logger.warning(
                "Clip %s rejected: duration %.2fs < %.1fs or invalid dims (%dx%d)",
                asset_id,
                duration,
                MIN_CLIP_DURATION,
                width,
                height,
            )
            temp_clip.unlink(missing_ok=True)
            return None

        if temp_clip != clip_path:
            temp_clip.replace(clip_path)

        return (clip_path, width, height, duration)

    except (subprocess.SubprocessError, OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Clip conversion error for %s: %s", asset_id, exc)
        temp_clip.unlink(missing_ok=True)
        return None


def capture_homepage(
    url: str,
    work_dir: Path,
    settings: Settings,
    *,
    max_shots: int = 2,
    _url_validator: Callable[[str], bool] | None = None,
) -> list[MediaAsset]:
    """Capture 1 or 2 clean screenshots of the project homepage using Playwright Chromium.

    Screenshots are taken headlessly at 1440x900 viewport. Filters out bot walls,
    error pages, blank shots, and duplicates.
    Any failure returns an empty list and logs a warning. Never raises an exception.
    """
    if not url or not isinstance(url, str) or not url.strip():
        return []
    url = url.strip()

    try:
        parsed = urlsplit(url)
    except ValueError:
        return []

    host = (parsed.hostname or "").lower()
    for prohibited in PROHIBITED_HOMEPAGE_DOMAINS:
        if host == prohibited or host.endswith("." + prohibited):
            logger.warning("Skipping homepage on prohibited domain %s: %s", host, url)
            return []

    validator = _url_validator or is_public_url
    if not validator(url):
        logger.warning("Skipping homepage on non-public URL: %s", url)
        return []

    start_time = time.monotonic()
    media_dir = work_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    dns_cache: dict[str, bool] = {}
    assets: list[MediaAsset] = []
    accepted_thumbs: list[list[int]] = []

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    viewport={"width": 1440, "height": 900},
                    device_scale_factor=1,
                    accept_downloads=False,
                    permissions=[],
                    service_workers="block",
                    locale="en-US",
                )
                page = context.new_page()
                page.on("dialog", lambda dialog: dialog.dismiss())

                def route_handler(route):
                    req = route.request
                    r_url = req.url
                    if req.resource_type in ("media", "websocket", "eventsource"):
                        route.abort()
                        return
                    try:
                        p_url = urlsplit(r_url)
                    except ValueError:
                        route.abort()
                        return
                    if p_url.scheme.lower() not in ("http", "https"):
                        route.abort()
                        return
                    r_host = (p_url.hostname or "").lower()
                    if not r_host:
                        route.abort()
                        return
                    if _url_validator is not None:
                        if not _url_validator(r_url):
                            route.abort()
                            return
                    else:
                        if r_host not in dns_cache:
                            dns_cache[r_host] = is_public_url(f"{p_url.scheme}://{p_url.netloc}/")
                        if not dns_cache[r_host]:
                            route.abort()
                            return
                    route.continue_()

                page.route("**/*", route_handler)

                resp = page.goto(url, timeout=20_000, wait_until="load")
                page.wait_for_timeout(1500)

                if resp is None or resp.status >= 400:
                    status_code = resp.status if resp else "None"
                    logger.warning("Homepage failed with HTTP %s: %s", status_code, url)
                    return []

                # Bot wall and error page detection
                title = (page.title() or "").lower()
                body_text = ""
                try:
                    body_text = (page.locator("body").inner_text(timeout=2000) or "").lower()[:2000]
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not read homepage body text: %s", exc)
                combined_text = f"{title}\n{body_text}"

                for phrase in BOT_WALL_PHRASES:
                    if phrase in combined_text:
                        logger.warning(
                            "Homepage skipped: detected bot wall or error keyword '%s'", phrase
                        )
                        return []

                if (time.monotonic() - start_time) > HOMEPAGE_TIMEOUT_SECONDS:
                    logger.warning("Homepage capture exceeded timeout limit for %s", url)
                    return []

                # Shot 1: Top viewport
                shot1_bytes = page.screenshot(full_page=False)
                shot1_tmp = media_dir / "temp_h1.png"
                shot1_tmp.write_bytes(shot1_bytes)
                res1 = process_image(shot1_tmp, media_dir, "h1", accepted_thumbs)
                shot1_tmp.unlink(missing_ok=True)
                if res1:
                    p1, w1, h1 = res1
                    assets.append(
                        MediaAsset(
                            id="h1",
                            kind="image",
                            path=p1,
                            width=w1,
                            height=h1,
                            duration_seconds=0.0,
                            alt="Homepage screenshot",
                            origin="homepage",
                            source_url=url,
                        )
                    )

                # Shot 2: Scroll down 800px if page is taller than 1800px
                if max_shots >= 2 and (time.monotonic() - start_time) < (
                    HOMEPAGE_TIMEOUT_SECONDS - 5
                ):
                    scroll_height = page.evaluate(
                        "() => document.documentElement.scrollHeight || document.body.scrollHeight"
                    )
                    if scroll_height > 1800:
                        page.evaluate("() => window.scrollBy(0, 800)")
                        page.wait_for_timeout(800)
                        shot2_bytes = page.screenshot(full_page=False)
                        shot2_tmp = media_dir / "temp_h2.png"
                        shot2_tmp.write_bytes(shot2_bytes)
                        res2 = process_image(shot2_tmp, media_dir, "h2", accepted_thumbs)
                        shot2_tmp.unlink(missing_ok=True)
                        if res2:
                            p2, w2, h2 = res2
                            assets.append(
                                MediaAsset(
                                    id="h2",
                                    kind="image",
                                    path=p2,
                                    width=w2,
                                    height=h2,
                                    duration_seconds=0.0,
                                    alt="Homepage screenshot",
                                    origin="homepage",
                                    source_url=url,
                                )
                            )

            finally:
                browser.close()

    except Exception as exc:  # noqa: BLE001
        logger.warning("Homepage capture error for %s: %s", url, exc)
        return []

    return assets


def compute_ranking_score(asset: MediaAsset) -> int:
    """Compute the visual ranking score for an asset.

    Score = area (capped at 1920x1080)
            + 500,000 (if alt or filename has screenshot/demo/preview/etc.)
            - 500,000 (if alt or filename has logo/icon/banner/etc.)
    """
    area = min(asset.width * asset.height, 1920 * 1080)
    url_name = Path(urlsplit(asset.source_url).path).name if asset.source_url else ""
    text = f"{asset.alt} {url_name} {asset.path.name}".lower()

    score = area
    if any(kw in text for kw in BONUS_KEYWORDS):
        score += 500_000
    if any(kw in text for kw in PENALTY_KEYWORDS):
        score -= 500_000
    return score


def collect_media(
    repo: RepoData,
    work_dir: Path,
    settings: Settings,
    *,
    session: requests.Session | None = None,
    force: bool = False,
    limit: int = 4,
    _url_validator: Callable[[str], bool] | None = None,
) -> list[MediaAsset]:
    """Collect README media and homepage screenshots, ranked and cached in manifest.json.

    Returns up to limit MediaAssets, with at most 2 clips.
    When force is False and manifest.json exists with all referenced files present,
    returns cached assets immediately without any network calls.
    """
    media_dir = work_dir / "media"
    manifest_path = media_dir / "manifest.json"

    # Fast cache check
    if not force and manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            cached_assets: list[MediaAsset] = []
            all_exist = True
            for item in data:
                rel_path = item.get("path")
                if not rel_path:
                    all_exist = False
                    break
                full_path = media_dir / rel_path
                if not full_path.is_file():
                    all_exist = False
                    break
                cached_assets.append(
                    MediaAsset(
                        id=item["id"],
                        kind=item["kind"],
                        path=full_path,
                        width=int(item["width"]),
                        height=int(item["height"]),
                        duration_seconds=float(item.get("duration_seconds", 0.0)),
                        alt=item.get("alt", ""),
                        origin=item.get("origin", "readme"),
                        source_url=item.get("source_url", ""),
                    )
                )
            if all_exist and cached_assets:
                logger.info("Loaded %d media assets from manifest.json cache", len(cached_assets))
                return cached_assets
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Manifest cache read failed, re-collecting: %s", exc)

    media_dir.mkdir(parents=True, exist_ok=True)
    refs_found = len(repo.media)
    downloaded_count = 0
    rejections: dict[str, int] = {}
    accepted_candidates: list[MediaAsset] = []
    accepted_thumbs: list[list[int]] = []

    # Download README media refs one by one
    for ref in repo.media:
        if len(accepted_candidates) >= limit * 2:
            break

        dl_path = download_media(ref, media_dir, session=session)
        if dl_path is None:
            rejections["download_or_safety_failed"] = (
                rejections.get("download_or_safety_failed", 0) + 1
            )
            continue

        downloaded_count += 1
        suffix = dl_path.suffix.lower()

        if suffix in (".mp4", ".webm"):
            clip_res = convert_to_clip(dl_path, media_dir, ref.id)
            if clip_res:
                c_path, w, h, dur = clip_res
                accepted_candidates.append(
                    MediaAsset(
                        id=ref.id,
                        kind="clip",
                        path=c_path,
                        width=w,
                        height=h,
                        duration_seconds=dur,
                        alt=ref.alt,
                        origin="readme",
                        source_url=ref.url,
                    )
                )
            else:
                rejections["clip_conversion_failed"] = (
                    rejections.get("clip_conversion_failed", 0) + 1
                )

        elif suffix == ".gif":
            clip_res = convert_to_clip(dl_path, media_dir, ref.id)
            if clip_res:
                c_path, w, h, dur = clip_res
                accepted_candidates.append(
                    MediaAsset(
                        id=ref.id,
                        kind="clip",
                        path=c_path,
                        width=w,
                        height=h,
                        duration_seconds=dur,
                        alt=ref.alt,
                        origin="readme",
                        source_url=ref.url,
                    )
                )
            else:
                # Fall back to first frame as still image
                img_res = process_image(dl_path, media_dir, ref.id, accepted_thumbs)
                if img_res:
                    i_path, w, h = img_res
                    accepted_candidates.append(
                        MediaAsset(
                            id=ref.id,
                            kind="image",
                            path=i_path,
                            width=w,
                            height=h,
                            duration_seconds=0.0,
                            alt=ref.alt,
                            origin="readme",
                            source_url=ref.url,
                        )
                    )
                else:
                    rejections["image_quality_failed"] = (
                        rejections.get("image_quality_failed", 0) + 1
                    )

        else:
            img_res = process_image(dl_path, media_dir, ref.id, accepted_thumbs)
            if img_res:
                i_path, w, h = img_res
                accepted_candidates.append(
                    MediaAsset(
                        id=ref.id,
                        kind="image",
                        path=i_path,
                        width=w,
                        height=h,
                        duration_seconds=0.0,
                        alt=ref.alt,
                        origin="readme",
                        source_url=ref.url,
                    )
                )
            else:
                rejections["image_quality_failed"] = rejections.get("image_quality_failed", 0) + 1

    # Capture homepage only if fewer than limit assets exist
    if len(accepted_candidates) < limit and repo.homepage:
        needed = limit - len(accepted_candidates)
        shots = capture_homepage(
            repo.homepage,
            work_dir,
            settings,
            max_shots=min(2, needed),
            _url_validator=_url_validator,
        )
        accepted_candidates.extend(shots)

    # Ranking and selection: keep best limit assets with at most 2 clips
    sorted_candidates = sorted(accepted_candidates, key=compute_ranking_score, reverse=True)
    final_assets: list[MediaAsset] = []
    clip_count = 0
    for asset in sorted_candidates:
        if len(final_assets) >= limit:
            break
        if asset.kind == "clip":
            if clip_count >= 2:
                continue
            clip_count += 1
        final_assets.append(asset)

    # Write manifest.json
    manifest_data = [
        {
            "id": a.id,
            "kind": a.kind,
            "path": a.path.name,
            "width": a.width,
            "height": a.height,
            "duration_seconds": a.duration_seconds,
            "alt": a.alt,
            "origin": a.origin,
            "source_url": a.source_url,
        }
        for a in final_assets
    ]
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    logger.info(
        "Media collection complete: refs_found=%d, downloaded=%d, rejected=%s, selected=%d",
        refs_found,
        downloaded_count,
        rejections,
        len(final_assets),
    )

    return final_assets


def credit_line(assets: list[MediaAsset], repo_label: str) -> str:
    """Generate a human-readable attribution credit line for the media assets used.

    Args:
        assets: The list of media assets included in the video.
        repo_label: Repository name or owner/name label (e.g. 'octocat/Hello-World').

    Returns:
        Attribution string, or empty string if no assets exist.
    """
    if not assets:
        return ""
    origins = {a.origin for a in assets}
    has_readme = "readme" in origins
    has_homepage = "homepage" in origins
    if has_readme and has_homepage:
        return f"Visuals from the {repo_label} README and homepage"
    if has_readme:
        return f"Visuals from the {repo_label} README"
    if has_homepage:
        return f"Visuals from the {repo_label} homepage"
    return ""


def main() -> None:
    """CLI entry point for testing media collection on a saved repo.json."""
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python -m GRC.media <repo.json> <work_dir>\n")
        sys.exit(1)

    repo_file = Path(sys.argv[1])
    work_dir = Path(sys.argv[2])

    if not repo_file.is_file():
        sys.stderr.write(f"Error: repo file {repo_file} does not exist\n")
        sys.exit(1)

    try:
        repo_data = RepoData.model_validate_json(repo_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Error parsing {repo_file}: {exc}\n")
        sys.exit(1)

    settings = get_settings()
    assets = collect_media(repo_data, work_dir, settings)

    label = (
        f"{repo_data.owner}/{repo_data.name}"
        if repo_data.owner and repo_data.name
        else "repository"
    )
    print(f"\nCollected {len(assets)} media assets for {label}:\n")
    if not assets:
        print("  (No media assets collected)")
    else:
        header = f"{'ID':<6} {'Kind':<8} {'Origin':<10} {'Dimensions':<12} {'Duration':<10} {'File':<24} {'Alt'}"
        print(header)
        print("-" * len(header))
        for a in assets:
            dims = f"{a.width}x{a.height}"
            dur = f"{a.duration_seconds:.1f}s" if a.kind == "clip" else "-"
            rel = a.path.name
            print(
                f"{a.id:<6} {a.kind:<8} {a.origin:<10} {dims:<12} {dur:<10} {rel:<24} {a.alt[:30]}"
            )

        credit = credit_line(assets, label)
        if credit:
            print(f"\nCredit: {credit}\n")


if __name__ == "__main__":
    main()
