"""Offline unit and integration tests for repo2video/media.py."""

from __future__ import annotations

import http.server
import io
import json
import socket
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from GRC.config import Settings
from GRC.media import (
    MediaAsset,
    capture_homepage,
    collect_media,
    convert_to_clip,
    credit_line,
    download_media,
    is_public_url,
    process_image,
)
from GRC.models import MediaRef, RepoData


def _create_test_image(
    width: int,
    height: int,
    *,
    solid: bool = False,
    color: tuple[int, int, int] = (120, 150, 200),
    fmt: str = "PNG",
) -> bytes:
    """Generate a Pillow image in memory as bytes."""
    img = Image.new("RGB", (width, height), color=color)
    if not solid:
        pixels = img.load()
        for x in range(0, width, 5):
            for y in range(0, height, 5):
                r = (x * 7) % 256
                g = (y * 11) % 256
                b = ((x + y) * 13) % 256
                pixels[x, y] = (r, g, b)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _mock_response(
    content: bytes,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock requests.Response with iter_content support."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.content = content

    def iter_content(chunk_size: int = 8192, decode_unicode: bool = False):
        del decode_unicode
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    resp.iter_content = iter_content
    return resp


class TestUrlSafety:
    """Tests for is_public_url SSRF protection."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # Public endpoints (True)
            ("https://github.com", True),
            ("https://github.com/foo/bar.png", True),
            ("http://example.com", True),
            ("https://140.82.121.4", True),
            ("http://example.com:80/image.jpg", True),
            ("https://example.com:443/image.jpg", True),
            # Loopback (False)
            ("http://127.0.0.1", False),
            ("http://127.0.0.2:80", False),
            ("http://localhost", False),
            ("http://localhost:80", False),
            ("http://[::1]", False),
            ("http://[::ffff:127.0.0.1]", False),
            # Internal names (False)
            ("http://app.local", False),
            ("http://database.internal", False),
            # Private IPv4 (False)
            ("http://10.0.0.1", False),
            ("http://10.255.0.1", False),
            ("http://172.16.0.1", False),
            ("http://172.31.255.255", False),
            ("http://192.168.1.1", False),
            # Link-local / Cloud metadata (False)
            ("http://169.254.169.254/latest/meta-data/", False),
            ("http://[fe80::1]", False),
            # Carrier-grade NAT (False)
            ("http://100.64.0.1", False),
            # Multicast / Unspecified / Broadcast (False)
            ("http://224.0.0.1", False),
            ("http://0.0.0.0", False),
            ("http://255.255.255.255", False),
            # Unsupported schemes (False)
            ("file:///etc/passwd", False),
            ("ftp://example.com/file.png", False),
            ("javascript:alert(1)", False),
            # Credentials in URL (False)
            ("http://user:pass@example.com", False),
            # Non-standard ports (False)
            ("http://example.com:8080", False),
            ("https://example.com:8443", False),
            # Empty / malformed (False)
            ("", False),
            ("not-a-url", False),
        ],
    )
    def test_is_public_url_table(self, url: str, expected: bool) -> None:
        """Verify is_public_url rejects all private/unsafe URLs and accepts public ones."""
        orig_getaddrinfo = socket.getaddrinfo

        def mock_getaddrinfo(host, port, *args, **kwargs):
            if host in ("github.com", "example.com"):
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
            if host in ("app.local", "database.internal"):
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.2", 80))]
            return orig_getaddrinfo(host, port, *args, **kwargs)

        with patch("socket.getaddrinfo", side_effect=mock_getaddrinfo):
            assert is_public_url(url) == expected

    def test_hostname_resolving_to_private_ip_is_blocked(self) -> None:
        """A hostname that resolves to 192.168.1.50 must be rejected."""
        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.50", 80))],
        ):
            assert is_public_url("http://attacker-controlled-domain.com") is False

    def test_unresolvable_hostname_is_blocked(self) -> None:
        """A hostname that fails DNS resolution must be rejected."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
            assert is_public_url("http://non-existent-domain-xyz.com") is False


class TestDownloading:
    """Tests for download_media safety and limits."""

    def test_redirect_to_private_address_blocked(self, tmp_path: Path) -> None:
        """Redirecting to a private IP (e.g. 192.168.1.1) must be aborted."""
        session = MagicMock(spec=requests.Session)
        session.cookies = MagicMock()

        redirect_resp = MagicMock(spec=requests.Response)
        redirect_resp.status_code = 302
        redirect_resp.headers = {"Location": "http://192.168.1.1/secret.png"}
        session.get.return_value = redirect_resp

        ref = MediaRef(id="m1", url="https://example.com/redirect.png", kind="image")
        with patch("GRC.media.is_public_url", side_effect=lambda u: "192.168" not in u):
            res = download_media(ref, tmp_path, session=session)

        assert res is None
        assert list(tmp_path.glob("*")) == []

    def test_redirect_chain_longer_than_3_stopped(self, tmp_path: Path) -> None:
        """Redirect chains exceeding 3 hops must be stopped."""
        session = MagicMock(spec=requests.Session)
        session.cookies = MagicMock()

        def mock_get(url, *args, **kwargs):
            del args, kwargs
            hop = int(url.split("/")[-1].replace(".png", ""))
            resp = MagicMock(spec=requests.Response)
            resp.status_code = 302
            resp.headers = {"Location": f"https://example.com/hop/{hop + 1}.png"}
            return resp

        session.get.side_effect = mock_get
        ref = MediaRef(id="m1", url="https://example.com/hop/1.png", kind="image")

        with patch("GRC.media.is_public_url", return_value=True):
            res = download_media(ref, tmp_path, session=session)

        assert res is None
        assert session.get.call_count == 4  # Initial + 3 redirects

    def test_oversized_download_aborted_and_partial_file_deleted(self, tmp_path: Path) -> None:
        """Downloads exceeding the size limit (10MB for image) must be deleted."""
        session = MagicMock(spec=requests.Session)
        session.cookies = MagicMock()

        # Generate PNG header + 11MB of dummy data
        png_header = b"\x89PNG\r\n\x1a\n"
        big_payload = png_header + b"X" * (11 * 1024 * 1024)
        session.get.return_value = _mock_response(big_payload)

        ref = MediaRef(id="m1", url="https://example.com/huge.png", kind="image")
        with patch("GRC.media.is_public_url", return_value=True):
            res = download_media(ref, tmp_path, session=session)

        assert res is None
        assert not (tmp_path / "m1.png").exists()

    def test_wrong_magic_number_skipped(self, tmp_path: Path) -> None:
        """Unknown or wrong magic numbers must be rejected immediately."""
        session = MagicMock(spec=requests.Session)
        session.cookies = MagicMock()
        session.get.return_value = _mock_response(b"INVALID_MAGIC_BYTES_1234567890")

        ref = MediaRef(id="m1", url="https://example.com/bad_magic.png", kind="image")
        with patch("GRC.media.is_public_url", return_value=True):
            res = download_media(ref, tmp_path, session=session)

        assert res is None
        assert list(tmp_path.glob("*")) == []

    def test_content_type_lying_handled(self, tmp_path: Path) -> None:
        """Mismatched Content-Type is resolved by magic bytes."""
        session = MagicMock(spec=requests.Session)
        session.cookies = MagicMock()

        # Case A: Content-Type is text/html but payload is valid PNG
        valid_png = _create_test_image(500, 300)
        session.get.return_value = _mock_response(valid_png, headers={"Content-Type": "text/html"})

        ref = MediaRef(id="m1", url="https://example.com/actually_png.html", kind="image")
        with patch("GRC.media.is_public_url", return_value=True):
            res = download_media(ref, tmp_path, session=session)

        assert res is not None
        assert res.name == "m1.png"
        assert res.exists()

        # Case B: Content-Type is image/png but payload is HTML error page
        session.get.return_value = _mock_response(
            b"<!DOCTYPE html><html>404</html>", headers={"Content-Type": "image/png"}
        )
        ref2 = MediaRef(id="m2", url="https://example.com/fake.png", kind="image")
        with patch("GRC.media.is_public_url", return_value=True):
            res2 = download_media(ref2, tmp_path, session=session)

        assert res2 is None


class TestImageQualityFilter:
    """Tests for Pillow-based image filtering rules."""

    def test_image_too_small_rejected(self, tmp_path: Path) -> None:
        """Images smaller than 480x270 must be rejected."""
        small_bytes = _create_test_image(400, 250)
        p = tmp_path / "small.png"
        p.write_bytes(small_bytes)

        res = process_image(p, tmp_path, "m1", [])
        assert res is None

    def test_blank_image_rejected(self, tmp_path: Path) -> None:
        """Solid/flat images with stddev < 4.0 must be rejected."""
        solid_bytes = _create_test_image(600, 400, solid=True)
        p = tmp_path / "solid.png"
        p.write_bytes(solid_bytes)

        res = process_image(p, tmp_path, "m1", [])
        assert res is None

    def test_extreme_aspect_ratio_rejected(self, tmp_path: Path) -> None:
        """Aspect ratios outside 0.4 to 4.0 must be rejected."""
        # 500x1500 (aspect 0.33)
        tall_bytes = _create_test_image(500, 1500)
        p1 = tmp_path / "tall.png"
        p1.write_bytes(tall_bytes)
        assert process_image(p1, tmp_path, "m1", []) is None

        # 2500x500 (aspect 5.0)
        wide_bytes = _create_test_image(2500, 500)
        p2 = tmp_path / "wide.png"
        p2.write_bytes(wide_bytes)
        assert process_image(p2, tmp_path, "m2", []) is None

    def test_near_duplicate_rejected(self, tmp_path: Path) -> None:
        """Images whose 16x16 thumbnail mean difference is < 3 must be rejected."""
        good_bytes = _create_test_image(600, 400)
        p1 = tmp_path / "img1.png"
        p1.write_bytes(good_bytes)

        accepted_thumbs: list[list[int]] = []
        res1 = process_image(p1, tmp_path, "m1", accepted_thumbs)
        assert res1 is not None
        assert len(accepted_thumbs) == 1

        # Second identical image
        p2 = tmp_path / "img2.png"
        p2.write_bytes(good_bytes)
        res2 = process_image(p2, tmp_path, "m2", accepted_thumbs)
        assert res2 is None
        assert len(accepted_thumbs) == 1

    def test_valid_image_accepted_and_resized(self, tmp_path: Path) -> None:
        """Images with longest side > 2400 must be shrunk to <= 2400."""
        big_bytes = _create_test_image(3000, 1500)
        p = tmp_path / "big.png"
        p.write_bytes(big_bytes)

        res = process_image(p, tmp_path, "m1", [])
        assert res is not None
        out_path, w, h = res
        assert out_path.exists()
        assert max(w, h) <= 2400


class TestRankingAndSelection:
    """Tests for ranking score, limits, and clip caps."""

    def test_ranking_prefers_screenshot_over_logo(self, tmp_path: Path) -> None:
        """Ranking should apply +500,000 for screenshot and -500,000 for logo."""
        logo_bytes = _create_test_image(800, 600, color=(10, 50, 90))
        screen_bytes = _create_test_image(800, 600, color=(220, 180, 40))
        session = MagicMock(spec=requests.Session)
        session.cookies = MagicMock()

        def mock_get(url, *args, **kwargs):
            del args, kwargs
            content = screen_bytes if "screenshot" in url else logo_bytes
            return _mock_response(content)

        session.get.side_effect = mock_get

        repo = RepoData(
            url="https://github.com/foo/bar",
            owner="foo",
            name="bar",
            media=[
                MediaRef(
                    id="m1", url="https://example.com/logo.png", alt="Project logo", kind="image"
                ),
                MediaRef(
                    id="m2",
                    url="https://example.com/screenshot.png",
                    alt="Dashboard screenshot",
                    kind="image",
                ),
            ],
        )

        with patch("GRC.media.is_public_url", return_value=True):
            assets = collect_media(repo, tmp_path, Settings(), session=session, limit=1)

        assert len(assets) == 1
        assert assets[0].id == "m2"
        assert "screenshot" in assets[0].alt.lower()

    def test_two_clip_cap_and_limit(self, tmp_path: Path) -> None:
        """Collect media must respect max limit and cap at 2 clips."""
        media_dir = tmp_path / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        repo = RepoData(
            url="https://github.com/foo/bar",
            owner="foo",
            name="bar",
            media=[
                MediaRef(id="m1", url="https://example.com/clip1.mp4", kind="video"),
                MediaRef(id="m2", url="https://example.com/clip2.mp4", kind="video"),
                MediaRef(id="m3", url="https://example.com/clip3.mp4", kind="video"),
                MediaRef(id="m4", url="https://example.com/screen1.png", kind="image"),
                MediaRef(id="m5", url="https://example.com/screen2.png", kind="image"),
            ],
        )

        def mock_download(ref, dest_dir, session=None):
            del session
            ext = ".mp4" if ref.kind == "video" else ".png"
            p = dest_dir / f"{ref.id}{ext}"
            p.touch()
            return p

        def mock_clip(source_path, target_dir, asset_id):
            return (target_dir / f"{asset_id}.mp4", 1280, 720, 5.0)

        def mock_image(source_path, target_dir, asset_id, thumbs):
            return (target_dir / f"{asset_id}.png", 1280, 720)

        with (
            patch("GRC.media.download_media", side_effect=mock_download),
            patch("GRC.media.convert_to_clip", side_effect=mock_clip),
            patch("GRC.media.process_image", side_effect=mock_image),
        ):
            assets = collect_media(repo, tmp_path, Settings(), limit=4, force=True)

        assert len(assets) == 4
        clips = [a for a in assets if a.kind == "clip"]
        images = [a for a in assets if a.kind == "image"]
        assert len(clips) == 2  # Capped at 2 clips
        assert len(images) == 2

    def test_manifest_reuse_no_network(self, tmp_path: Path) -> None:
        """When manifest.json exists and all files are present, no network requests are made."""
        media_dir = tmp_path / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        img_file = media_dir / "m1.png"
        img_file.write_bytes(_create_test_image(600, 400))

        manifest_data = [
            {
                "id": "m1",
                "kind": "image",
                "path": "m1.png",
                "width": 600,
                "height": 400,
                "duration_seconds": 0.0,
                "alt": "Cached image",
                "origin": "readme",
                "source_url": "https://example.com/m1.png",
            }
        ]
        (media_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        session = MagicMock(spec=requests.Session)
        session.get.side_effect = RuntimeError("Network called when cache should be used!")

        repo = RepoData(
            url="https://github.com/foo/bar",
            owner="foo",
            name="bar",
            media=[MediaRef(id="m1", url="https://example.com/m1.png", kind="image")],
        )

        assets = collect_media(repo, tmp_path, Settings(), session=session, force=False)
        assert len(assets) == 1
        assert assets[0].id == "m1"
        assert assets[0].alt == "Cached image"
        session.get.assert_not_called()


class TestAttributionCreditLine:
    """Tests for credit_line attribution generator."""

    def test_credit_line_variants(self, tmp_path: Path) -> None:
        """Verify credit_line returns the appropriate string for all origin combinations."""
        # Empty
        assert credit_line([], "owner/repo") == ""

        # Readme only
        a_readme = [
            MediaAsset(
                id="m1",
                kind="image",
                path=tmp_path / "m1.png",
                width=800,
                height=600,
                origin="readme",
            )
        ]
        assert credit_line(a_readme, "owner/repo") == "Visuals from the owner/repo README"

        # Homepage only
        a_home = [
            MediaAsset(
                id="h1",
                kind="image",
                path=tmp_path / "h1.png",
                width=800,
                height=600,
                origin="homepage",
            )
        ]
        assert credit_line(a_home, "owner/repo") == "Visuals from the owner/repo homepage"

        # Both
        assert (
            credit_line(a_readme + a_home, "owner/repo")
            == "Visuals from the owner/repo README and homepage"
        )


class _TestHttpHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP server handler for Playwright capture tests."""

    def do_GET(self) -> None:
        if self.path == "/bot-wall":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><head><title>Verify you are human</title></head><body>Captcha</body></html>"
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = """<!DOCTYPE html>
        <html>
        <head><title>Project Showcase</title></head>
        <body style="margin: 0; background: #1a202c; color: #fff; font-family: sans-serif;">
            <div style="height: 1000px; padding: 40px; background: linear-gradient(to bottom, #2b6cb0, #2d3748);">
                <h1>Main Dashboard Showcase</h1>
                <p>Detailed performance analytics and reporting.</p>
            </div>
            <div style="height: 1200px; padding: 40px; background: linear-gradient(to bottom, #2d3748, #1a202c);">
                <h2>Deep Architectural Overview</h2>
                <p>System components and workflows.</p>
            </div>
        </body>
        </html>"""
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        del format, args  # Suppress server logs during test run


@pytest.fixture(scope="module")
def local_web_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _TestHttpHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.mark.browser
class TestHomepageCapture:
    """Playwright-based homepage capture tests."""

    def test_capture_homepage_success(self, local_web_server: str, tmp_path: Path) -> None:
        """Capture homepage screenshots (h1 and h2) against local test server."""
        assets = capture_homepage(
            local_web_server,
            tmp_path,
            Settings(),
            max_shots=2,
            _url_validator=lambda u: True,
        )
        assert len(assets) == 2
        assert assets[0].id == "h1"
        assert assets[0].origin == "homepage"
        assert assets[0].width == 1440
        assert assets[0].height == 900
        assert assets[0].path.exists()

        assert assets[1].id == "h2"
        assert assets[1].origin == "homepage"
        assert assets[1].width == 1440
        assert assets[1].height == 900
        assert assets[1].path.exists()

    def test_capture_homepage_bot_wall_skipped(self, local_web_server: str, tmp_path: Path) -> None:
        """A page containing bot wall keywords must be skipped."""
        assets = capture_homepage(
            f"{local_web_server}/bot-wall",
            tmp_path,
            Settings(),
            max_shots=2,
            _url_validator=lambda u: True,
        )
        assert assets == []


@pytest.mark.ffmpeg
class TestClipConversion:
    """FFmpeg-based GIF to MP4 conversion tests."""

    def test_gif_to_clip_conversion(self, tmp_path: Path) -> None:
        """Convert a 3-second animated GIF into a normalized MP4 clip."""
        frames = [
            Image.new("RGB", (640, 480), color=((i * 15) % 256, (i * 20) % 256, 150))
            for i in range(15)
        ]
        gif_path = tmp_path / "demo.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)

        res = convert_to_clip(gif_path, tmp_path, "m1")
        assert res is not None
        clip_path, width, height, duration = res

        assert clip_path.exists()
        assert clip_path.suffix == ".mp4"
        assert width % 2 == 0
        assert height % 2 == 0
        assert width <= 1600
        assert height <= 800
        assert 2.5 <= duration <= 3.5
