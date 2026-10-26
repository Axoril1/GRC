"""Tests for repo2video/scene_arch.py architecture diagram scene rendering."""

from __future__ import annotations

import io
import itertools

import pytest
from PIL import Image

from repo2video.anim import AnimationRenderer
from repo2video.models import ArchitectureSpec, EdgeSpec, NodeSpec, Scene
from repo2video.scene_arch import (
    NodeBox,
    build_arch_html,
    layout_graph,
)


def is_chromium_available() -> bool:
    """Check if Playwright Chromium browser is installed and launchable."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            b.close()
        return True
    except (PlaywrightError, OSError, RuntimeError):
        return False


skip_if_no_chromium = pytest.mark.skipif(
    not is_chromium_available(),
    reason="Playwright Chromium browser executable is not installed",
)


def _boxes_overlap(b1: NodeBox, b2: NodeBox) -> bool:
    """Return True if two node boxes overlap with a margin."""
    return not (
        (b1.x + b1.w <= b2.x)
        or (b2.x + b2.w <= b1.x)
        or (b1.y + b1.h <= b2.y)
        or (b2.y + b2.h <= b1.y)
    )


# ==============================================================================
# Non-Browser Unit Tests
# ==============================================================================


def test_layout_linear_chain() -> None:
    """Verify linear chain A -> B -> C assigns layers forward and produces LR layout."""
    spec = ArchitectureSpec(
        nodes=[
            NodeSpec(id="client", label="Client", group="input"),
            NodeSpec(id="server", label="Server API", group="core"),
            NodeSpec(id="db", label="Database", group="storage"),
        ],
        edges=[
            EdgeSpec(source="client", target="server", label="HTTP"),
            EdgeSpec(source="server", target="db", label="Query"),
        ],
    )

    layout = layout_graph(spec, area_w=1680, area_h=740)

    assert layout.orientation == "LR"
    assert len(layout.nodes) == 3
    assert len(layout.edges) == 2

    node_dict = {n.id: n for n in layout.nodes}
    assert node_dict["client"].layer == 0
    assert node_dict["server"].layer == 1
    assert node_dict["db"].layer == 2

    # In LR layout, x-coordinates must increase monotonically with layer
    assert node_dict["client"].x < node_dict["server"].x < node_dict["db"].x

    # Width and height positive
    for node in layout.nodes:
        assert node.w > 0
        assert node.h > 0


def test_layout_branching_graph() -> None:
    """Verify branching diamond graph A -> B, A -> C, B -> D, C -> D places D at layer 2."""
    spec = ArchitectureSpec(
        nodes=[
            NodeSpec(id="gateway", label="Gateway", group="input"),
            NodeSpec(id="auth", label="Auth Service", group="core"),
            NodeSpec(id="payment", label="Payment Service", group="core"),
            NodeSpec(id="ledger", label="Ledger DB", group="storage"),
        ],
        edges=[
            EdgeSpec(source="gateway", target="auth", label="req"),
            EdgeSpec(source="gateway", target="payment", label="req"),
            EdgeSpec(source="auth", target="ledger", label="log"),
            EdgeSpec(source="payment", target="ledger", label="transact"),
        ],
    )

    layout = layout_graph(spec)

    node_dict = {n.id: n for n in layout.nodes}
    assert node_dict["gateway"].layer == 0
    assert node_dict["auth"].layer == 1
    assert node_dict["payment"].layer == 1
    assert node_dict["ledger"].layer == 2

    # In layer 1, auth and payment must not overlap vertically
    assert not _boxes_overlap(node_dict["auth"], node_dict["payment"])


def test_layout_cycle_breaks_cleanly() -> None:
    """Verify cyclic graph A -> B -> C -> A detects cycle, breaks it, and finishes cleanly."""
    spec = ArchitectureSpec(
        nodes=[
            NodeSpec(id="agent", label="Agent", group="core"),
            NodeSpec(id="planner", label="Planner", group="core"),
            NodeSpec(id="executor", label="Executor", group="core"),
        ],
        edges=[
            EdgeSpec(source="agent", target="planner", label="goal"),
            EdgeSpec(source="planner", target="executor", label="plan"),
            EdgeSpec(source="executor", target="agent", label="feedback"),
        ],
    )

    layout = layout_graph(spec)

    assert len(layout.nodes) == 3
    assert len(layout.edges) == 3
    for node in layout.nodes:
        assert node.w > 0 and node.h > 0


def test_layout_isolated_node() -> None:
    """Verify diagram with an isolated node (1 edge total) lays out all nodes without error."""
    spec = ArchitectureSpec(
        nodes=[
            NodeSpec(id="client", label="Client", group="input"),
            NodeSpec(id="server", label="Server", group="core"),
            NodeSpec(id="cache", label="Redis Cache", group="storage"),
        ],
        edges=[
            EdgeSpec(source="client", target="server", label="calls"),
        ],
    )

    layout = layout_graph(spec)

    assert len(layout.nodes) == 3
    node_ids = {n.id for n in layout.nodes}
    assert node_ids == {"client", "server", "cache"}

    # No overlapping boxes
    for b1, b2 in itertools.combinations(layout.nodes, 2):
        assert not _boxes_overlap(b1, b2)


def test_layout_dense_8_nodes_12_edges() -> None:
    """Verify dense maximum-size graph (8 nodes, 12 edges) meets all layout constraints."""
    spec = ArchitectureSpec(
        nodes=[
            NodeSpec(id="ui", label="Web UI", group="input"),
            NodeSpec(id="mobile", label="Mobile App", group="input"),
            NodeSpec(id="gw", label="API Gateway", group="core"),
            NodeSpec(id="auth", label="Auth Service", group="core"),
            NodeSpec(id="worker", label="Task Worker", group="core"),
            NodeSpec(id="queue", label="Message Queue", group="storage"),
            NodeSpec(id="db", label="Primary DB", group="storage"),
            NodeSpec(id="cloud", label="Cloud Storage", group="external"),
        ],
        edges=[
            EdgeSpec(source="ui", target="gw", label="HTTPS"),
            EdgeSpec(source="mobile", target="gw", label="HTTPS"),
            EdgeSpec(source="gw", target="auth", label="verify"),
            EdgeSpec(source="gw", target="worker", label="dispatch"),
            EdgeSpec(source="worker", target="queue", label="enqueue"),
            EdgeSpec(source="queue", target="worker", label="consume"),
            EdgeSpec(source="auth", target="db", label="read"),
            EdgeSpec(source="worker", target="db", label="write"),
            EdgeSpec(source="worker", target="cloud", label="upload"),
            EdgeSpec(source="gw", target="queue", label="push"),
            EdgeSpec(source="ui", target="cloud", label="fetch assets"),
            EdgeSpec(source="mobile", target="cloud", label="fetch cdn"),
        ],
    )

    layout = layout_graph(spec, area_w=1680, area_h=740)

    assert len(layout.nodes) == 8
    assert layout.scale >= 0.6

    # Verify no overlaps between any pair of nodes
    for b1, b2 in itertools.combinations(layout.nodes, 2):
        assert not _boxes_overlap(b1, b2), f"Overlap between {b1.id} and {b2.id}"

    # Verify all nodes fit within area bounds
    for b in layout.nodes:
        assert b.x >= 0
        assert b.y >= 0
        assert b.x + b.w <= 1680
        assert b.y + b.h <= 740


def test_orientation_switch() -> None:
    """Verify orientation is LR when layers <= 4 and switches to TB when layers >= 5."""
    # 4 layers linear chain -> LR
    spec_4 = ArchitectureSpec(
        nodes=[
            NodeSpec(id="n1", label="Stage 1", group="input"),
            NodeSpec(id="n2", label="Stage 2", group="core"),
            NodeSpec(id="n3", label="Stage 3", group="core"),
            NodeSpec(id="n4", label="Stage 4", group="output"),
        ],
        edges=[
            EdgeSpec(source="n1", target="n2"),
            EdgeSpec(source="n2", target="n3"),
            EdgeSpec(source="n3", target="n4"),
        ],
    )
    layout_4 = layout_graph(spec_4)
    assert layout_4.orientation == "LR"

    # 5 layers linear chain -> TB
    spec_5 = ArchitectureSpec(
        nodes=[
            NodeSpec(id="n1", label="Stage 1", group="input"),
            NodeSpec(id="n2", label="Stage 2", group="core"),
            NodeSpec(id="n3", label="Stage 3", group="core"),
            NodeSpec(id="n4", label="Stage 4", group="core"),
            NodeSpec(id="n5", label="Stage 5", group="output"),
        ],
        edges=[
            EdgeSpec(source="n1", target="n2"),
            EdgeSpec(source="n2", target="n3"),
            EdgeSpec(source="n3", target="n4"),
            EdgeSpec(source="n4", target="n5"),
        ],
    )
    layout_5 = layout_graph(spec_5)
    assert layout_5.orientation == "TB"


def test_layout_determinism() -> None:
    """Verify layout_graph computes identical coordinates and paths across multiple calls."""
    spec = ArchitectureSpec(
        nodes=[
            NodeSpec(id="front", label="Frontend", group="input"),
            NodeSpec(id="backend", label="Backend", group="core"),
            NodeSpec(id="db", label="Postgres DB", group="storage"),
        ],
        edges=[
            EdgeSpec(source="front", target="backend", label="REST"),
            EdgeSpec(source="backend", target="db", label="SQL"),
        ],
    )

    layouts = [layout_graph(spec) for _ in range(5)]

    first = layouts[0]
    for other in layouts[1:]:
        assert [(n.x, n.y, n.w, n.h) for n in other.nodes] == [
            (n.x, n.y, n.w, n.h) for n in first.nodes
        ]
        assert [(e.source, e.target, e.start, e.end, e.svg_d) for e in other.edges] == [
            (e.source, e.target, e.start, e.end, e.svg_d) for e in first.edges
        ]


def test_build_arch_html_escapes_content() -> None:
    """Verify build_arch_html escapes HTML entities in heading, labels, and groups."""
    spec = ArchitectureSpec(
        nodes=[
            NodeSpec(id="in1", label="Node <script>", group="input"),
            NodeSpec(id="core1", label='Core & "Quotes"', group="core"),
            NodeSpec(id="out1", label="Output 'Safe'", group="output"),
        ],
        edges=[
            EdgeSpec(source="in1", target="core1", label="Send > Data"),
            EdgeSpec(source="core1", target="out1", label="Data & Output"),
        ],
    )
    scene = Scene(
        id=1,
        type="architecture",
        heading='Danger <script>alert("xss")</script> & Heading',
        narration="Here is a detailed explanation of the architecture diagram showcasing how all the components interact in our system.",
        architecture=spec,
    )

    html_out = build_arch_html(scene, duration=6.0)

    # Raw script tags must never appear
    assert "<script>alert" not in html_out
    assert "Node <script>" not in html_out
    # Escaped versions must be present
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html_out
    assert "Node &lt;script&gt;" in html_out
    assert "Core &amp; &quot;Quotes&quot;" in html_out
    assert "Send &gt; Data" in html_out


# ==============================================================================
# Browser-Marked Integration Tests
# ==============================================================================


@pytest.mark.browser
@skip_if_no_chromium
def test_architecture_scene_renders_and_animates() -> None:
    """Verify Playwright renders the architecture diagram with visible animated elements."""
    spec = ArchitectureSpec(
        nodes=[
            NodeSpec(id="user", label="Web Client", group="input"),
            NodeSpec(id="api", label="API Gateway", group="core"),
            NodeSpec(id="db", label="Postgres DB", group="storage"),
        ],
        edges=[
            EdgeSpec(source="user", target="api", label="HTTP POST"),
            EdgeSpec(source="api", target="db", label="Insert"),
        ],
    )
    scene = Scene(
        id=1,
        type="architecture",
        heading="System Architecture Overview",
        narration="Here is a detailed walkthrough of the end-to-end data flow through the architecture of our application.",
        architecture=spec,
    )

    duration = 6.0
    html_page = build_arch_html(scene, duration=duration, theme="dark", accent="#3b82f6")

    renderer = AnimationRenderer()
    frame_early = renderer.capture_frame(html_page, 0.2)
    frame_late = renderer.capture_frame(html_page, 4.5)
    renderer.close()

    img_early = Image.open(io.BytesIO(frame_early)).convert("L")
    img_late = Image.open(io.BytesIO(frame_late)).convert("L")

    # Dimensions must be Full HD
    assert img_early.size == (1920, 1080)
    assert img_late.size == (1920, 1080)

    # In the diagram visual area (x: 120 to 1800, y: 200 to 940)
    diagram_crop_early = img_early.crop((120, 200, 1800, 940))
    diagram_crop_late = img_late.crop((120, 200, 1800, 940))

    # Early frame has nodes at low opacity (~0.12 or fading in)
    # Late frame has fully visible nodes, badges, glow cycles, and edges
    early_pixels = sum(1 for p in diagram_crop_early.tobytes() if p > 50)
    late_pixels = sum(1 for p in diagram_crop_late.tobytes() if p > 50)

    assert late_pixels > early_pixels
