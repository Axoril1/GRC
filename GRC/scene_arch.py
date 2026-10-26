"""Animated architecture diagram scene builder with deterministic DAG layout."""

from __future__ import annotations

import html
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from GRC.anim import wrap_page
from GRC.errors import SceneError
from GRC.models import ArchitectureSpec, EdgeSpec, NodeSpec, Scene
from GRC.scenes import DEFAULT_ACCENT, FOOTER_CSS, footer_html, theme_css

if TYPE_CHECKING:
    from GRC.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "EdgePath",
    "GraphLayout",
    "NodeBox",
    "build_arch_html",
    "layout_graph",
]


@dataclass(frozen=True)
class NodeBox:
    """Bounding box and presentation metadata for an architecture node.

    Attributes:
        id: Unique node identifier.
        x: X-coordinate of top-left corner in visual area pixels.
        y: Y-coordinate of top-left corner in visual area pixels.
        w: Width of node box in pixels.
        h: Height of node box in pixels.
        label: Text label of the component.
        group: Group category ('input', 'core', 'output', 'storage', 'external').
        layer: Layer index assigned by topological layering.
    """

    id: str
    x: float
    y: float
    w: float
    h: float
    label: str
    group: str
    layer: int


@dataclass(frozen=True)
class EdgePath:
    """Cubic Bezier path and arrowhead geometry for a directed connection.

    Attributes:
        source: Origin node id.
        target: Destination node id.
        label: Text label on the connection (may be empty).
        start: Starting point (x, y).
        c1: First cubic Bezier control point (x, y).
        c2: Second cubic Bezier control point (x, y).
        end: Terminal point (x, y).
        label_pos: Center point for label pill (x, y).
        svg_d: SVG path 'd' attribute string.
        arrow_points: Tuple of 3 points defining the arrowhead triangle.
    """

    source: str
    target: str
    label: str
    start: tuple[float, float]
    c1: tuple[float, float]
    c2: tuple[float, float]
    end: tuple[float, float]
    label_pos: tuple[float, float]
    svg_d: str
    arrow_points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class GraphLayout:
    """Complete computed diagram layout with boxes, paths, and scaling.

    Attributes:
        nodes: Positioned node boxes.
        edges: Routed edge Bezier paths.
        orientation: Diagram flow direction ('LR' for left-to-right or 'TB' for top-to-bottom).
        scale: Uniform scaling factor applied to fit inside area.
    """

    nodes: list[NodeBox]
    edges: list[EdgePath]
    orientation: str
    scale: float


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def _find_cycles_and_back_edges(
    nodes: list[NodeSpec], edges: list[EdgeSpec]
) -> set[tuple[str, str]]:
    """Identify back edges using depth-first search cycle detection.

    Args:
        nodes: List of diagram nodes.
        edges: List of directed edges.

    Returns:
        Set of (source, target) edge pairs identified as back edges.
    """
    adj: dict[str, list[str]] = {n.id: [] for n in nodes}
    for e in edges:
        if e.source in adj and e.target in adj:
            adj[e.source].append(e.target)

    # Deterministic visit order
    for targets in adj.values():
        targets.sort()

    state: dict[str, int] = {n.id: 0 for n in nodes}  # 0: unvisited, 1: visiting, 2: visited
    back_edges: set[tuple[str, str]] = set()

    def dfs(u: str) -> None:
        state[u] = 1
        for v in adj[u]:
            if state[v] == 1:
                back_edges.add((u, v))
            elif state[v] == 0:
                dfs(v)
        state[u] = 2

    for node in nodes:
        if state[node.id] == 0:
            dfs(node.id)

    return back_edges


def _layer_dag(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
    back_edges: set[tuple[str, str]],
) -> dict[str, int]:
    """Compute longest-path topological layers from source nodes.

    Back edges are reversed for layering purposes so all layering edges point forward.
    Nodes with no edges are assigned to the last layer.

    Args:
        nodes: Diagram nodes.
        edges: Directed edges.
        back_edges: Set of detected cycle back edges.

    Returns:
        Mapping of node id to 0-based layer index.
    """
    dag_adj: dict[str, list[str]] = {n.id: [] for n in nodes}
    dag_in_deg: dict[str, int] = {n.id: 0 for n in nodes}

    connected_nodes: set[str] = set()
    for e in edges:
        if e.source not in dag_adj or e.target not in dag_adj:
            continue
        connected_nodes.add(e.source)
        connected_nodes.add(e.target)
        if (e.source, e.target) in back_edges:
            dag_adj[e.target].append(e.source)
            dag_in_deg[e.source] += 1
        else:
            dag_adj[e.source].append(e.target)
            dag_in_deg[e.target] += 1

    # Sources are connected nodes with in-degree 0
    sources = [n.id for n in nodes if n.id in connected_nodes and dag_in_deg[n.id] == 0]
    if not sources and connected_nodes:
        # Fallback if any anomalies
        sources = [min(connected_nodes)]

    # Longest path via topological processing
    layer_map: dict[str, int] = {s: 0 for s in sources}
    # Track incoming degrees for standard Kahn processing
    cur_in_deg = dict(dag_in_deg)
    queue = list(sources)

    while queue:
        u = queue.pop(0)
        u_layer = layer_map.get(u, 0)
        for v in sorted(dag_adj[u]):
            layer_map[v] = max(layer_map.get(v, 0), u_layer + 1)
            cur_in_deg[v] -= 1
            if cur_in_deg[v] == 0:
                queue.append(v)

    max_layer = max(layer_map.values(), default=0)

    # Isolated nodes go into the last layer
    for n in nodes:
        if n.id not in layer_map:
            layer_map[n.id] = max_layer

    return layer_map


def _barycenter_order(
    layers: list[list[str]],
    edges: list[EdgeSpec],
    back_edges: set[tuple[str, str]],
) -> list[list[str]]:
    """Refine node ordering within layers using two downward and two upward sweeps.

    Args:
        layers: List of node ID lists representing layers.
        edges: Graph edges.
        back_edges: Back edges reversed in the DAG.

    Returns:
        New list of node ID lists ordered by barycenter heuristic.
    """
    dag_preds: dict[str, list[str]] = {}
    dag_succs: dict[str, list[str]] = {}
    for layer in layers:
        for nid in layer:
            dag_preds[nid] = []
            dag_succs[nid] = []

    for e in edges:
        if e.source not in dag_preds or e.target not in dag_preds:
            continue
        if (e.source, e.target) in back_edges:
            u, v = e.target, e.source
        else:
            u, v = e.source, e.target
        dag_succs[u].append(v)
        dag_preds[v].append(u)

    cur_layers = [list(layer) for layer in layers]
    num_layers = len(cur_layers)

    def sweep_down() -> None:
        for L in range(1, num_layers):
            prev_order = {nid: idx for idx, nid in enumerate(cur_layers[L - 1])}
            barys: list[tuple[float, int, str]] = []
            for idx, nid in enumerate(cur_layers[L]):
                preds = [prev_order[p] for p in dag_preds[nid] if p in prev_order]
                avg = sum(preds) / len(preds) if preds else float(idx)
                barys.append((avg, idx, nid))
            barys.sort(key=lambda item: (item[0], item[1]))
            cur_layers[L] = [item[2] for item in barys]

    def sweep_up() -> None:
        for L in range(num_layers - 2, -1, -1):
            next_order = {nid: idx for idx, nid in enumerate(cur_layers[L + 1])}
            barys: list[tuple[float, int, str]] = []
            for idx, nid in enumerate(cur_layers[L]):
                succs = [next_order[s] for s in dag_succs[nid] if s in next_order]
                avg = sum(succs) / len(succs) if succs else float(idx)
                barys.append((avg, idx, nid))
            barys.sort(key=lambda item: (item[0], item[1]))
            cur_layers[L] = [item[2] for item in barys]

    # Two downward and two upward sweeps
    sweep_down()
    sweep_up()
    sweep_down()
    sweep_up()

    return cur_layers


def layout_graph(
    spec: ArchitectureSpec,
    area_w: int = 1680,
    area_h: int = 740,
) -> GraphLayout:
    """Compute a deterministic, layered hierarchical layout for an architecture diagram.

    Algorithm:
    1. Detect cycles via DFS and reverse back edges for DAG layering.
    2. Assign layers using longest path from sources.
    3. Order nodes within layers with barycenter heuristic (2 down, 2 up sweeps).
    4. Select orientation: 'LR' if 4 or fewer layers, else 'TB'.
    5. Node size: width = clamp(len(label) * 18 + 80, 220, 360), height = 96.
       Layer spacing: 120px; node spacing: 40px.
    6. Center and scale uniformly to fit (0.6 <= scale <= 1.0). If scale < 0.6,
       iteratively drop lowest-degree nodes until it fits.
    7. Route cubic Bezier curves with tangent-oriented triangular arrowheads.

    Args:
        spec: ArchitectureSpec containing nodes and edges.
        area_w: Available diagram visual width in pixels (default: 1680).
        area_h: Available diagram visual height in pixels (default: 740).

    Returns:
        GraphLayout containing positioned NodeBoxes and EdgePaths.

    Raises:
        SceneError: If spec has no nodes.
    """
    if not spec.nodes:
        raise SceneError(
            "Architecture diagram specification has no nodes.",
            hint="Ensure ArchitectureSpec has at least 3 nodes.",
        )

    working_nodes = list(spec.nodes)
    working_edges = list(spec.edges)

    while True:
        node_map = {n.id: n for n in working_nodes}
        # Degree calculation
        degrees: dict[str, int] = {n.id: 0 for n in working_nodes}
        valid_edges: list[EdgeSpec] = []
        for e in working_edges:
            if e.source in node_map and e.target in node_map:
                degrees[e.source] += 1
                degrees[e.target] += 1
                valid_edges.append(e)

        # 1. Cycle detection
        back_edges = _find_cycles_and_back_edges(working_nodes, valid_edges)

        # 2. Layering
        layer_map = _layer_dag(working_nodes, valid_edges, back_edges)
        num_layers = max(layer_map.values()) + 1

        raw_layers: list[list[str]] = [[] for _ in range(num_layers)]
        for n in working_nodes:
            raw_layers[layer_map[n.id]].append(n.id)

        # 3. Barycenter ordering
        ordered_layers = _barycenter_order(raw_layers, valid_edges, back_edges)

        # 4. Orientation: 4 or fewer layers -> LR; otherwise TB
        orientation = "LR" if num_layers <= 4 else "TB"

        # 5. Dimensions and natural bounding box
        node_w_map = {n.id: _clamp(len(n.label) * 18.0 + 80.0, 220.0, 360.0) for n in working_nodes}
        node_h = 96.0
        layer_spacing = 120.0
        node_spacing = 40.0

        if orientation == "LR":
            # Layers span X, nodes within layer span Y
            col_widths = [max(node_w_map[nid] for nid in layer) for layer in ordered_layers]
            natural_w = sum(col_widths) + max(0, num_layers - 1) * layer_spacing
            natural_h = max(
                len(layer) * node_h + max(0, len(layer) - 1) * node_spacing
                for layer in ordered_layers
            )
        else:
            # Layers span Y, nodes within layer span X
            layer_heights = [node_h for _ in ordered_layers]
            natural_h = sum(layer_heights) + max(0, num_layers - 1) * layer_spacing
            natural_w = max(
                sum(node_w_map[nid] for nid in layer) + max(0, len(layer) - 1) * node_spacing
                for layer in ordered_layers
            )

        # Scale calculation
        scale_val = min(1.0, area_w / max(1.0, natural_w), area_h / max(1.0, natural_h))

        if scale_val >= 0.6 or len(working_nodes) <= 3:
            final_scale = max(0.6, scale_val)
            break

        # Scale is too small: drop lowest-degree node
        sorted_by_degree = sorted(
            working_nodes,
            key=lambda n: (degrees[n.id], -len(n.label), n.id),
        )
        dropped = sorted_by_degree[0]
        logger.warning(
            "Architecture diagram natural size (%dx%d) exceeds area; dropping lowest-degree node '%s' (degree %d) to fit within canvas.",
            round(natural_w),
            round(natural_h),
            dropped.id,
            degrees[dropped.id],
        )
        working_nodes = [n for n in working_nodes if n.id != dropped.id]
        working_edges = [
            e for e in working_edges if e.source != dropped.id and e.target != dropped.id
        ]

    # Compute unscaled coordinates
    unscaled_coords: dict[str, tuple[float, float, float, float]] = {}

    if orientation == "LR":
        cur_x = 0.0
        for l_idx, layer in enumerate(ordered_layers):
            c_w = col_widths[l_idx]
            l_h = len(layer) * node_h + max(0, len(layer) - 1) * node_spacing
            cur_y = (natural_h - l_h) / 2.0
            for nid in layer:
                nw = node_w_map[nid]
                nx = cur_x + (c_w - nw) / 2.0
                unscaled_coords[nid] = (nx, cur_y, nw, node_h)
                cur_y += node_h + node_spacing
            cur_x += c_w + layer_spacing
    else:
        cur_y = 0.0
        for layer in ordered_layers:
            l_w = sum(node_w_map[nid] for nid in layer) + max(0, len(layer) - 1) * node_spacing
            cur_x = (natural_w - l_w) / 2.0
            for nid in layer:
                nw = node_w_map[nid]
                unscaled_coords[nid] = (cur_x, cur_y, nw, node_h)
                cur_x += nw + node_spacing
            cur_y += node_h + layer_spacing

    # Scale and center in area
    scaled_w = natural_w * final_scale
    scaled_h = natural_h * final_scale
    offset_x = (area_w - scaled_w) / 2.0
    offset_y = (area_h - scaled_h) / 2.0

    boxes: list[NodeBox] = []
    box_map: dict[str, NodeBox] = {}

    for n in working_nodes:
        ux, uy, uw, uh = unscaled_coords[n.id]
        bx = round(offset_x + ux * final_scale, 1)
        by = round(offset_y + uy * final_scale, 1)
        bw = round(uw * final_scale, 1)
        bh = round(uh * final_scale, 1)
        box = NodeBox(
            id=n.id,
            x=bx,
            y=by,
            w=bw,
            h=bh,
            label=n.label,
            group=n.group,
            layer=layer_map[n.id],
        )
        boxes.append(box)
        box_map[n.id] = box

    # Route edge Bezier curves
    edges_out: list[EdgePath] = []
    arrow_len = 12.0 * final_scale
    arrow_width = 8.0 * final_scale

    for e in valid_edges:
        if e.source not in box_map or e.target not in box_map:
            continue
        src = box_map[e.source]
        tgt = box_map[e.target]

        if orientation == "LR":
            # Left-to-right flow
            start_pt = (src.x + src.w, src.y + src.h / 2.0)
            end_pt = (tgt.x, tgt.y + tgt.h / 2.0)
            dx = end_pt[0] - start_pt[0]
            if dx >= 0:
                c1 = (start_pt[0] + 0.40 * dx, start_pt[1])
                c2 = (end_pt[0] - 0.40 * dx, end_pt[1])
            else:
                # Cycle / back edge loop
                loop_h = 100.0 * final_scale
                c1 = (start_pt[0] + 60.0 * final_scale, start_pt[1] - loop_h)
                c2 = (end_pt[0] - 60.0 * final_scale, end_pt[1] - loop_h)
        else:
            # Top-to-bottom flow
            start_pt = (src.x + src.w / 2.0, src.y + src.h)
            end_pt = (tgt.x + tgt.w / 2.0, tgt.y)
            dy = end_pt[1] - start_pt[1]
            if dy >= 0:
                c1 = (start_pt[0], start_pt[1] + 0.40 * dy)
                c2 = (end_pt[0], end_pt[1] - 0.40 * dy)
            else:
                loop_w = 100.0 * final_scale
                c1 = (start_pt[0] + loop_w, start_pt[1] + 60.0 * final_scale)
                c2 = (end_pt[0] + loop_w, end_pt[1] - 60.0 * final_scale)

        # SVG cubic Bezier path
        svg_d = (
            f"M {start_pt[0]:.1f} {start_pt[1]:.1f} "
            f"C {c1[0]:.1f} {c1[1]:.1f}, {c2[0]:.1f} {c2[1]:.1f}, {end_pt[0]:.1f} {end_pt[1]:.1f}"
        )

        # Midpoint B(0.5)
        mid_x = 0.125 * start_pt[0] + 0.375 * c1[0] + 0.375 * c2[0] + 0.125 * end_pt[0]
        mid_y = 0.125 * start_pt[1] + 0.375 * c1[1] + 0.375 * c2[1] + 0.125 * end_pt[1]
        label_pos = (round(mid_x, 1), round(mid_y, 1))

        # Arrowhead triangle oriented along final tangent (end_pt - c2)
        tangent_x = end_pt[0] - c2[0]
        tangent_y = end_pt[1] - c2[1]
        t_norm = math.hypot(tangent_x, tangent_y)
        if t_norm > 1e-4:
            tx = tangent_x / t_norm
            ty = tangent_y / t_norm
        else:
            tx, ty = (1.0, 0.0) if orientation == "LR" else (0.0, 1.0)

        # Normal vector perpendicular to tangent
        nx, ny = -ty, tx
        base_x = end_pt[0] - tx * arrow_len
        base_y = end_pt[1] - ty * arrow_len
        w2 = arrow_width / 2.0

        p_tip = (round(end_pt[0], 1), round(end_pt[1], 1))
        p_left = (round(base_x + nx * w2, 1), round(base_y + ny * w2, 1))
        p_right = (round(base_x - nx * w2, 1), round(base_y - ny * w2, 1))

        edges_out.append(
            EdgePath(
                source=e.source,
                target=e.target,
                label=e.label,
                start=(round(start_pt[0], 1), round(start_pt[1], 1)),
                c1=(round(c1[0], 1), round(c1[1], 1)),
                c2=(round(c2[0], 1), round(c2[1], 1)),
                end=(round(end_pt[0], 1), round(end_pt[1], 1)),
                label_pos=label_pos,
                svg_d=svg_d,
                arrow_points=(p_tip, p_left, p_right),
            )
        )

    return GraphLayout(
        nodes=boxes,
        edges=edges_out,
        orientation=orientation,
        scale=round(final_scale, 3),
    )


def build_arch_html(
    scene: Scene,
    spec: ArchitectureSpec | None = None,
    *,
    theme: str = "dark",
    accent: str = DEFAULT_ACCENT,
    repo_label: str = "",
    index: int = 1,
    total: int = 1,
    duration: float = 5.0,
    settings: Settings | None = None,
) -> str:
    """Build a complete animated HTML document for an architecture diagram scene.

    Renders positioned node cards and SVG Bezier connections with progressive
    layer-by-layer entrances, path drawing, and a traveling accent pulse.

    Args:
        scene: Scene definition containing heading (and optional architecture).
        spec: ArchitectureSpec defining diagram nodes and edges (defaults to scene.architecture).
        theme: Color theme, either 'dark' or 'light'.
        accent: Hex accent color for highlights and accents.
        repo_label: Repository name or label for the footer.
        index: Current scene index (1-based).
        total: Total number of scenes.
        duration: Scene duration in seconds.
        settings: Application settings.

    Returns:
        Complete HTML document string.
    """
    del settings
    target_spec = spec or scene.architecture
    if target_spec is None:
        raise SceneError(
            "Architecture scene requires an ArchitectureSpec.",
            hint="Set scene.architecture or pass spec explicitly.",
        )

    layout = layout_graph(target_spec, area_w=1680, area_h=740)

    escaped_heading = html.escape(scene.heading)
    footer_markup = footer_html(repo_label, index, total)

    # Distinct groups present for legend
    distinct_groups = sorted({n.group for n in layout.nodes})

    # Render node HTML elements
    node_elements: list[str] = []
    for n in layout.nodes:
        escaped_label = html.escape(n.label)
        font_sz = round(max(20.0, 28.0 * layout.scale), 1)
        node_elements.append(
            f'<div id="node-{n.id}" class="arch-node group-{n.group}" '
            f'data-layer="{n.layer}" '
            f'style="left: {n.x}px; top: {n.y}px; width: {n.w}px; height: {n.h}px; font-size: {font_sz}px;">'
            f'<span class="node-label">{escaped_label}</span>'
            f"</div>"
        )
    nodes_html = "\n".join(node_elements)

    # Render SVG edges
    svg_paths: list[str] = []
    svg_markers: list[str] = []
    svg_labels: list[str] = []

    for i, e in enumerate(layout.edges):
        p_tip, p_left, p_right = e.arrow_points
        pts_str = f"{p_tip[0]},{p_tip[1]} {p_left[0]},{p_left[1]} {p_right[0]},{p_right[1]}"

        stroke_w = max(2.0, 3.5 * layout.scale)
        svg_paths.append(
            f'<path id="edge-path-{i}" class="arch-edge" d="{e.svg_d}" '
            f'fill="none" stroke-width="{stroke_w:.1f}" />'
        )
        svg_markers.append(f'<polygon id="edge-arrow-{i}" class="arch-arrow" points="{pts_str}" />')

        if e.label:
            escaped_el = html.escape(e.label)
            font_size = round(max(14.0, 20.0 * layout.scale), 1)
            svg_labels.append(
                f'<g id="edge-label-{i}" class="arch-edge-label" transform="translate({e.label_pos[0]}, {e.label_pos[1]})">'
                f'<rect x="-50" y="-16" width="100" height="32" rx="8" fill="var(--surface)" stroke="var(--border)" />'
                f'<text x="0" y="5" text-anchor="middle" font-size="{font_size}px" fill="var(--muted)">{escaped_el}</text>'
                f"</g>"
            )

    svg_content = "\n".join(svg_paths + svg_markers + svg_labels)

    # Legend HTML if 3 or more groups present
    legend_html = ""
    if len(distinct_groups) >= 3:
        legend_items: list[str] = []
        for g in distinct_groups:
            legend_items.append(
                f'<div class="legend-item"><span class="legend-dot group-{g}"></span>'
                f'<span class="legend-text">{html.escape(g.capitalize())}</span></div>'
            )
        legend_html = f'<div class="arch-legend">{"".join(legend_items)}</div>'

    body_html = f"""
<div class="scene-container">
  <div class="scene-heading">{escaped_heading}</div>
  <div class="visual-area">
    <div id="arch-card" class="arch-card">
      <div class="card-bar">
        <div class="card-dots">
          <span class="dot red"></span>
          <span class="dot yellow"></span>
          <span class="dot green"></span>
        </div>
        <div class="card-title">Architecture</div>
        {legend_html}
      </div>
      <div class="diagram-canvas">
        <svg class="arch-svg" viewBox="0 0 1680 660">
          {svg_content}
        </svg>
        {nodes_html}
      </div>
    </div>
  </div>
  {footer_markup}
</div>
"""

    css = f"""
{FOOTER_CSS}

body {{
  width: 1920px;
  height: 1080px;
  overflow: hidden;
  background-color: var(--bg);
  background-image: radial-gradient(circle at 90% 10%, var(--accent-soft) 0%, transparent 65%);
  color: var(--text);
  font-family: Inter, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.35;
}}

.scene-container {{
  width: 1920px;
  height: 1080px;
  position: relative;
  overflow: hidden;
}}

.scene-heading {{
  position: absolute;
  top: 120px;
  left: 120px;
  right: 120px;
  height: 68px;
  font-size: 56px;
  font-weight: 700;
  line-height: 1.2;
  color: var(--text);
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}}

.visual-area {{
  position: absolute;
  top: 200px;
  left: 120px;
  width: 1680px;
  height: 760px;
}}

.arch-card {{
  width: 100%;
  height: 100%;
  border-radius: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  box-shadow: 0 20px 48px rgba(0, 0, 0, 0.35);
  opacity: 0;
  transform: translateY(24px);
}}

.card-bar {{
  height: 60px;
  min-height: 60px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  background: rgba(0, 0, 0, 0.15);
  border-bottom: 1px solid var(--border);
  position: relative;
}}

.card-dots {{
  display: flex;
  align-items: center;
  gap: 10px;
}}

.dot {{
  width: 16px;
  height: 16px;
  border-radius: 50%;
}}
.dot.red {{ background: #ff5f56; }}
.dot.yellow {{ background: #ffbd2e; }}
.dot.green {{ background: #27c93f; }}

.card-title {{
  font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
  font-size: 24px;
  color: var(--muted);
  margin-left: 20px;
  font-weight: 600;
}}

.arch-legend {{
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 20px;
}}

.legend-item {{
  display: flex;
  align-items: center;
  gap: 8px;
}}

.legend-dot {{
  width: 12px;
  height: 12px;
  border-radius: 50%;
}}
.legend-dot.group-input {{ background: var(--accent); }}
.legend-dot.group-core {{ background: var(--text); }}
.legend-dot.group-output {{ background: #10b981; }}
.legend-dot.group-storage {{ background: #f59e0b; }}
.legend-dot.group-external {{ background: #6b7280; }}

.legend-text {{
  font-size: 18px;
  color: var(--muted);
  font-weight: 500;
}}

.diagram-canvas {{
  flex: 1;
  position: relative;
  overflow: hidden;
}}

.arch-svg {{
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}}

.arch-edge {{
  stroke: var(--muted);
  opacity: 0.6;
}}

.arch-arrow {{
  fill: var(--muted);
  opacity: 0;
}}

.arch-edge-label {{
  opacity: 0;
  font-family: 'JetBrains Mono', monospace;
}}

.arch-node {{
  position: absolute;
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 12px 20px;
  font-weight: 700;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
  transition: box-shadow 0.2s ease;
  user-select: none;
  opacity: 0;
  transform: scale(0.9);
}}

/* Group color schemes */
.arch-node.group-input {{
  background: var(--accent);
  color: #ffffff;
  border: 2px solid var(--accent);
}}

.arch-node.group-core {{
  background: var(--surface);
  color: var(--text);
  border: 3px solid var(--accent);
}}

.arch-node.group-output {{
  background: rgba(16, 185, 129, 0.18);
  color: #10b981;
  border: 2px solid #10b981;
}}

.arch-node.group-storage {{
  background: rgba(245, 158, 11, 0.18);
  color: #f59e0b;
  border: 2px solid #f59e0b;
}}

.arch-node.group-external {{
  background: rgba(107, 114, 128, 0.18);
  color: var(--muted);
  border: 2px solid #6b7280;
}}

.node-label {{
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.slide-footer {{
  position: absolute;
  bottom: 60px;
  left: 120px;
  right: 120px;
}}
"""

    # Animation timing data
    max_layer = max((n.layer for n in layout.nodes), default=0)
    num_layers = max_layer + 1

    # Map node list in topological order for pulse phase
    nodes_topo = sorted(layout.nodes, key=lambda n: (n.layer, n.x, n.y))
    node_ids_json = [n.id for n in nodes_topo]

    scene_js = f"""
const NUM_LAYERS = {num_layers};
const EDGE_COUNT = {len(layout.edges)};
const NODE_IDS = {node_ids_json};

window.__setTime = function(t) {{
  const D = window.__DURATION || {duration};
  const card = document.getElementById("arch-card");

  // 1. Entrance animation (0.0 to 0.5s)
  const pEntry = progress(t, 0, 0.5);
  const eEntry = ease.outCubic(pEntry);
  if (card) {{
    card.style.opacity = eEntry;
    card.style.transform = "translateY(" + lerp(24, 0, eEntry) + "px)";
  }}

  // 2. Nodes entrance (0.08*D to 0.55*D) layer by layer
  const tNodeStart = 0.08 * D;
  const tNodeEnd = 0.55 * D;
  for (let i = 0; i < NODE_IDS.length; i++) {{
    const el = document.getElementById("node-" + NODE_IDS[i]);
    if (!el) continue;
    const layer = parseInt(el.getAttribute("data-layer") || "0", 10);
    const layerDelay = (NUM_LAYERS > 1) ? (layer / (NUM_LAYERS - 1)) : 0;
    const lStart = lerp(tNodeStart, Math.max(tNodeStart, tNodeEnd - 0.40), layerDelay);
    const pNode = progress(t, lStart, lStart + 0.40);
    const eNode = ease.outCubic(pNode);

    el.style.opacity = eNode;
    el.style.transform = "scale(" + lerp(0.90, 1.0, eNode) + ")";
  }}

  // 3. Edges drawing (0.40*D to 0.80*D)
  const tEdgeStart = 0.40 * D;
  const tEdgeEnd = 0.80 * D;
  const pEdges = ease.inOutCubic(progress(t, tEdgeStart, tEdgeEnd));

  for (let j = 0; j < EDGE_COUNT; j++) {{
    const pathEl = document.getElementById("edge-path-" + j);
    const arrowEl = document.getElementById("edge-arrow-" + j);
    const labelEl = document.getElementById("edge-label-" + j);
    if (!pathEl) continue;

    // Estimate path length if getTotalLength is unavailable
    const pathLen = (pathEl.getTotalLength ? pathEl.getTotalLength() : 600);
    pathEl.style.strokeDasharray = pathLen;
    pathEl.style.strokeDashoffset = lerp(pathLen, 0, pEdges);

    // Arrowhead and label appear when line is 90% drawn
    const isDrawn = (pEdges >= 0.90);
    const markerOpacity = progress(pEdges, 0.90, 1.0);
    if (arrowEl) arrowEl.style.opacity = markerOpacity;
    if (labelEl) labelEl.style.opacity = markerOpacity;
  }}

  // 4. Accent highlight pulse through nodes in layer order (0.80*D to 0.98*D)
  const tPulseStart = 0.80 * D;
  const tPulseEnd = 0.98 * D;
  const pulseDur = (tPulseEnd - tPulseStart) / Math.max(1, NODE_IDS.length);

  for (let i = 0; i < NODE_IDS.length; i++) {{
    const el = document.getElementById("node-" + NODE_IDS[i]);
    if (!el) continue;

    const pNodeStart = tPulseStart + i * pulseDur;
    const pNodeEnd = pNodeStart + pulseDur;

    if (t >= pNodeStart && t < pNodeEnd) {{
      const pGlow = Math.sin(progress(t, pNodeStart, pNodeEnd) * Math.PI);
      el.style.boxShadow = "0 0 " + lerp(12, 32, pGlow) + "px var(--accent)";
      el.style.borderColor = "var(--accent)";
    }} else {{
      el.style.boxShadow = "0 10px 24px rgba(0, 0, 0, 0.2)";
    }}
  }}
}};
"""

    return wrap_page(body_html, scene_js, css=css, theme_css=theme_css(theme, accent))
