"""2D/3D terrain reconstruction from crater depth estimates."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go


def build_depth_map(
    image_shape: tuple[int, int],
    depth_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Construct a scene-wide depth map by blending crater depressions.

    Physics note:
    Each crater is modeled as a smooth bowl-shaped depression whose amplitude is
    set by estimated depth. Superposition of these local depressions forms a
    coarse digital elevation proxy suitable for hazard visualization.

    Performance note:
    For efficiency, each crater Gaussian is only computed within a local bounding
    box of ±3σ around the crater center, rather than across the entire image.
    This reduces computation from O(N × H × W) to O(N × local_patch_area).

    Args:
        image_shape: (height, width) of source scene.
        depth_rows: Per-crater depth records with center/radius/depth fields.

    Returns:
        Float32 depth map in meters where larger values indicate deeper terrain.
    """

    h, w = image_shape
    depth_map = np.zeros((h, w), dtype=np.float32)

    for row in depth_rows:
        cx = float(row.get("center_x", 0))
        cy = float(row.get("center_y", 0))
        radius = max(4.0, float(row.get("radius_px", 8)))
        depth_m = max(0.0, float(row.get("depth_m", 0.0)))

        if depth_m < 0.001:
            continue

        sigma = radius * 0.62
        extent = int(3.0 * sigma) + 1

        # Compute bounding box for local patch
        y0 = max(0, int(cy) - extent)
        y1 = min(h, int(cy) + extent + 1)
        x0 = max(0, int(cx) - extent)
        x1 = min(w, int(cx) + extent + 1)

        if y1 <= y0 or x1 <= x0:
            continue

        # Local coordinate grids
        yy_local, xx_local = np.mgrid[y0:y1, x0:x1]
        dx = xx_local - cx
        dy = yy_local - cy
        dist_sq = dx * dx + dy * dy

        crater = depth_m * np.exp(-dist_sq / (2.0 * sigma * sigma))
        depth_map[y0:y1, x0:x1] += crater.astype(np.float32)

    depth_map = cv2.GaussianBlur(depth_map, (0, 0), sigmaX=1.4, sigmaY=1.4)
    return depth_map


def _downsample_for_3d(depth_map: np.ndarray, target_size: int = 720) -> np.ndarray:
    """Downsample depth map for smooth 3D rendering with many craters.

    When the depth map is large or contains many features, rendering a full-res
    3D surface in Plotly becomes slow. This produces a manageable mesh while
    preserving crater morphology through area-based interpolation.

    Args:
        depth_map: Full-resolution depth map.
        target_size: Target max dimension for the downsampled map.

    Returns:
        Downsampled depth map suitable for 3D surface rendering.
    """

    h, w = depth_map.shape
    target_size = int(np.clip(target_size, 200, 1400))
    max_dim = max(h, w)
    if max_dim <= target_size:
        return depth_map

    scale = target_size / float(max_dim)
    new_w = max(32, int(round(w * scale)))
    new_h = max(32, int(round(h * scale)))
    return cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_heatmap_figure(depth_map: np.ndarray) -> go.Figure:
    """Create a dark-theme heatmap figure for terrain depth inspection.

    Args:
        depth_map: 2D depth map in meters.

    Returns:
        Plotly heatmap figure.
    """

    fig = go.Figure(
        data=go.Heatmap(
            z=depth_map,
            colorscale="Viridis",
            colorbar=dict(title="Depth (m)", tickfont=dict(color="#00ff9f")),
        )
    )
    fig.update_layout(
        title="Depth Heatmap",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#00ff9f", family="Courier New"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_surface_figure(
    depth_map: np.ndarray,
    downsample: bool = True,
    target_size: int = 720,
) -> go.Figure:
    """Render an interactive 3D terrain surface for mission evaluation.

    Physics note:
    Surface plots transform local depth estimates into a spatially intuitive
    relief model. Operators can rotate and zoom to inspect crater morphology and
    identify safer landing basins with lower local roughness.

    Performance note:
    When downsample=True, the depth map is reduced to target_size for smoother
    WebGL rendering while allowing larger meshes than the legacy 200-size path.

    Args:
        depth_map: 2D depth map in meters.
        downsample: Whether to downsample for performance.
        target_size: Target dimension for downsampling.

    Returns:
        Plotly 3D surface figure.
    """

    render_map = _downsample_for_3d(depth_map, target_size) if downsample else depth_map

    # NASA-inspired colorscale: deep space blue → teal → amber → white
    nasa_colorscale = [
        [0.0, "#0a0e1a"],
        [0.15, "#0d2847"],
        [0.3, "#0e4d64"],
        [0.45, "#0f7b6f"],
        [0.6, "#1db87e"],
        [0.75, "#ffb300"],
        [0.9, "#ff6f20"],
        [1.0, "#ffffff"],
    ]

    fig = go.Figure(
        data=[
            go.Surface(
                z=-render_map,
                colorscale=nasa_colorscale,
                showscale=True,
                colorbar=dict(
                    title="Elevation (m)",
                    tickfont=dict(color="#00ff9f"),
                    len=0.75,
                ),
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.6,
                    specular=0.3,
                    roughness=0.5,
                    fresnel=0.2,
                ),
                lightposition=dict(x=100, y=200, z=300),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="#00ff9f",
                        project_z=True,
                    )
                ),
            )
        ]
    )

    fig.update_layout(
        title="TERRAIN ELEVATION MODEL — MARE SURFACE SCAN",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Elevation",
            bgcolor="#0a0e1a",
            xaxis=dict(color="#00ff9f", showbackground=False),
            yaxis=dict(color="#00ff9f", showbackground=False),
            zaxis=dict(color="#00ff9f", showbackground=False),
            camera=dict(eye=dict(x=1.8, y=1.6, z=0.85)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.4),
        ),
        paper_bgcolor="#0a0e1a",
        font=dict(color="#00ff9f", family="Courier New"),
        margin=dict(l=0, r=0, t=44, b=0),
    )

    return fig


def make_contour_figure(depth_map: np.ndarray) -> go.Figure:
    """Generate a top-down contour representation of reconstructed terrain.

    Args:
        depth_map: 2D depth map in meters.

    Returns:
        Plotly contour figure.
    """

    fig = go.Figure(
        data=go.Contour(
            z=depth_map,
            colorscale="Viridis",
            contours=dict(coloring="heatmap", showlabels=True),
            colorbar=dict(title="Depth (m)", tickfont=dict(color="#00ff9f")),
        )
    )
    fig.update_layout(
        title="Top-Down Contour Map",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#00ff9f", family="Courier New"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def normalized_hazard_cost(depth_map: np.ndarray) -> np.ndarray:
    """Convert depth map to normalized traversal cost field.

    Path-planning concept:
    Deep depressions are treated as riskier terrain. Normalized costs let A*
    balance geometric distance against geophysical hazard.

    Args:
        depth_map: Depth map in meters.

    Returns:
        Float32 cost grid in [1, 10].
    """

    if float(np.max(depth_map)) <= 1e-6:
        return np.ones_like(depth_map, dtype=np.float32)

    z = depth_map / (float(np.max(depth_map)) + 1e-9)
    return (1.0 + 9.0 * z).astype(np.float32)
