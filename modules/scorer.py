"""Landing safety scoring logic based on crater geometry and density."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go


def _local_density(
    rows: list[dict[str, Any]],
    index: int,
    radius_px: float,
) -> int:
    """Count neighboring craters around a target crater center.

    Risk model note:
    High local crater density increases touchdown uncertainty and limits
    maneuvering margin for final descent.

    Args:
        rows: Crater records with center coordinates.
        index: Index of the crater under evaluation.
        radius_px: Neighborhood radius in pixels.

    Returns:
        Number of neighboring craters within radius.
    """

    cx = rows[index]["center_x"]
    cy = rows[index]["center_y"]
    count = 0
    for j, row in enumerate(rows):
        if j == index:
            continue
        dx = row["center_x"] - cx
        dy = row["center_y"] - cy
        if (dx * dx + dy * dy) <= radius_px * radius_px:
            count += 1
    return count


def classify_zone(score: float) -> str:
    """Map numeric safety score to mission zone class.

    Args:
        score: Landing safety score in [0, 100].

    Returns:
        Zone label: SAFE, CAUTION, or HAZARD.
    """

    if score >= 70:
        return "SAFE"
    if score >= 40:
        return "CAUTION"
    return "HAZARD"


def score_landing_safety(
    depth_rows: list[dict[str, Any]],
    depth_threshold_m: float,
    landing_gear_span_m: float,
    density_radius_px: int,
    pixel_scale_m: float = 1.0,
) -> dict[str, Any]:
    """Compute crater-level landing safety scores and classifications.

    Risk model:
    - Depth penalty: larger depth relative to threshold increases risk smoothly.
    - Diameter penalty: crater width relative to gear span increases risk smoothly.
    - Density penalty: clusters of impacts increase obstacle congestion.

    Args:
        depth_rows: Per-crater depth table from photometric module.
        depth_threshold_m: Acceptable depth threshold Td.
        landing_gear_span_m: Effective landing gear footprint span.
        density_radius_px: Neighborhood radius for crater density estimation.
        pixel_scale_m: Meters represented per image pixel.

    Returns:
        Dictionary with scored rows, summary counts, and overall score.
    """

    scored: list[dict[str, Any]] = []

    for i, row in enumerate(depth_rows):
        depth_m = float(row["depth_m"])
        diameter_m = float(row["diameter_px"]) * pixel_scale_m
        neighbors = _local_density(depth_rows, i, radius_px=float(density_radius_px))

        score = 100.0

        # Continuous penalties preserve slider sensitivity even in rough scenes.
        depth_scale = max(0.2, float(depth_threshold_m) * 5.0)
        depth_penalty = 45.0 * (depth_m / (depth_m + depth_scale))
        score -= depth_penalty

        gear_scale = max(0.2, float(landing_gear_span_m) * 3.0)
        diameter_penalty = 30.0 * (diameter_m / (diameter_m + gear_scale))
        score -= diameter_penalty

        score -= min(25.0, neighbors * 4.0)
        score = float(np.clip(score, 0.0, 100.0))

        zone = classify_zone(score)
        scored.append(
            {
                **row,
                "diameter_m": round(diameter_m, 3),
                "neighbor_count": int(neighbors),
                "safety_score": round(score, 2),
                "zone": zone,
            }
        )

    safe = sum(1 for r in scored if r["zone"] == "SAFE")
    caution = sum(1 for r in scored if r["zone"] == "CAUTION")
    hazard = sum(1 for r in scored if r["zone"] == "HAZARD")
    overall = float(np.mean([r["safety_score"] for r in scored])) if scored else 0.0

    return {
        "rows": scored,
        "summary": {"safe": safe, "caution": caution, "hazard": hazard},
        "overall_score": round(overall, 2),
    }


def annotate_hazard_map(
    image: np.ndarray,
    scored_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Draw zone-colored crater boxes and labels on the scene.

    Args:
        image: Grayscale or RGB base image.
        scored_rows: Scored crater rows.

    Returns:
        RGB hazard map image.
    """

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()

    color_map = {
        "SAFE": (0, 255, 159),
        "CAUTION": (255, 179, 0),
        "HAZARD": (255, 60, 60),
    }

    for row in scored_rows:
        c = color_map[row["zone"]]
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
        text = f"{row['crater_id']}:{row['safety_score']:.0f}"
        cv2.putText(canvas, text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

    return canvas


def build_safety_gauge(overall_score: float) -> go.Figure:
    """Create horizontal-style safety gauge for mission summary.

    Args:
        overall_score: Global terrain safety score in [0, 100].

    Returns:
        Plotly indicator figure.
    """

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall_score,
            number={"suffix": " / 100", "font": {"color": "#00ff9f"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#00ff9f"},
                "bar": {"color": "#00ff9f"},
                "bgcolor": "#111827",
                "steps": [
                    {"range": [0, 40], "color": "#3b1010"},
                    {"range": [40, 70], "color": "#3a2a08"},
                    {"range": [70, 100], "color": "#103022"},
                ],
            },
            title={"text": "Overall Terrain Safety", "font": {"color": "#00ff9f"}},
        )
    )
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        margin=dict(l=18, r=18, t=45, b=10),
        font=dict(color="#00ff9f", family="Courier New"),
    )
    return fig


def build_score_map(
    image_shape: tuple[int, int],
    scored_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Rasterize crater scores into a per-pixel terrain score map.

    Path-planning concept:
    A dense score map allows A* to optimize routes over continuous risk, not
    just binary obstacle masks.

    Args:
        image_shape: (height, width) scene size.
        scored_rows: Crater score records.

    Returns:
        Float32 score map in [0, 100].
    """

    h, w = image_shape
    score_map = np.full((h, w), 82.0, dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    for row in scored_rows:
        cx = float(row["center_x"])
        cy = float(row["center_y"])
        radius = max(6.0, float(row["radius_px"]))
        crater_score = float(row["safety_score"])

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        influence = np.exp(-(dist**2) / (2.0 * (radius * 1.15) ** 2))

        local = score_map * (1.0 - influence) + crater_score * influence
        score_map = np.minimum(score_map, local.astype(np.float32))

    return np.clip(score_map, 0.0, 100.0).astype(np.float32)
