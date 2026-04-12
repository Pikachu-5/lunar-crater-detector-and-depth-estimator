"""Photometric depth estimation for crater morphology."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def compute_otsu_shadow_mask(roi: np.ndarray) -> np.ndarray:
    """Segment shadow pixels in a crater ROI using Otsu thresholding.

    Physics note:
    Under directional sunlight, crater interiors cast dark shadows whose extent
    scales with local relief. Otsu's method automatically chooses a threshold
    that separates darker shadow regions from brighter terrain in bimodal ROIs.

    Args:
        roi: Grayscale crater crop.

    Returns:
        Binary mask where shadow pixels are 255.
    """

    blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0, sigmaY=1.0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shadow = np.where(blur < th, 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, kernel)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)
    return shadow


def measure_shadow_length(mask: np.ndarray, solar_angle_deg: float) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Estimate shadow length along opposite solar direction.

    Physics note:
    Shadow-based depth inference depends on geometric projection. For a known
    sun direction, the shadow extent is approximated by projecting shadow pixels
    onto the anti-sun axis and measuring the projection range.

    Args:
        mask: Binary shadow mask.
        solar_angle_deg: Solar azimuth angle in image plane degrees.

    Returns:
        Tuple of (length_px, point_start, point_end) in ROI coordinates.
    """

    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return 0.0, (0, 0), (0, 0)

    anti = math.radians((solar_angle_deg + 180.0) % 360.0)
    v = np.array([math.cos(anti), math.sin(anti)], dtype=np.float64)

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    proj = pts @ v

    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))

    p0 = (int(pts[i_min, 0]), int(pts[i_min, 1]))
    p1 = (int(pts[i_max, 0]), int(pts[i_max, 1]))
    length = float(max(0.0, proj[i_max] - proj[i_min]))

    return length, p0, p1


def depth_from_shadow(
    shadow_length_px: float,
    solar_incidence_angle_deg: float,
    pixel_scale_m: float,
) -> float:
    """Convert shadow length into crater depth using photometric geometry.

    Physics note:
    A simplified relation for local relief is
    depth ~= shadow_length * tan(theta), where theta is solar incidence angle.
    Pixel length is converted to meters through image scale metadata.

    Args:
        shadow_length_px: Measured shadow length in pixels.
        solar_incidence_angle_deg: Solar incidence angle in degrees.
        pixel_scale_m: Meters represented by one pixel.

    Returns:
        Estimated depth in meters.
    """

    theta = math.radians(np.clip(solar_incidence_angle_deg, 1.0, 89.0))
    return float(shadow_length_px * pixel_scale_m * math.tan(theta))


def estimate_crater_depths(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    solar_incidence_angle_deg: float,
    pixel_scale_m: float = 1.0,
) -> dict[str, Any]:
    """Estimate depth and slope for each detected crater with ROI diagnostics.

    Computer vision note:
    ROI-level analysis keeps each crater's measurements transparent. Operators
    can inspect the original crop, shadow mask, and measurement overlay to audit
    each depth estimate and understand uncertainty sources.

    Args:
        image: Grayscale scene image.
        detections: Crater detections with bbox fields.
        solar_incidence_angle_deg: User-specified solar incidence angle.
        pixel_scale_m: Meters per pixel.

    Returns:
        Dictionary with per-crater depth table and ROI visual artifacts.
    """

    h, w = image.shape
    rows: list[dict[str, Any]] = []
    rois: list[dict[str, Any]] = []

    for det in detections:
        x1 = int(max(0, det["x1"]))
        y1 = int(max(0, det["y1"]))
        x2 = int(min(w - 1, det["x2"]))
        y2 = int(min(h - 1, det["y2"]))

        if x2 <= x1 + 2 or y2 <= y1 + 2:
            continue

        roi = image[y1:y2, x1:x2]
        mask = compute_otsu_shadow_mask(roi)
        shadow_len, p0, p1 = measure_shadow_length(mask, solar_angle_deg=solar_incidence_angle_deg)

        depth_m = depth_from_shadow(
            shadow_length_px=shadow_len,
            solar_incidence_angle_deg=solar_incidence_angle_deg,
            pixel_scale_m=pixel_scale_m,
        )

        radius_m = max(0.1, 0.5 * det["diameter_px"] * pixel_scale_m)
        slope_deg = float(np.degrees(np.arctan2(depth_m, radius_m)))

        annotated = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        cv2.arrowedLine(annotated, p0, p1, (255, 179, 0), 2, tipLength=0.22)
        cv2.putText(
            annotated,
            f"L={shadow_len:.1f}px",
            (6, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 159),
            1,
            cv2.LINE_AA,
        )

        row = {
            "crater_id": det["crater_id"],
            "shadow_length_px": round(float(shadow_len), 3),
            "solar_angle_deg": round(float(solar_incidence_angle_deg), 3),
            "depth_m": round(float(depth_m), 3),
            "slope_estimate_deg": round(float(slope_deg), 3),
            "confidence": float(det.get("confidence", 0.7)),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "diameter_px": float(det["diameter_px"]),
            "center_x": int(det["center_x"]),
            "center_y": int(det["center_y"]),
            "radius_px": int(det["radius_px"]),
        }
        rows.append(row)

        rois.append(
            {
                "crater_id": det["crater_id"],
                "original": roi,
                "mask": mask,
                "annotated": annotated,
            }
        )

    return {
        "rows": rows,
        "rois": rois,
        "formula": "depth_m = shadow_length_px * pixel_scale_m * tan(theta)",
    }


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Compute intersection-over-union for two crater bounding boxes.

    Computer vision note:
    IoU quantifies geometric overlap between detections from separate passes.
    Multi-angle fusion relies on consistent crater correspondence, and IoU is a
    standard criterion for matching observations of the same object.

    Args:
        a: First crater record with x1,y1,x2,y2.
        b: Second crater record with x1,y1,x2,y2.

    Returns:
        IoU value in [0, 1].
    """

    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def fuse_depth_estimates(
    primary_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
    iou_threshold: float = 0.25,
) -> dict[str, Any]:
    """Fuse two depth passes via IoU-based crater correspondence.

    Physics note:
    Multiple illumination conditions provide complementary shadow geometry.
    Weighted fusion of matched crater depths can reduce random estimation error
    and improve robustness against local segmentation noise.

    Args:
        primary_rows: Depth rows from first view.
        secondary_rows: Depth rows from second view.
        iou_threshold: Minimum IoU to accept a crater match.

    Returns:
        Dictionary containing fused rows and uncertainty reduction estimate.
    """

    fused_rows: list[dict[str, Any]] = []
    used_secondary: set[int] = set()

    for p in primary_rows:
        best_idx = -1
        best_iou = 0.0
        for idx, s in enumerate(secondary_rows):
            if idx in used_secondary:
                continue
            score = bbox_iou(p, s)
            if score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            s = secondary_rows[best_idx]
            used_secondary.add(best_idx)
            w1 = float(np.clip(p.get("confidence", 0.7), 0.05, 0.99))
            w2 = float(np.clip(s.get("confidence", 0.7), 0.05, 0.99))
            fused = (w1 * p["depth_m"] + w2 * s["depth_m"]) / (w1 + w2)

            fused_rows.append(
                {
                    "crater_id": p["crater_id"],
                    "single_angle_depth_m": p["depth_m"],
                    "secondary_depth_m": s["depth_m"],
                    "fused_depth_m": round(float(fused), 3),
                    "iou": round(best_iou, 3),
                }
            )

    if len(fused_rows) == 0:
        return {"rows": [], "uncertainty_reduction_pct": 0.0}

    single = np.array([r["single_angle_depth_m"] for r in fused_rows], dtype=np.float64)
    fused = np.array([r["fused_depth_m"] for r in fused_rows], dtype=np.float64)

    sigma_single = float(np.std(single) + 1e-9)
    sigma_fused = float(np.std(fused) + 1e-9)
    reduction = max(0.0, (sigma_single - sigma_fused) / sigma_single * 100.0)

    return {
        "rows": fused_rows,
        "uncertainty_reduction_pct": round(reduction, 2),
    }
