"""Photometric depth estimation for crater morphology."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def crater_circle_mask(shape: tuple[int, int], center: tuple[float, float], radius_px: float) -> np.ndarray:
    """Boolean mask of the crater interior inside an ROI.

    Args:
        shape: ROI shape as (height, width).
        center: Crater centre in ROI coordinates as (x, y).
        radius_px: Crater radius in pixels.

    Returns:
        Boolean array, True inside the crater circle.
    """

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = max(1.0, float(radius_px))
    return ((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2) <= r * r


def compute_otsu_shadow_mask(
    roi: np.ndarray,
    center: tuple[float, float] | None = None,
    radius_px: float | None = None,
) -> np.ndarray:
    """Segment shadow pixels in a crater ROI using Otsu thresholding.

    Physics note:
    Under directional sunlight, crater interiors cast dark shadows whose extent
    scales with local relief. Otsu's method automatically chooses a threshold
    that separates darker shadow regions from brighter terrain in bimodal ROIs.

    The shadow is a subset of the crater interior, so when the crater geometry
    is supplied the threshold is computed from interior pixels only and the
    mask is clipped to the crater circle. This keeps bright surrounding terrain
    out of the measurement.

    Args:
        roi: Grayscale crater crop.
        center: Optional crater centre in ROI coordinates as (x, y).
        radius_px: Optional crater radius in pixels.

    Returns:
        Binary mask where shadow pixels are 255.
    """

    blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0, sigmaY=1.0)

    if center is not None and radius_px is not None:
        circle = crater_circle_mask(roi.shape, center, radius_px)
        interior = blur[circle]
        if interior.size < 3:
            return np.zeros_like(roi, dtype=np.uint8)
        # cv2.threshold returns (threshold_value, thresholded_image); the VALUE
        # is what a pixel must be compared against.
        threshold_value, _ = cv2.threshold(
            interior.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        shadow = np.where((blur < threshold_value) & circle, 255, 0).astype(np.uint8)
    else:
        threshold_value, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        shadow = np.where(blur < threshold_value, 255, 0).astype(np.uint8)
        circle = None

    kernel = np.ones((3, 3), np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, kernel)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)
    if circle is not None:
        shadow[~circle] = 0
    return shadow


def measure_shadow_length(mask: np.ndarray, solar_azimuth_deg: float) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Estimate shadow length along opposite solar direction.

    Physics note:
    Shadow-based depth inference depends on geometric projection. For a known
    sun direction, the shadow extent is approximated by projecting shadow pixels
    onto the anti-sun axis and measuring the projection range.

    Args:
        mask: Binary shadow mask.
        solar_azimuth_deg: Solar azimuth angle in image plane degrees.

    Returns:
        Tuple of (length_px, point_start, point_end) in ROI coordinates.
    """

    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return 0.0, (0, 0), (0, 0)

    anti = math.radians((solar_azimuth_deg + 180.0) % 360.0)
    v = np.array([math.cos(anti), math.sin(anti)], dtype=np.float64)

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    proj = pts @ v

    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))

    p0 = (int(pts[i_min, 0]), int(pts[i_min, 1]))
    p1 = (int(pts[i_max, 0]), int(pts[i_max, 1]))
    length = float(max(0.0, proj[i_max] - proj[i_min]))

    return length, p0, p1


MAX_SHADOW_CIRCLE_FRACTION = 0.60
MIN_SHADOW_PIXELS = 3

# Half-width of the analysis window, in crater radii, measured from the crater
# centre. Larger than 1.0 so the window extends past the rim: a shadow that fills
# the bowl must not be mistaken for one truncated by the edge of the crop.
ROI_WINDOW_RADIUS_FACTOR = 1.4


def measure_crater_shadow(
    roi: np.ndarray,
    center: tuple[float, float],
    radius_px: float,
    solar_azimuth_deg: float,
) -> dict[str, Any]:
    """Measure one crater's shadow, or report why it is not measurable.

    Validity model:
    A shadow measurement is only meaningful when a compact dark region sits on
    the shadowed side of the crater and is fully inside the ROI. Each failure
    below makes the projected length something other than a shadow length, so
    the function returns ``length_px = None`` with a reason instead of a number.

    Shadow-side convention: inside a crater the cast shadow lies against the
    UP-SUN rim, i.e. on the half of the interior towards the sun, and extends
    down-sun across the floor. (Terrain outside the rim is shadowed on the
    anti-solar side, but that shadow is set by rim height, not crater depth.)
    Measured on ray-cast scenes (scripts/shadow_sanity.py, 29 craters over
    seeds 42/7/123): the mean shadow position is +0.53 crater radii along the
    direction towards the sun, and 0 of 29 craters have their interior shadow
    on the anti-solar side. ``solar_azimuth_deg`` is the direction TOWARDS the
    sun in image coordinates.

    Args:
        roi: Grayscale crater crop.
        center: Crater centre in ROI coordinates as (x, y).
        radius_px: Crater radius in pixels.
        solar_azimuth_deg: Direction towards the sun, degrees in the image plane.

    Returns:
        Dict with length_px (float or None), mask, p0, p1, mask_circle_fraction
        and reason (None when measurable).
    """

    circle = crater_circle_mask(roi.shape, center, radius_px)
    circle_px = int(np.count_nonzero(circle))
    mask = compute_otsu_shadow_mask(roi, center=center, radius_px=radius_px)
    empty = np.zeros_like(mask)

    def fail(reason: str, fraction: float, keep: np.ndarray | None = None) -> dict[str, Any]:
        return {
            "length_px": None,
            "mask": keep if keep is not None else mask,
            "p0": (0, 0),
            "p1": (0, 0),
            "mask_circle_fraction": fraction,
            "reason": reason,
        }

    if circle_px <= 0:
        return fail("crater circle falls outside the ROI", 0.0, empty)

    fraction = float(np.count_nonzero(mask)) / float(circle_px)
    if fraction > MAX_SHADOW_CIRCLE_FRACTION:
        return fail(
            f"shadow mask covers {fraction * 100:.1f}% of the crater area "
            f"(> {MAX_SHADOW_CIRCLE_FRACTION * 100:.0f}%): ROI histogram was not bimodal",
            fraction,
        )

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sun = math.radians(solar_azimuth_deg % 360.0)
    sun_vec = (math.cos(sun), math.sin(sun))

    best_label, best_area = 0, 0
    for label in range(1, n_labels):
        cx, cy = centroids[label]
        side = (cx - center[0]) * sun_vec[0] + (cy - center[1]) * sun_vec[1]
        if side <= 0.0:
            continue  # component sits on the down-sun (lit) half of the floor
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_label, best_area = label, area

    if best_label == 0:
        return fail("no shadow component against the up-sun rim of the crater", fraction, empty)

    keep = np.where(labels == best_label, 255, 0).astype(np.uint8)
    if best_area < MIN_SHADOW_PIXELS:
        return fail(f"only {best_area} shadow pixels survived cleanup (< {MIN_SHADOW_PIXELS})", fraction, keep)

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    if x <= 0 or y <= 0 or x + w >= roi.shape[1] or y + h >= roi.shape[0]:
        return fail(
            "shadow component touches the ROI boundary, so its length is a lower bound, not a measurement",
            fraction,
            keep,
        )

    length, p0, p1 = measure_shadow_length(keep, solar_azimuth_deg=solar_azimuth_deg)
    if length <= 0.0:
        return fail("shadow projection range is zero", fraction, keep)

    return {
        "length_px": float(length),
        "mask": keep,
        "p0": p0,
        "p1": p1,
        "mask_circle_fraction": fraction,
        "reason": None,
    }


def depth_from_shadow(
    shadow_length_px: float,
    solar_elevation_angle_deg: float,
    pixel_scale_m: float,
) -> float:
    """Convert shadow length into crater depth using photometric geometry.

    Physics note:
    A simplified relation for local relief is
    depth ~= shadow_length * tan(theta), where theta is the solar ELEVATION
    angle measured up from the local horizontal (not incidence from the surface
    normal; for incidence i the relation would be depth = L / tan(i)).
    Pixel length is converted to meters through image scale metadata.

    Args:
        shadow_length_px: Measured shadow length in pixels.
        solar_elevation_angle_deg: Solar elevation above the horizontal, degrees.
        pixel_scale_m: Meters represented by one pixel.

    Returns:
        Estimated depth in meters.
    """

    theta = math.radians(np.clip(solar_elevation_angle_deg, 1.0, 89.0))
    return float(shadow_length_px * pixel_scale_m * math.tan(theta))


def estimate_crater_depths(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    solar_elevation_angle_deg: float,
    solar_azimuth_deg: float = 35.0,
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
        solar_elevation_angle_deg: User-specified solar elevation above the horizontal.
        solar_azimuth_deg: User-specified solar azimuth for shadow direction.
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

        # Analysis window: a square of +/- ROI_WINDOW_RADIUS_FACTOR * radius about
        # the crater centre, clipped to the image. The detection box itself is too
        # tight - it IS the crater - so a shadow spanning the bowl always reached
        # its edge and tripped the truncation guard. The crater circle used for
        # masking keeps the detection radius; only the observable window grows, so
        # "component touches the ROI boundary" once again means the shadow left the
        # window rather than that the box was drawn tight.
        cx_full = float(det["center_x"])
        cy_full = float(det["center_y"])
        half = max(3.0, ROI_WINDOW_RADIUS_FACTOR * float(det["radius_px"]))
        rx1 = int(max(0, math.floor(cx_full - half)))
        ry1 = int(max(0, math.floor(cy_full - half)))
        rx2 = int(min(w, math.ceil(cx_full + half) + 1))
        ry2 = int(min(h, math.ceil(cy_full + half) + 1))

        roi = image[ry1:ry2, rx1:rx2]
        if roi.shape[0] < 3 or roi.shape[1] < 3:
            continue

        measurement = measure_crater_shadow(
            roi,
            center=(cx_full - rx1, cy_full - ry1),
            radius_px=float(det["radius_px"]),
            solar_azimuth_deg=solar_azimuth_deg,
        )
        mask = measurement["mask"]
        p0, p1 = measurement["p0"], measurement["p1"]
        shadow_len = measurement["length_px"]

        if shadow_len is None:
            depth_m = None
            slope_deg = None
        else:
            depth_m = depth_from_shadow(
                shadow_length_px=shadow_len,
                solar_elevation_angle_deg=solar_elevation_angle_deg,
                pixel_scale_m=pixel_scale_m,
            )
            radius_m = max(0.1, 0.5 * det["diameter_px"] * pixel_scale_m)
            slope_deg = float(np.degrees(np.arctan2(depth_m, radius_m)))

        annotated = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        if shadow_len is None:
            label = "NOT MEASURABLE"
        else:
            cv2.arrowedLine(annotated, p0, p1, (255, 179, 0), 2, tipLength=0.22)
            label = f"L={shadow_len:.1f}px  AZ={solar_azimuth_deg:.0f}deg"
        cv2.putText(
            annotated,
            label,
            (6, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 159) if shadow_len is not None else (255, 80, 80),
            1,
            cv2.LINE_AA,
        )

        row = {
            "crater_id": det["crater_id"],
            "shadow_length_px": None if shadow_len is None else round(float(shadow_len), 3),
            "solar_angle_deg": round(float(solar_elevation_angle_deg), 3),
            "solar_elevation_deg": round(float(solar_elevation_angle_deg), 3),
            "solar_azimuth_deg": round(float(solar_azimuth_deg), 3),
            "depth_m": None if depth_m is None else round(float(depth_m), 3),
            "slope_estimate_deg": None if slope_deg is None else round(float(slope_deg), 3),
            "measurable": shadow_len is not None,
            "not_measurable_reason": measurement["reason"],
            "shadow_mask_circle_fraction": round(float(measurement["mask_circle_fraction"]), 4),
            "confidence": None if det.get("confidence") is None else float(det["confidence"]),
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

    n_not_measurable = sum(1 for r in rows if not r["measurable"])
    return {
        "rows": rows,
        "rois": rois,
        "formula": "depth_m = shadow_length_px * pixel_scale_m * tan(theta)",
        "n_craters": len(rows),
        "n_not_measurable": n_not_measurable,
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
        Dictionary containing fused rows and the mean absolute depth difference
        between the two views for matched craters.
    """

    fused_rows: list[dict[str, Any]] = []
    used_secondary: set[int] = set()

    for p in primary_rows:
        if p.get("depth_m") is None:
            continue  # not measurable in the primary view: nothing to fuse
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
            if s.get("depth_m") is None:
                continue  # not measurable in the secondary view
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
        return {"rows": [], "mean_abs_view_difference_m": None}

    # No per-estimate uncertainty exists, so no "uncertainty reduction" is
    # reported. The measurable quantity is how much the two views disagree.
    diffs = [abs(r["single_angle_depth_m"] - r["secondary_depth_m"]) for r in fused_rows]

    return {
        "rows": fused_rows,
        "mean_abs_view_difference_m": round(float(np.mean(diffs)), 3),
    }
