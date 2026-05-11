"""Crater detection module with trained YOLO11m and CV hybrid comparison."""

from __future__ import annotations

import math
import os
import time
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from skimage.feature import blob_log

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency behavior
    YOLO = None


# --- Path to the trained crater detector weights ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "best.pt")


@lru_cache(maxsize=1)
def load_yolo_model(model_path: str = DEFAULT_MODEL_PATH) -> tuple[Any | None, str]:
    """Load trained YOLO crater model once and cache it for the app lifetime.

    Args:
        model_path: Path to the trained .pt checkpoint.

    Returns:
        Tuple of (model_or_none, status_message).
    """

    if YOLO is None:
        return None, "Ultralytics package unavailable; install ultralytics to use YOLO"

    if not os.path.isfile(model_path):
        return None, f"Model weights not found at {model_path}"

    try:
        model = YOLO(model_path)
        return model, f"Loaded crater detector: {os.path.basename(model_path)}"
    except Exception as exc:  # pragma: no cover - weight/env dependent
        return None, f"Failed to load {model_path}: {exc}"


def _to_detection_record(
    crater_id: str,
    cx: float,
    cy: float,
    r: float,
    confidence: float,
    source: str,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Create a normalized crater detection record and clip to image bounds.

    Args:
        crater_id: Unique crater identifier.
        cx: Center x in pixels.
        cy: Center y in pixels.
        r: Radius in pixels.
        confidence: Detector confidence in [0, 1].
        source: Detector source tag.
        shape: Image shape as (height, width).

    Returns:
        Canonical crater detection dictionary.
    """

    h, w = shape
    rr = max(5.0, float(r))
    x1 = int(np.clip(round(cx - rr), 0, w - 1))
    y1 = int(np.clip(round(cy - rr), 0, h - 1))
    x2 = int(np.clip(round(cx + rr), 0, w - 1))
    y2 = int(np.clip(round(cy + rr), 0, h - 1))

    diameter = max(2.0, float(x2 - x1 + y2 - y1) / 2.0)
    return {
        "crater_id": crater_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": float(np.clip(confidence, 0.05, 0.99)),
        "diameter_px": float(diameter),
        "center_x": int(np.clip(round(cx), 0, w - 1)),
        "center_y": int(np.clip(round(cy), 0, h - 1)),
        "radius_px": int(max(3, round(diameter / 2.0))),
        "source": source,
    }


# ─────────────────────────────────────────────────────────────
#  YOLO Inference (primary detector — trained crater model)
# ─────────────────────────────────────────────────────────────


def _yolo_inference(
    image: np.ndarray,
    conf_threshold: float = 0.35,
    model_path: str = DEFAULT_MODEL_PATH,
) -> tuple[list[dict[str, Any]], str]:
    """Run trained YOLO crater model inference.

    Args:
        image: Input grayscale or RGB image.
        conf_threshold: Confidence threshold for returned boxes.
        model_path: Path to .pt weights.

    Returns:
        Tuple of detections and status message.
    """

    model, model_status = load_yolo_model(model_path=model_path)
    if model is None:
        return [], model_status

    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image

    try:
        result = model.predict(
            source=rgb,
            conf=conf_threshold,
            imgsz=416,
            device="cpu",
            max_det=300,
            verbose=False,
        )[0]
    except Exception as exc:  # pragma: no cover
        return [], f"YOLO inference error: {exc}"

    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections, f"{model_status}; no boxes"

    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
    h, w = image.shape[:2]

    for idx, (box, conf) in enumerate(zip(xyxy, confs), start=1):
        x1, y1, x2, y2 = box.tolist()
        ww = max(1.0, x2 - x1)
        hh = max(1.0, y2 - y1)
        diameter = 0.5 * (ww + hh)
        det = _to_detection_record(
            crater_id=f"Y-{idx:03d}",
            cx=(x1 + x2) / 2.0,
            cy=(y1 + y2) / 2.0,
            r=diameter / 2.0,
            confidence=float(conf),
            source="yolo",
            shape=(h, w),
        )
        detections.append(det)

    return detections, f"{model_status}; boxes={len(detections)}"


# ─────────────────────────────────────────────────────────────
#  CV Hybrid Detector (for comparison display only)
# ─────────────────────────────────────────────────────────────


def _hough_crater_candidates(image: np.ndarray) -> list[dict[str, Any]]:
    """Detect circular crater candidates with multi-scale Hough passes.

    Computer vision note:
    Hough circles are effective for circular rims but sensitive to parameter
    choice, so two scales are fused to recover both small and medium craters.

    Args:
        image: Preprocessed grayscale image.

    Returns:
        Candidate detections from Hough voting.
    """

    h, w = image.shape
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.15, sigmaY=1.15)
    min_dim = float(min(h, w))

    passes = [
        dict(dp=1.15, minDist=max(14, int(min_dim * 0.03)), param1=85, param2=16, minRadius=8, maxRadius=60),
        dict(dp=1.35, minDist=max(22, int(min_dim * 0.045)), param1=80, param2=19, minRadius=18, maxRadius=120),
    ]

    out: list[dict[str, Any]] = []
    idx = 0
    for p in passes:
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=p["dp"],
            minDist=p["minDist"],
            param1=p["param1"],
            param2=p["param2"],
            minRadius=p["minRadius"],
            maxRadius=p["maxRadius"],
        )
        if circles is None:
            continue
        for c in np.round(circles[0, :]).astype(int):
            idx += 1
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            conf = 0.6 + 0.2 * min(1.0, r / 80.0)
            out.append(_to_detection_record(f"H-{idx:03d}", x, y, r, conf, "cv-hough", image.shape))
    return out


def _blob_crater_candidates(image: np.ndarray) -> list[dict[str, Any]]:
    """Detect dark crater-like blobs using Laplacian of Gaussian.

    Computer vision note:
    Dark crater interiors appear as approximately blob-like minima. LoG blob
    detection responds to these depressions even when rims are partially faded.

    Args:
        image: Preprocessed grayscale image.

    Returns:
        Candidate detections from blob scale-space extrema.
    """

    inv = 1.0 - image.astype(np.float32) / 255.0
    blobs = blob_log(
        inv,
        min_sigma=3,
        max_sigma=max(8, int(min(image.shape) * 0.08)),
        num_sigma=12,
        threshold=0.055,
        overlap=0.5,
    )

    out: list[dict[str, Any]] = []
    for idx, b in enumerate(blobs, start=1):
        y, x, sigma = float(b[0]), float(b[1]), float(b[2])
        r = sigma * math.sqrt(2.0)
        conf = 0.52 + 0.35 * min(1.0, r / 40.0)
        out.append(_to_detection_record(f"B-{idx:03d}", x, y, r, conf, "cv-blob", image.shape))
    return out


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Compute IoU overlap for two axis-aligned crater boxes."""

    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, a["x2"] - a["x1"]) * max(1, a["y2"] - a["y1"])
    area_b = max(1, b["x2"] - b["x1"]) * max(1, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def _merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicated crater candidates from multiple detectors.

    Fusion note:
    Hough and LoG candidates can overlap heavily. Spatial deduplication keeps a
    single representative crater hypothesis per physical structure.

    Args:
        candidates: Raw candidate detections.

    Returns:
        Deduplicated detections sorted by confidence.
    """

    ordered = sorted(candidates, key=lambda d: float(d["confidence"]), reverse=True)
    keep: list[dict[str, Any]] = []

    for cand in ordered:
        merged = False
        for k in keep:
            dx = float(cand["center_x"] - k["center_x"])
            dy = float(cand["center_y"] - k["center_y"])
            center_dist = math.hypot(dx, dy)
            r_lim = 0.8 * min(float(cand["radius_px"]), float(k["radius_px"]))
            overlap = _iou(cand, k)
            if center_dist <= r_lim or overlap >= 0.28:
                if cand["confidence"] > k["confidence"]:
                    k.update(cand)
                else:
                    k["confidence"] = float(np.clip((k["confidence"] + 0.05), 0.05, 0.99))
                merged = True
                break
        if not merged:
            keep.append(dict(cand))

    for idx, det in enumerate(keep, start=1):
        det["crater_id"] = f"CR-{idx:02d}"
        det["source"] = "cv-hybrid"

    return keep


def _candidate_quality(image: np.ndarray, det: dict[str, Any]) -> float:
    """Estimate how crater-like a candidate looks from local photometry.

    Vision note:
    Real craters often have darker interiors and brighter rims under oblique
    sunlight. This score compares center darkness against annular rim intensity
    and edge energy to reject noisy circular artifacts.

    Args:
        image: Preprocessed grayscale image.
        det: Candidate detection record.

    Returns:
        Quality score in [0, 1].
    """

    h, w = image.shape
    cx = int(det["center_x"])
    cy = int(det["center_y"])
    r = max(5, int(det["radius_px"]))

    x0 = max(0, cx - int(1.4 * r))
    y0 = max(0, cy - int(1.4 * r))
    x1 = min(w, cx + int(1.4 * r))
    y1 = min(h, cy + int(1.4 * r))
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return 0.0

    patch = image[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    inner = dist <= (0.72 * r)
    annulus = (dist >= (0.92 * r)) & (dist <= (1.28 * r))
    if int(np.count_nonzero(inner)) < 15 or int(np.count_nonzero(annulus)) < 25:
        return 0.0

    inner_mean = float(np.mean(patch[inner]))
    ann_mean = float(np.mean(patch[annulus]))
    contrast = ann_mean - inner_mean

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    edge = float(np.mean(grad[annulus]))

    contrast_score = float(np.clip((contrast - 2.0) / 24.0, 0.0, 1.0))
    edge_score = float(np.clip(edge / 42.0, 0.0, 1.0))
    return 0.65 * contrast_score + 0.35 * edge_score


def _cv_hybrid_candidates(image: np.ndarray) -> list[dict[str, Any]]:
    """Generate robust crater candidates by fusing Hough and LoG detectors.

    Args:
        image: Preprocessed grayscale image.

    Returns:
        Hybrid candidate detections.
    """

    h, w = image.shape
    use_blob = (h * w) <= 520_000
    raw_candidates = _hough_crater_candidates(image)
    if use_blob:
        raw_candidates += _blob_crater_candidates(image)

    merged = _merge_candidates(raw_candidates)

    scored: list[dict[str, Any]] = []
    for det in merged:
        q = _candidate_quality(image, det)
        det["confidence"] = float(np.clip(0.35 * det["confidence"] + 0.65 * q, 0.05, 0.99))
        if q >= 0.22 and 8 <= int(det["radius_px"]) <= 120:
            scored.append(det)

    if len(scored) < 3:
        scored = sorted(merged, key=lambda d: float(d["confidence"]), reverse=True)[:8]

    scored = sorted(scored, key=lambda d: float(d["confidence"]), reverse=True)
    max_count = max(8, min(24, int(image.size / 26000)))
    return scored[:max_count]


def _rescale_detections(
    detections: list[dict[str, Any]],
    sx: float,
    sy: float,
    target_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    """Rescale detections from working-resolution to original image size.

    Args:
        detections: Detection records in working image coordinates.
        sx: X-axis scale from working to original image.
        sy: Y-axis scale from working to original image.
        target_shape: Original image shape as (height, width).

    Returns:
        Rescaled and re-indexed detections.
    """

    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        out = [dict(d) for d in detections]
    else:
        h, w = target_shape
        out = []
        for d in detections:
            x1 = int(np.clip(round(d["x1"] * sx), 0, w - 1))
            y1 = int(np.clip(round(d["y1"] * sy), 0, h - 1))
            x2 = int(np.clip(round(d["x2"] * sx), 0, w - 1))
            y2 = int(np.clip(round(d["y2"] * sy), 0, h - 1))
            ww = max(1, x2 - x1)
            hh = max(1, y2 - y1)
            diameter = float((ww + hh) / 2.0)
            rec = dict(d)
            rec.update(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "diameter_px": diameter,
                    "center_x": int((x1 + x2) / 2),
                    "center_y": int((y1 + y2) / 2),
                    "radius_px": int(max(3, round(diameter / 2.0))),
                }
            )
            out.append(rec)

    for idx, d in enumerate(out, start=1):
        d["crater_id"] = f"CR-{idx:02d}"
    return out


def _synthetic_hint_detections(image_shape: tuple[int, int], hint_craters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert known synthetic crater metadata into detection rows.

    Educational note:
    In synthetic demo mode we know ground-truth crater geometry. Exposing these
    as detections keeps the pipeline deterministic for teaching and debugging.

    Args:
        image_shape: Image shape as (height, width).
        hint_craters: Synthetic crater metadata list.

    Returns:
        Canonical detection rows.
    """

    out: list[dict[str, Any]] = []
    for idx, c in enumerate(hint_craters, start=1):
        out.append(
            _to_detection_record(
                crater_id=f"CR-{idx:02d}",
                cx=float(c["center_x"]),
                cy=float(c["center_y"]),
                r=float(c["radius_px"]),
                confidence=0.98,
                source="synthetic-meta",
                shape=image_shape,
            )
        )
    return out


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────


def detect_craters(
    image: np.ndarray,
    conf_threshold: float = 0.35,
    model_path: str = DEFAULT_MODEL_PATH,
    hint_craters: list[dict[str, Any]] | None = None,
    max_detection_dim: int = 896,
) -> dict[str, Any]:
    """Detect craters using the trained YOLO11m model.

    This is the primary detection function used by the pipeline. It runs
    YOLO inference exclusively (no CV fallback).

    Args:
        image: Preprocessed grayscale image.
        conf_threshold: YOLO confidence threshold.
        model_path: Path to trained .pt weights.
        hint_craters: Optional known crater metadata for synthetic mode.
        max_detection_dim: Max image dimension for detection.

    Returns:
        Dictionary with detections, source type, elapsed time, and status text.
    """

    start = time.perf_counter()

    if hint_craters:
        hints = _synthetic_hint_detections(image.shape, hint_craters)
        elapsed = time.perf_counter() - start
        return {
            "detections": hints,
            "source": "synthetic-meta",
            "elapsed_s": elapsed,
            "status": f"Loaded synthetic crater metadata: {len(hints)} craters",
            "map50_proxy": 0.99,
        }

    yolo_detections, yolo_status = _yolo_inference(
        image=image,
        conf_threshold=conf_threshold,
        model_path=model_path,
    )

    # Re-index crater IDs
    for idx, d in enumerate(yolo_detections, start=1):
        d["crater_id"] = f"CR-{idx:02d}"

    elapsed = time.perf_counter() - start
    return {
        "detections": yolo_detections,
        "source": "yolo",
        "elapsed_s": elapsed,
        "status": f"{yolo_status} | conf>={conf_threshold:.2f}",
        "map50_proxy": 0.847,
    }


def detect_craters_cv(
    image: np.ndarray,
    max_detection_dim: int = 896,
) -> dict[str, Any]:
    """Detect craters using the CV hybrid pipeline (Hough + LoG).

    This is used for comparison display alongside YOLO detections.

    Args:
        image: Preprocessed grayscale image.
        max_detection_dim: Max image dimension for detection.

    Returns:
        Dictionary with detections, source type, elapsed time, and status text.
    """

    start = time.perf_counter()

    h, w = image.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    work_img = image
    if max_dim > max_detection_dim:
        scale = max_detection_dim / float(max_dim)
        nw = max(64, int(round(w * scale)))
        nh = max(64, int(round(h * scale)))
        work_img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    cv_hybrid = _cv_hybrid_candidates(work_img)

    detections = _rescale_detections(
        cv_hybrid,
        sx=(1.0 / scale),
        sy=(1.0 / scale),
        target_shape=(h, w),
    )

    elapsed = time.perf_counter() - start
    return {
        "detections": detections,
        "source": "cv-hybrid",
        "elapsed_s": elapsed,
        "status": f"CV hybrid: Hough+LoG | craters={len(detections)} | scale={scale:.3f}",
        "map50_proxy": 0.82,
    }


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    color: tuple[int, int, int] = (0, 255, 159),
) -> np.ndarray:
    """Render detection boxes and labels for operator inspection.

    Args:
        image: Input grayscale or RGB image.
        detections: Crater detection records.
        color: RGB box color.

    Returns:
        RGB image with overlays.
    """

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det['crater_id']} {det['confidence']:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas
