"""Compare shadow-segmentation candidates for modules/depth.py against synthetic ground truth.

Candidates (all use the ground-truth boxes so detection error is excluded):
    shipped   - modules.depth as shipped (Otsu inside the crater circle + validity guard)
    a0_full   - polarity-correct Otsu over the whole ROI, no circle restriction
    a_otsu    - Otsu on pixels inside the crater circle
    b_pXX     - XXth percentile of interior pixels
    c_kX      - interior mean - X * interior std
Candidates a/b/c share: circle restriction, 3x3 open/close, keep the largest
component against the up-sun rim (see scripts/shadow_sanity.py check 2), and
return "not measurable" when the mask covers more than 60% of the crater area.

Scenes:
    legacy   - shadow-free generator (no cast shadows; depth lives in brightness)
    raycast  - occlusion shadows ray-marched over the height field (item 3)

Run:
    .venv/Scripts/python scripts/depth_segmentation_experiment.py
"""

from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import modules.depth as depth_mod  # noqa: E402
from modules.depth import measure_crater_shadow, measure_shadow_length  # noqa: E402
from modules.detector import ground_truth_detections  # noqa: E402
from utils.synthetic import generate_synthetic_lunar_surface  # noqa: E402

SEEDS = [42, 7, 123]
AZ = 35.0
SCALE = 1.0
KERNEL = np.ones((3, 3), np.uint8)
MAX_FRACTION = 0.60

SCENES = {
    "legacy": {"kwargs": {}, "elevation": 35.0},
    "raycast": {"kwargs": {"cast_shadows": True, "sun_elevation_deg": 20.0, "sun_angle_deg": AZ}, "elevation": 20.0},
}
METHODS = ["shipped", "shipped_nofrac", "a0_full", "a_otsu", "b_p15", "b_p20", "b_p25", "b_p30", "c_k0.5", "c_k1.0"]


def segment(roi: np.ndarray, cx: float, cy: float, r: float, method: str) -> tuple[np.ndarray | None, float]:
    """Return (mask, mask fraction of circle); mask None means not measurable."""

    if method in ("shipped", "shipped_nofrac"):
        # "shipped_nofrac" is a DIAGNOSTIC only: it measures how much of the
        # shipped estimator's rejection rate comes from the >60% area guard.
        # The shipped guard itself is not changed.
        original = depth_mod.MAX_SHADOW_CIRCLE_FRACTION
        if method == "shipped_nofrac":
            depth_mod.MAX_SHADOW_CIRCLE_FRACTION = 1.01
        try:
            res = measure_crater_shadow(roi, (cx, cy), r, solar_azimuth_deg=AZ)
        finally:
            depth_mod.MAX_SHADOW_CIRCLE_FRACTION = original
        return (None if res["length_px"] is None else res["mask"]), res["mask_circle_fraction"]

    blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0, sigmaY=1.0)
    h, w = roi.shape
    yy, xx = np.mgrid[0:h, 0:w]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    circle_px = float(max(1, np.count_nonzero(circle)))

    if method == "a0_full":
        t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = np.where(blur < t, 255, 0).astype(np.uint8)
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)
        return m, float(np.count_nonzero(m)) / circle_px

    interior = blur[circle]
    if interior.size < 3:
        return None, 0.0
    if method == "a_otsu":
        t, _ = cv2.threshold(interior.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method.startswith("b_p"):
        t = float(np.percentile(interior, int(method[3:])))
    elif method.startswith("c_k"):
        t = float(interior.mean() - float(method[3:]) * interior.std())
    else:
        raise ValueError(method)

    m = np.where((blur < t) & circle, 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)
    m[~circle] = 0
    frac = float(np.count_nonzero(m)) / circle_px
    if frac > MAX_FRACTION:
        return None, frac

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    sun = np.array([math.cos(math.radians(AZ)), math.sin(math.radians(AZ))])
    best, best_area = 0, 0
    for i in range(1, n):
        side = (cents[i, 0] - cx) * sun[0] + (cents[i, 1] - cy) * sun[1]
        if side <= 0:  # keep components against the up-sun rim
            continue
        if stats[i, cv2.CC_STAT_AREA] > best_area:
            best, best_area = i, stats[i, cv2.CC_STAT_AREA]
    if best == 0 or best_area < 3:
        return None, frac
    return np.where(labels == best, 255, 0).astype(np.uint8), frac


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial_corr(est, depth, radius) -> float:
    """Correlation of estimate with depth after removing the shared radius trend."""

    r_ed, r_er, r_dr = pearson(est, depth), pearson(est, radius), pearson(depth, radius)
    denom = math.sqrt(max(1e-12, (1 - r_er**2) * (1 - r_dr**2)))
    return float((r_ed - r_er * r_dr) / denom)


def evaluate(method: str, scene: str) -> dict:
    cfg = SCENES[scene]
    theta = cfg["elevation"]
    per_seed, pooled = {}, {"est": [], "gt": [], "rad": [], "slope": []}
    n_total = n_bad = 0
    for seed in SEEDS:
        syn = generate_synthetic_lunar_surface(size=512, seed=seed, **cfg["kwargs"])
        dets = ground_truth_detections(syn["image"].shape, syn["craters"])
        est, gt, rad, slope = [], [], [], []
        for det, crater in zip(dets, syn["craters"]):
            x1, y1, x2, y2 = det["x1"], det["y1"], min(511, det["x2"]), min(511, det["y2"])
            roi = syn["image"][y1:y2, x1:x2]
            n_total += 1
            mask, _ = segment(roi, det["center_x"] - x1, det["center_y"] - y1, det["radius_px"], method)
            if mask is None or int(np.count_nonzero(mask)) < 3:
                n_bad += 1
                continue
            L, _, _ = measure_shadow_length(mask, AZ)
            if L <= 0:
                n_bad += 1
                continue
            depth = L * SCALE * math.tan(math.radians(theta))
            est.append(depth)
            gt.append(crater["true_depth"])
            rad.append(crater["radius_px"])
            slope.append(math.degrees(math.atan2(depth, max(0.1, 0.5 * det["diameter_px"] * SCALE))))
        per_seed[seed] = {"n": len(est), "r_gt": pearson(est, gt)}
        for key, vals in zip(("est", "gt", "rad", "slope"), (est, gt, rad, slope)):
            pooled[key].extend(vals)
    return {
        "scene": scene,
        "method": method,
        "n_measured": len(pooled["est"]),
        "n_not_measurable": n_bad,
        "n_total": n_total,
        "seed42_r_gt": per_seed[42]["r_gt"],
        "seed42_n": per_seed[42]["n"],
        "pooled_r_gt": pearson(pooled["est"], pooled["gt"]),
        "pooled_r_radius": pearson(pooled["est"], pooled["rad"]),
        "partial_r_gt_given_radius": partial_corr(pooled["est"], pooled["gt"], pooled["rad"]),
        "slope_min": min(pooled["slope"]) if pooled["slope"] else float("nan"),
        "slope_max": max(pooled["slope"]) if pooled["slope"] else float("nan"),
        "slope_std": float(np.std(pooled["slope"])) if pooled["slope"] else float("nan"),
    }


def main() -> None:
    for scene in SCENES:
        cfg = SCENES[scene]
        depths, radii = [], []
        for seed in SEEDS:
            syn = generate_synthetic_lunar_surface(size=512, seed=seed, **cfg["kwargs"])
            depths += [c["true_depth"] for c in syn["craters"]]
            radii += [c["radius_px"] for c in syn["craters"]]
        print(f"\n===== scene: {scene} (estimator theta = {cfg['elevation']} deg) =====")
        print(f"  pooled true depth range {min(depths)!r}..{max(depths)!r}; "
              f"r(true depth, radius) = {pearson(depths, radii)!r}")
        print("  a radius-only estimator scores r_gt = r(true depth, radius); "
              "partial removes that shared trend")
        for method in METHODS:
            res = evaluate(method, scene)
            print(f"  {method:9s} n={res['n_measured']:2d}/{res['n_total']:2d} "
                  f"seed42_r_gt={res['seed42_r_gt']!r} (n={res['seed42_n']}) "
                  f"pooled_r_gt={res['pooled_r_gt']!r} r_radius={res['pooled_r_radius']!r} "
                  f"partial={res['partial_r_gt_given_radius']!r} "
                  f"slope={res['slope_min']:.2f}-{res['slope_max']:.2f} (std {res['slope_std']:.3f})")


if __name__ == "__main__":
    main()
