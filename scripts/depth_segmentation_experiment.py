"""Compare shadow-segmentation candidates for modules/depth.py against synthetic ground truth.

Candidates (all use the ground-truth boxes so detection error is excluded):
    current   - modules.depth as shipped (compute_otsu_shadow_mask, full ROI)
    a0_fixed  - current method with only the threshold-unpacking bug fixed (full ROI, no other change)
    a_otsu    - Otsu on pixels inside the crater circle
    b_pXX     - XXth percentile of interior pixels
    c_kX      - interior mean - X * interior std
Candidates a/b/c share: circle restriction, 3x3 open/close, reject components whose
centroid is on the lit side, keep the largest remaining component, and return NaN
when the mask covers more than 60% of the crater area.

Ground truth depth = bowl amplitude (depth_scale) recovered by replaying the
generator RNG (scripts/audit_run.py::replay_crater_specs, verified bitwise).
"""

from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from audit_run import replay_crater_specs  # noqa: E402
from modules.depth import compute_otsu_shadow_mask, measure_shadow_length  # noqa: E402
from modules.detector import ground_truth_detections  # noqa: E402
from modules.preprocess import preprocess_pipeline  # noqa: E402
from utils.synthetic import generate_synthetic_lunar_surface  # noqa: E402

SEEDS = [42, 7, 123]
THETA, AZ, SCALE = 35.0, 35.0, 1.0
FACTOR = math.cos(math.radians(AZ)) + math.sin(math.radians(AZ))
KERNEL = np.ones((3, 3), np.uint8)


def segment(roi: np.ndarray, cx: float, cy: float, r: float, method: str) -> tuple[np.ndarray | None, float]:
    blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0, sigmaY=1.0)
    if method == "current":
        return compute_otsu_shadow_mask(roi), float("nan")
    if method == "a0_fixed":
        t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = np.where(blur < t, 255, 0).astype(np.uint8)
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)
        return m, float("nan")

    h, w = roi.shape
    yy, xx = np.mgrid[0:h, 0:w]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    interior = blur[circle]
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

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    v = np.array([math.cos(math.radians(AZ)), math.sin(math.radians(AZ))])
    best, best_area = 0, 0
    for i in range(1, n):
        side = (cents[i, 0] - cx) * v[0] + (cents[i, 1] - cy) * v[1]
        if side > 0:  # lit half (see mean-intensity check in AUDIT_AFTER.md)
            continue
        if stats[i, cv2.CC_STAT_AREA] > best_area:
            best, best_area = i, stats[i, cv2.CC_STAT_AREA]
    if best == 0:
        return None, 0.0
    m = np.where(labels == best, 255, 0).astype(np.uint8)
    frac = best_area / float(np.count_nonzero(circle))
    if frac > 0.60:
        return None, frac
    return m, frac


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 else float("nan")


def evaluate(method: str) -> dict:
    per_seed = {}
    pooled = {"est": [], "gt": [], "rad": [], "slope": [], "L": [], "boxL": []}
    nan_total = 0
    for seed in SEEDS:
        syn = generate_synthetic_lunar_surface(size=512, seed=seed)
        specs = replay_crater_specs(seed)
        pp = preprocess_pipeline(syn["image"])
        dets = ground_truth_detections(syn["image"].shape, syn["craters"])
        est, gt, rad, slope, Ls, boxL, fracs = [], [], [], [], [], [], []
        nan = 0
        for d, s in zip(dets, specs):
            x1, y1, x2, y2 = d["x1"], d["y1"], min(511, d["x2"]), min(511, d["y2"])
            roi = syn["image"][y1:y2, x1:x2]
            mask, frac = segment(roi, d["center_x"] - x1, d["center_y"] - y1, d["radius_px"], method)
            if mask is None:
                nan += 1
                continue
            L, _, _ = measure_shadow_length(mask, AZ)
            depth = L * SCALE * math.tan(math.radians(THETA))
            est.append(depth)
            gt.append(s.depth_scale)
            rad.append(s.radius)
            slope.append(math.degrees(math.atan2(depth, max(0.1, 0.5 * d["diameter_px"] * SCALE))))
            Ls.append(L)
            boxL.append((x2 - x1 - 1) * FACTOR)
            fracs.append(frac)
        per_seed[seed] = {"r_gt": pearson(est, gt), "n": len(est), "nan": nan}
        nan_total += nan
        for k, vals in zip(pooled, (est, gt, rad, slope, Ls, boxL)):
            pooled[k].extend(vals)
    L_eq_box = sum(1 for a, b in zip(pooled["L"], pooled["boxL"]) if abs(round(a, 3) - round(b, 3)) < 0.0015)
    return {
        "per_seed": per_seed,
        "pooled_r_gt": pearson(pooled["est"], pooled["gt"]),
        "pooled_r_radius": pearson(pooled["est"], pooled["rad"]),
        "slope_min": min(pooled["slope"]) if pooled["slope"] else float("nan"),
        "slope_max": max(pooled["slope"]) if pooled["slope"] else float("nan"),
        "slope_std": float(np.std(pooled["slope"])) if pooled["slope"] else float("nan"),
        "L_equals_box_formula": f"{L_eq_box}/{len(pooled['L'])}",
        "n_measured": len(pooled["est"]),
        "n_nan": nan_total,
    }


def dump_masks(method: str, seed: int = 42, ids=("CR-01", "CR-05", "CR-08")) -> None:
    out = os.path.join(ROOT, "scripts", "audit_masks")
    os.makedirs(out, exist_ok=True)
    syn = generate_synthetic_lunar_surface(size=512, seed=seed)
    dets = ground_truth_detections(syn["image"].shape, syn["craters"])
    for d in dets:
        if d["crater_id"] not in ids:
            continue
        roi = syn["image"][d["y1"]:d["y2"], d["x1"]:d["x2"]]
        mask, frac = segment(roi, d["center_x"] - d["x1"], d["center_y"] - d["y1"], d["radius_px"], method)
        m = mask if mask is not None else np.zeros_like(roi)
        over = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        over[m > 0] = (0.5 * over[m > 0] + [0, 0, 127]).astype(np.uint8)
        cv2.circle(over, (d["center_x"] - d["x1"], d["center_y"] - d["y1"]), d["radius_px"], (0, 255, 0), 1)
        big = lambda a: cv2.resize(a, (a.shape[1] * 4, a.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out, f"{method}_{d['crater_id']}.png"),
                    np.hstack([big(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)), big(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)), big(over)]))
        print(f"  {method} {d['crater_id']}: mask/ROI={np.count_nonzero(m) / m.size!r} mask/circle={frac!r} valid={mask is not None}")


if __name__ == "__main__":
    methods = ["current", "a0_fixed", "a_otsu", "b_p15", "b_p20", "b_p25", "b_p30", "c_k0.5", "c_k1.0"]
    for meth in methods:
        print(meth, evaluate(meth))
    if len(sys.argv) > 1:
        for meth in sys.argv[1:]:
            dump_masks(meth)
