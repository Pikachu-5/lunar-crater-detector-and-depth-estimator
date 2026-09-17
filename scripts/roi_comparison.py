"""ROI window comparison with an honest radius-only baseline.

Compares two ways of cropping the analysis window for shadow measurement:
    box     - the detection box itself (previous behaviour)
    window  - a square of +/- ROI_WINDOW_RADIUS_FACTOR * radius about the crater
              centre, clipped to the image (current behaviour)

For every configuration it reports, on the SAME craters:
    - coverage (measured / detected) and the failure reasons
    - r(shadow-based depth estimate, true depth)
    - r(radius-only estimate, true depth): depth predicted from crater radius
      alone, ignoring the shadow entirely. Pearson r of a linear function of
      radius equals r(radius, true depth), so that is what is reported.
    - partial correlation of the estimate with true depth controlling for radius
    - r(estimate, radius), and the radius-only baseline over ALL detected craters

Run:
    .venv/Scripts/python scripts/roi_comparison.py
"""

from __future__ import annotations

import collections
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import modules.depth as depth_mod  # noqa: E402
from modules.depth import measure_crater_shadow  # noqa: E402
from modules.detector import detect_craters, ground_truth_detections  # noqa: E402
from modules.preprocess import preprocess_pipeline  # noqa: E402
from utils.synthetic import generate_synthetic_lunar_surface  # noqa: E402

SEEDS = [42, 7, 123]
AZ = 35.0
ELEV = 20.0
SCALE = 1.0
CONF = 0.35


def corr(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def partial_corr(est, depth, radius) -> float | None:
    r_ed, r_er, r_dr = corr(est, depth), corr(est, radius), corr(depth, radius)
    if None in (r_ed, r_er, r_dr):
        return None
    denom = math.sqrt(max(1e-12, (1 - r_er**2) * (1 - r_dr**2)))
    return float((r_ed - r_er * r_dr) / denom)


def partial_p_value(r: float | None, n: int, n_controls: int = 1) -> float | None:
    """Two-sided p-value for a partial correlation (t-test, df = n - 2 - controls)."""

    if r is None or n - 2 - n_controls <= 0 or abs(r) >= 1.0:
        return None
    from scipy import stats

    df = n - 2 - n_controls
    t = r * math.sqrt(df / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), df))


def crop(image: np.ndarray, det: dict, mode: str) -> tuple[np.ndarray, float, float]:
    """Return (roi, centre x in roi, centre y in roi) for the requested ROI mode."""

    h, w = image.shape
    if mode == "box":
        x1, y1 = int(max(0, det["x1"])), int(max(0, det["y1"]))
        x2, y2 = int(min(w - 1, det["x2"])), int(min(h - 1, det["y2"]))
        return image[y1:y2, x1:x2], float(det["center_x"] - x1), float(det["center_y"] - y1)

    half = max(3.0, depth_mod.ROI_WINDOW_RADIUS_FACTOR * float(det["radius_px"]))
    x1 = int(max(0, math.floor(det["center_x"] - half)))
    y1 = int(max(0, math.floor(det["center_y"] - half)))
    x2 = int(min(w, math.ceil(det["center_x"] + half) + 1))
    y2 = int(min(h, math.ceil(det["center_y"] + half) + 1))
    return image[y1:y2, x1:x2], float(det["center_x"] - x1), float(det["center_y"] - y1)


def evaluate(box_source: str, mode: str) -> dict:
    reasons: collections.Counter[str] = collections.Counter()
    meas = {"est": [], "gt": [], "rad": []}
    every = {"gt": [], "rad": []}
    slopes: list[float] = []
    total = 0

    for seed in SEEDS:
        syn = generate_synthetic_lunar_surface(
            size=512, seed=seed, cast_shadows=True, sun_elevation_deg=ELEV, sun_angle_deg=AZ
        )
        image = syn["image"]
        if box_source == "ground_truth":
            dets = ground_truth_detections(image.shape, syn["craters"])
        else:
            dets = detect_craters(preprocess_pipeline(image)["smoothed"], conf_threshold=CONF)["detections"]

        for det in dets:
            crater = min(
                syn["craters"],
                key=lambda c: (c["center_x"] - det["center_x"]) ** 2 + (c["center_y"] - det["center_y"]) ** 2,
            )
            total += 1
            every["gt"].append(crater["true_depth"])
            every["rad"].append(crater["radius_px"])

            roi, cx, cy = crop(image, det, mode)
            if roi.shape[0] < 3 or roi.shape[1] < 3:
                reasons["ROI too small"] += 1
                continue
            res = measure_crater_shadow(roi, (cx, cy), float(det["radius_px"]), solar_azimuth_deg=AZ)
            if res["length_px"] is None:
                reasons[(res["reason"] or "unknown").split(":")[0].split(",")[0]] += 1
                continue
            reasons["MEASURED"] += 1
            depth = res["length_px"] * SCALE * math.tan(math.radians(ELEV))
            meas["est"].append(depth)
            meas["gt"].append(crater["true_depth"])
            meas["rad"].append(crater["radius_px"])
            slopes.append(math.degrees(math.atan2(depth, max(0.1, 0.5 * det["diameter_px"] * SCALE))))

    shadow_r = corr(meas["est"], meas["gt"])
    radius_r = corr(meas["rad"], meas["gt"])
    return {
        "box_source": box_source,
        "roi_mode": mode,
        "detected": total,
        "measured": len(meas["est"]),
        "coverage": len(meas["est"]) / total if total else None,
        "shadow_r": shadow_r,
        "radius_only_r_same_craters": radius_r,
        "difference": None if (shadow_r is None or radius_r is None) else shadow_r - radius_r,
        "partial_r_given_radius": partial_corr(meas["est"], meas["gt"], meas["rad"]),
        "partial_p_value": partial_p_value(partial_corr(meas["est"], meas["gt"], meas["rad"]), len(meas["est"])),
        "r_est_vs_radius": corr(meas["est"], meas["rad"]),
        "radius_only_r_all_detected": corr(every["rad"], every["gt"]),
        "slope_min": min(slopes) if slopes else None,
        "slope_max": max(slopes) if slopes else None,
        "slope_std": float(np.std(slopes)) if slopes else None,
        "reasons": {k: v for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])},
    }


def fmt(v) -> str:
    return "n/a" if v is None else (repr(v) if isinstance(v, float) else str(v))


def main() -> None:
    print(f"Ray-cast scenes, seeds {SEEDS}, sun elevation {ELEV} deg, azimuth {AZ} deg, "
          f"ROI_WINDOW_RADIUS_FACTOR = {depth_mod.ROI_WINDOW_RADIUS_FACTOR}")
    rows = []
    for box_source in ("ground_truth", "yolo"):
        for mode in ("box", "window"):
            rows.append(evaluate(box_source, mode))

    for r in rows:
        print(f"\n--- boxes={r['box_source']}  ROI={r['roi_mode']} ---")
        print(f"  coverage: {r['measured']}/{r['detected']} = {fmt(r['coverage'])}")
        print(f"  shadow estimate vs true depth : {fmt(r['shadow_r'])}")
        print(f"  radius-only, SAME craters     : {fmt(r['radius_only_r_same_craters'])}")
        print(f"  difference (shadow - radius)  : {fmt(r['difference'])}")
        print(f"  partial r given radius        : {fmt(r['partial_r_given_radius'])} "
              f"(p = {fmt(r['partial_p_value'])}, n = {r['measured']})")
        print(f"  r(estimate, radius)           : {fmt(r['r_est_vs_radius'])}")
        print(f"  radius-only, ALL detected     : {fmt(r['radius_only_r_all_detected'])}")
        print(f"  slope spread                  : {fmt(r['slope_min'])} .. {fmt(r['slope_max'])} "
              f"(std {fmt(r['slope_std'])})")
        print(f"  reasons: {r['reasons']}")

    print("\nHeadline: does the shadow measurement beat radius-only on the same craters?")
    for r in rows:
        if r["difference"] is None:
            verdict = "cannot tell (too few measured)"
        elif r["difference"] > 0:
            verdict = f"yes, by {r['difference']!r}"
        else:
            verdict = f"NO, worse by {abs(r['difference'])!r}"
        print(f"  boxes={r['box_source']:12s} ROI={r['roi_mode']:6s}: {verdict}")


if __name__ == "__main__":
    main()
