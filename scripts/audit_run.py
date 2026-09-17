"""Headless instrumented audit of the crater pipeline (Parts B and C of AUDIT.md).

Imports the project's own modules directly and reproduces the call sequence
used by app.py (preprocess -> detect -> depth -> terrain -> score -> A*),
with the same default parameters as the sidebar / session-state defaults.

Outputs:
    scripts/audit_results.json  - every raw number
    scripts/audit_tables.md     - markdown tables pasted into AUDIT.md

Run from the project root:
    .venv/Scripts/python scripts/audit_run.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import modules.depth as depth_mod  # noqa: E402
from modules.depth import compute_otsu_shadow_mask, estimate_crater_depths, measure_shadow_length  # noqa: E402
from modules.detector import _to_detection_record, detect_craters, detect_craters_cv, ground_truth_detections  # noqa: E402
from modules.pathfinder import _downsample_score_map, plan_descent_paths  # noqa: E402
from modules.preprocess import preprocess_pipeline  # noqa: E402
from modules.scorer import annotate_hazard_map, build_score_map, score_landing_safety  # noqa: E402
from modules.terrain3d import build_depth_map, make_contour_figure, make_heatmap_figure, make_surface_figure  # noqa: E402
from utils.synthetic import CraterSpec, _apply_crater_signature, generate_synthetic_lunar_surface  # noqa: E402

# Defaults exactly as in app.py (line refs in AUDIT.md).
THETA = 35            # app.py:555
AZIMUTH = 35          # app.py:558
TD = 1.8              # app.py:561
GEAR = 2.6            # app.py:564
DENSITY = 85          # app.py:567
PIXEL_SCALE = 1.0     # app.py:570
CLIP = 2.2            # app.py:127
GRID = 8              # app.py:129
SIGMA = 1.2           # app.py:131
CONF = 0.35           # app.py:336
TERRAIN_TARGET = 512  # _terrain_profile_target_size("High Fidelity (512 MB)", (512, 512)) -> clip(min(512,720),380,720)
SIZE = 512            # app.py:83
SEEDS = [42, 7, 123]  # 42 is the generator default the app actually uses (synthetic.py seed=42)
N_TIMED_RUNS = 5

OUT_JSON = os.path.join(ROOT, "scripts", "audit_results.json")
OUT_MD = os.path.join(ROOT, "scripts", "audit_tables.md")


def detect_ground_truth(image: np.ndarray, craters: list[dict[str, Any]]) -> dict[str, Any]:
    """Ground-truth boxes from generator metadata (not model output); replaces the removed hint_craters path."""

    return {"detections": ground_truth_detections(image.shape, craters), "source": "ground-truth",
            "status": f"ground truth, not model output: {len(craters)} craters"}


def t_now() -> float:
    return time.perf_counter()


def mean_or_none(xs: list[Any]) -> float | None:
    """Mean of the values that exist; None when nothing is measurable."""

    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def summary(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": min(xs),
        "median": statistics.median(xs),
        "max": max(xs),
        "mean": statistics.fmean(xs),
    }


# ---------------------------------------------------------------------------
# Ground truth replay (synthetic.py does not return depth_scale; replay its RNG)
# ---------------------------------------------------------------------------
def replay_crater_specs(seed: int, size: int = SIZE) -> list[CraterSpec]:
    """Replay generate_synthetic_lunar_surface's RNG draws in identical order."""

    rng = np.random.default_rng(seed)
    rng.normal(loc=120.0, scale=10.0, size=(size, size))  # synthetic.py:111
    count = int(rng.integers(8, 12 + 1))                     # synthetic.py:115
    specs = []
    for i in range(count):
        radius = int(rng.integers(10, 61))                   # synthetic.py:119
        cx = int(rng.integers(radius + 8, size - radius - 8))
        cy = int(rng.integers(radius + 8, size - radius - 8))
        specs.append(
            CraterSpec(
                crater_id=f"CR-{i+1:02d}",
                cx=cx,
                cy=cy,
                radius=radius,
                depth_scale=float(rng.uniform(25.0, 60.0)),
                rim_scale=float(rng.uniform(8.0, 18.0)),
            )
        )
    return specs


def verify_replay(synth: dict[str, Any], specs: list[CraterSpec]) -> bool:
    hm = np.zeros((SIZE, SIZE), dtype=np.float32)
    luma = np.zeros((SIZE, SIZE), dtype=np.float32)
    for s in specs:
        _apply_crater_signature(hm, luma, s, sun_angle_deg=35.0)
    centres_ok = [(c["center_x"], c["center_y"], c["radius_px"]) for c in synth["craters"]] == [
        (s.cx, s.cy, s.radius) for s in specs
    ]
    return bool(centres_ok and np.array_equal(hm, synth["height_map"]))


def rim_to_floor_relief(height_map: np.ndarray, s: CraterSpec) -> float:
    """Max height on rim annulus [0.9r,1.1r] minus min height within 0.3r of centre."""

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    d = np.sqrt((xx - s.cx) ** 2 + (yy - s.cy) ** 2)
    rim = height_map[(d >= 0.9 * s.radius) & (d <= 1.1 * s.radius)]
    floor = height_map[d <= max(1.0, 0.3 * s.radius)]
    return float(rim.max() - floor.min())


# ---------------------------------------------------------------------------
# Detection matching (synthetic ground truth boxes)
# ---------------------------------------------------------------------------
def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return 0.0 if ua <= 0 else inter / ua


def match_to_gt(dets: list[dict[str, Any]], gt_craters: list[dict[str, Any]], thr: float = 0.5) -> int:
    gt = [
        _to_detection_record("gt", c["center_x"], c["center_y"], c["radius_px"], 1.0, "gt", (SIZE, SIZE))
        for c in gt_craters
    ]
    pairs = sorted(
        ((iou(d, g), i, j) for i, d in enumerate(dets) for j, g in enumerate(gt)), reverse=True
    )
    used_d, used_g, matched = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        matched += 1
    return matched


# ---------------------------------------------------------------------------
# Zero-depth diagnosis (mirrors depth.py:127-138)
# ---------------------------------------------------------------------------
def diagnose_depth(image: np.ndarray, dets: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    h, w = image.shape
    by_id = {r["crater_id"]: r for r in rows}
    out = []
    for det in dets:
        x1, y1 = int(max(0, det["x1"])), int(max(0, det["y1"]))
        x2, y2 = int(min(w - 1, det["x2"])), int(min(h - 1, det["y2"]))
        rec: dict[str, Any] = {"crater_id": det["crater_id"]}
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            rec["reason"] = "ROI <= 2 px on an axis: crater skipped, no row emitted (depth.py:133)"
            out.append(rec)
            continue
        roi = image[y1:y2, x1:x2]
        blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0, sigmaY=1.0)
        otsu_t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pre_morph = int(np.count_nonzero(blur < otsu_t))
        mask = compute_otsu_shadow_mask(
            roi,
            center=(float(det["center_x"] - x1), float(det["center_y"] - y1)),
            radius_px=float(det["radius_px"]),
        )
        n = int(np.count_nonzero(mask))
        L, _, _ = measure_shadow_length(mask, AZIMUTH)
        rec.update(
            {
                "roi_w": x2 - x1,
                "roi_h": y2 - y1,
                "otsu_threshold": float(otsu_t),
                "shadow_px_before_morph": pre_morph,
                "shadow_px_after_morph": n,
                "shadow_length_px_unrounded": L,
                "depth_m": by_id[det["crater_id"]]["depth_m"],
            }
        )
        row = by_id[det["crater_id"]]
        rec["measurable"] = bool(row.get("measurable", row["depth_m"] is not None))
        if not rec["measurable"]:
            rec["reason"] = row.get("not_measurable_reason") or "not measurable (no reason recorded)"
        elif row["depth_m"] == 0.0:
            rec["reason"] = "depth is exactly 0.0 m"
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(synth: dict[str, Any], use_hints: bool, timed: bool = True) -> dict[str, Any]:
    raw = synth["image"]
    tm: dict[str, float] = {}

    t = t_now()
    pp = preprocess_pipeline(raw, clip_limit=CLIP, tile_grid_size=(GRID, GRID), sigma=SIGMA)
    tm["preprocess"] = t_now() - t

    t = t_now()
    det = detect_ground_truth(pp["smoothed"], synth["craters"]) if use_hints else detect_craters(pp["smoothed"], conf_threshold=CONF)
    tm["detection"] = t_now() - t
    dets = det["detections"]

    t = t_now()
    depth = estimate_crater_depths(raw, dets, solar_elevation_angle_deg=THETA, solar_azimuth_deg=AZIMUTH, pixel_scale_m=PIXEL_SCALE)
    tm["depth"] = t_now() - t

    t = t_now()
    dmap = build_depth_map(raw.shape, depth["rows"])
    tm["terrain_depth_map"] = t_now() - t
    t = t_now()
    make_heatmap_figure(dmap)
    make_surface_figure(dmap, downsample=True, target_size=TERRAIN_TARGET)
    make_contour_figure(dmap)
    tm["terrain_figures"] = t_now() - t
    tm["terrain_total"] = tm["terrain_depth_map"] + tm["terrain_figures"]

    t = t_now()
    scoring = score_landing_safety(depth["rows"], TD, GEAR, DENSITY, PIXEL_SCALE)
    hazard_map = annotate_hazard_map(raw, scoring["rows"])
    score_map = build_score_map(raw.shape, scoring["rows"])
    tm["scoring"] = t_now() - t

    t = t_now()
    paths = plan_descent_paths(score_map, scoring["rows"], PIXEL_SCALE)
    tm["astar"] = t_now() - t

    return {
        "timing": tm,
        "detection": det,
        "pp": pp,
        "depth": depth,
        "scoring": scoring,
        "score_map": score_map,
        "paths": paths,
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main() -> None:
    import torch

    results: dict[str, Any] = {
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        }
    }
    md: list[str] = []

    synths = {s: generate_synthetic_lunar_surface(size=SIZE, seed=s) for s in SEEDS}
    determinism = np.array_equal(synths[42]["image"], generate_synthetic_lunar_surface(size=SIZE, seed=42)["image"])
    app_default_equals_seed42 = np.array_equal(generate_synthetic_lunar_surface(size=SIZE)["image"], synths[42]["image"])
    results["synthetic_determinism_seed42"] = bool(determinism)
    results["app_default_call_equals_seed42"] = bool(app_default_equals_seed42)

    # ---------------- B1 timing ----------------
    s42 = synths[42]
    warm = {}
    for label, hints in (("yolo", False), ("synthetic_meta", True)):
        t = t_now()
        r = run_pipeline(s42, use_hints=hints)
        warm[label] = {"wall_total": t_now() - t, "stages": r["timing"]}
    results["B1_warmup_run_excluded"] = warm

    timing: dict[str, dict[str, list[float]]] = {"yolo": {}, "synthetic_meta": {}}
    for _ in range(N_TIMED_RUNS):
        for label, hints in (("yolo", False), ("synthetic_meta", True)):
            r = run_pipeline(s42, use_hints=hints)
            for k, v in r["timing"].items():
                timing[label].setdefault(k, []).append(v)
    results["B1_timing_raw_s"] = timing
    b1_rows = []
    for label in ("yolo", "synthetic_meta"):
        for k, vs in timing[label].items():
            b1_rows.append([label, k, repr(statistics.fmean(vs)), repr(statistics.stdev(vs)), ", ".join(repr(v) for v in vs)])
    md.append("### B1 timing (seconds, seed 42, 5 runs after 1 excluded warm-up)\n")
    md.append(md_table(["detection path", "stage", "mean", "stdev (sample)", "raw runs"], b1_rows))
    md.append("\nWarm-up run (excluded; includes YOLO weight load on first call): "
              + json.dumps({k: {"wall_total": v["wall_total"], **v["stages"]} for k, v in warm.items()}))

    # ---------------- B2..B5 per seed ----------------
    per_seed: dict[int, Any] = {}
    b2_rows, b4_rows, b5_rows = [], [], []
    for seed in SEEDS:
        syn = synths[seed]
        gt_n = len(syn["craters"])
        ry = run_pipeline(syn, use_hints=False)
        rh = run_pipeline(syn, use_hints=True)
        cv = detect_craters_cv(ry["pp"]["smoothed"])
        seed_res: dict[str, Any] = {"gt_crater_count": gt_n}
        for label, dets in (("yolo", ry["detection"]["detections"]), ("cv_hybrid", cv["detections"]),
                            ("synthetic_meta", rh["detection"]["detections"])):
            confs = [d["confidence"] for d in dets if d["confidence"] is not None]
            diams = [d["diameter_px"] for d in dets]
            m = match_to_gt(dets, syn["craters"])
            seed_res[label] = {
                "count": len(dets),
                "confidence": summary(confs),
                "diameter_px": summary(diams),
                "matched_gt_iou50": m,
                "status": (ry["detection"] if label == "yolo" else cv if label == "cv_hybrid" else rh["detection"])["status"],
                "detections": dets,
            }
            cs, ds = summary(confs), summary(diams)
            b2_rows.append([seed, gt_n, label, len(dets),
                            repr(cs.get("min")), repr(cs.get("median")), repr(cs.get("max")),
                            repr(ds.get("min")), repr(ds.get("max")), f"{m}/{gt_n}"])

        for label, r in (("yolo", ry), ("synthetic_meta", rh)):
            dets = r["detection"]["detections"]
            rows = r["depth"]["rows"]
            diag = diagnose_depth(syn["image"], dets, rows)
            sc = r["scoring"]
            scores = [x["safety_score"] for x in sc["rows"]]
            p = r["paths"]
            ds_map, _, _ = _downsample_score_map(r["score_map"])
            seed_res[f"{label}_depth_rows"] = rows
            seed_res[f"{label}_depth_diagnostics"] = diag
            seed_res[f"{label}_scoring"] = {"rows": sc["rows"], "summary": sc["summary"], "overall": sc["overall_score"], "scores": summary(scores)}
            seed_res[f"{label}_paths"] = {
                "route_found": len(p["primary_path"]) > 0,
                "path_length_m": p["path_length_m"],
                "path_nodes": len(p["primary_path"]),
                "alternatives_succeeded": len(p["alternative_paths"]),
                "alt_nodes": [len(a) for a in p["alternative_paths"]],
                "start": p["start"],
                "goal": p["goal"],
                "planning_grid_shape": list(ds_map.shape),
                "goal_crater_id": p["goal_crater_id"],
                "goal_zone": p["goal_zone"],
                "hazard_craters_crossed": p["hazard_craters_crossed"],
                "hazard_craters_other_than_goal": p["hazard_craters_other_than_goal"],
            }
            ss = summary(scores)
            b4_rows.append([seed, label, len(scores), repr(ss.get("min")), repr(ss.get("median")), repr(ss.get("max")),
                            repr(ss.get("mean")), sc["summary"]["safe"], sc["summary"]["caution"], sc["summary"]["hazard"],
                            sc["summary"].get("unknown", 0), repr(sc["overall_score"])])
            pr = seed_res[f"{label}_paths"]
            b5_rows.append([seed, label, pr["route_found"], repr(pr["path_length_m"]), pr["path_nodes"],
                            f"{pr['alternatives_succeeded']}/3", pr["alt_nodes"], pr["start"], pr["goal"],
                            "x".join(map(str, pr["planning_grid_shape"])), pr["goal_zone"], f"{len(pr['hazard_craters_crossed'])}/{pr['hazard_craters_other_than_goal']}"])
        per_seed[seed] = seed_res
    results["per_seed"] = per_seed

    md.append("\n### B2 detection (defaults: CLAHE 2.2/8, sigma 1.2, conf 0.35)\n")
    md.append(md_table(["seed", "GT craters", "detector", "count", "conf min", "conf median", "conf max",
                        "diam px min", "diam px max", "matched GT @IoU>=0.5"], b2_rows))

    md.append("\n### B3 depth tables (seed 42, theta 35, azimuth 35, 1.0 m/px)\n")
    for label in ("synthetic_meta", "yolo"):
        diag = {d["crater_id"]: d for d in per_seed[42][f"{label}_depth_diagnostics"]}
        rows = []
        for r in per_seed[42][f"{label}_depth_rows"]:
            dg = diag[r["crater_id"]]
            fmt = lambda v: "not measurable" if v is None else v
            rows.append([r["crater_id"], fmt(r["shadow_length_px"]), fmt(r["depth_m"]), fmt(r["slope_estimate_deg"]),
                         r["confidence"], r["diameter_px"], dg.get("otsu_threshold", "-"),
                         dg.get("shadow_px_after_morph", "-"), dg.get("reason", "")])
        md.append(f"\n#### {label}\n")
        md.append(md_table(["id", "shadow length px", "depth m", "slope deg", "confidence", "diameter px",
                            "Otsu T", "shadow px", "not-measurable reason"], rows))
        skipped = [d for d in diag.values() if "roi_w" not in d]
        bad = [d for d in diag.values() if not d.get("measurable", True)]
        md.append(f"\nrows={len(per_seed[42][f'{label}_depth_rows'])}, skipped (no row)={len(skipped)}, "
                  f"not measurable={len(bad)}")
    nm_counts = []
    pooled_nm = {"yolo": [0, 0], "synthetic_meta": [0, 0]}
    for seed in SEEDS:
        for label in ("yolo", "synthetic_meta"):
            diag = per_seed[seed][f"{label}_depth_diagnostics"]
            bad = [d for d in diag if not d.get("measurable", True)]
            sk = [d for d in diag if "roi_w" not in d]
            n_rows = len(per_seed[seed][f"{label}_depth_rows"])
            pooled_nm[label][0] += len(bad)
            pooled_nm[label][1] += n_rows
            nm_counts.append([seed, label, n_rows, len(bad), len(sk),
                              "; ".join(sorted({d["reason"] for d in bad + sk})) or "-"])
    results["not_measurable_pooled"] = {k: {"not_measurable": v[0], "rows": v[1]} for k, v in pooled_nm.items()}
    md.append("\n#### Not-measurable count across seeds\n")
    md.append(md_table(["seed", "detections", "rows", "not measurable", "skipped", "reasons"], nm_counts))
    md.append("\nPooled: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in pooled_nm.items()))

    md.append("\n### B4 scoring (Td 1.8, gear 2.6, density 85 px, 1.0 m/px)\n")
    md.append(md_table(["seed", "detections", "n", "score min", "median", "max", "mean", "SAFE", "CAUTION", "HAZARD",
                        "UNKNOWN", "overall_score"], b4_rows))
    md.append("\n### B5 pathfinding\n")
    md.append(md_table(["seed", "detections", "route found", "path length m", "path nodes", "alternatives ok",
                        "alt nodes", "start px", "goal px", "A* grid (h x w)", "goal zone", "HAZARD craters crossed (excl. goal)"], b5_rows))

    # ---------------- C1 solar angle sweep ----------------
    syn = synths[42]
    pp = preprocess_pipeline(syn["image"], clip_limit=CLIP, tile_grid_size=(GRID, GRID), sigma=SIGMA)
    det_sets = {
        "synthetic_meta": detect_ground_truth(pp["smoothed"], syn["craters"])["detections"],
        "yolo": detect_craters(pp["smoothed"], CONF)["detections"],
    }
    original_fn = depth_mod.depth_from_shadow

    def inverse_tan(shadow_length_px: float, solar_elevation_angle_deg: float, pixel_scale_m: float) -> float:
        theta = math.radians(np.clip(solar_elevation_angle_deg, 1.0, 89.0))
        return float(shadow_length_px * pixel_scale_m / math.tan(theta))

    c1: dict[str, list[dict[str, Any]]] = {}
    for label, dets in det_sets.items():
        c1[label] = []
        for theta in range(5, 86, 10):
            depth_mod.depth_from_shadow = original_fn
            a = estimate_crater_depths(syn["image"], dets, theta, AZIMUTH, PIXEL_SCALE)["rows"]
            depth_mod.depth_from_shadow = inverse_tan
            b = estimate_crater_depths(syn["image"], dets, theta, AZIMUTH, PIXEL_SCALE)["rows"]
            depth_mod.depth_from_shadow = original_fn
            c1[label].append({
                "theta": theta,
                "tan": math.tan(math.radians(theta)),
                "mean_depth_tan_code": mean_or_none([r["depth_m"] for r in a]),
                "mean_depth_inverse_tan": mean_or_none([r["depth_m"] for r in b]),
                "mean_shadow_length_px": mean_or_none([r["shadow_length_px"] for r in a]),
                "n": len(a),
                "n_measurable": sum(1 for r in a if r["depth_m"] is not None),
            })
    results["C1"] = c1
    for label, rows in c1.items():
        md.append(f"\n### C1 solar angle sweep — seed 42, detections={label}, azimuth 35, 1.0 m/px\n")
        md.append(md_table(["theta deg", "tan(theta)", "mean shadow length px", "mean depth m (code: × tan)",
                            "mean depth m (alt: ÷ tan)", "n craters", "n measurable"],
                           [[r["theta"], repr(r["tan"]), repr(r["mean_shadow_length_px"]), repr(r["mean_depth_tan_code"]),
                             repr(r["mean_depth_inverse_tan"]), r["n"], r["n_measurable"]] for r in rows]))

    # ---------------- C2 ground truth ----------------
    specs = replay_crater_specs(42)
    replay_ok = verify_replay(syn, specs)
    rows = estimate_crater_depths(syn["image"], det_sets["synthetic_meta"], THETA, AZIMUTH, PIXEL_SCALE)["rows"]
    c2_rows, errs, rel_errs, signed = [], [], [], []
    # Ground truth comes straight from the generator's metadata (utils/synthetic.py
    # "true_depth"); the RNG replay is kept only as an independent cross-check.
    true_depths = [c["true_depth"] for c in syn["craters"]]
    returned_equals_replay = true_depths == [s.depth_scale for s in specs]
    n_c2_skipped = 0
    for s, r, gt_depth in zip(specs, rows, true_depths):
        assert s.crater_id == r["crater_id"]
        if r["depth_m"] is None:
            n_c2_skipped += 1
            continue
        relief = rim_to_floor_relief(syn["height_map"], s)
        e = r["depth_m"] - gt_depth
        errs.append(abs(e))
        signed.append(e)
        rel_errs.append(abs(r["depth_m"] - relief))
        c2_rows.append({"crater_id": s.crater_id, "radius_px": s.radius, "gt_depth_scale": gt_depth,
                        "gt_rim_to_floor_relief": relief, "est_depth_m": r["depth_m"],
                        "shadow_length_px": r["shadow_length_px"], "error_vs_depth_scale": e,
                        "abs_error_vs_depth_scale": abs(e), "error_vs_relief": r["depth_m"] - relief})
    est = np.array([r["est_depth_m"] for r in c2_rows])
    gt = np.array([r["gt_depth_scale"] for r in c2_rows])
    rad = np.array([r["radius_px"] for r in c2_rows], dtype=float)
    shadow = np.array([r["shadow_length_px"] for r in c2_rows])
    c2 = {
        "replay_verified_bitwise_equal_height_map": replay_ok,
        "craters_skipped_not_measurable": n_c2_skipped,
        "returned_true_depth_equals_replay": returned_equals_replay,
        "rows": c2_rows,
        "MAE_vs_depth_scale": statistics.fmean(errs),
        "mean_signed_error_vs_depth_scale": statistics.fmean(signed),
        "n_over": sum(1 for e in signed if e > 0),
        "n_under": sum(1 for e in signed if e < 0),
        "MAE_vs_rim_to_floor_relief": statistics.fmean(rel_errs),
        "mean_signed_error_vs_relief": statistics.fmean(r["error_vs_relief"] for r in c2_rows),
        "pearson_est_vs_depth_scale": float(np.corrcoef(est, gt)[0, 1]),
        "pearson_est_vs_radius_px": float(np.corrcoef(est, rad)[0, 1]),
        "pearson_shadow_len_vs_radius_px": float(np.corrcoef(shadow, rad)[0, 1]),
    }
    results["C2"] = c2
    md.append(f"\n### C2 ground truth (seed 42, synthetic_meta detections, theta 35, 1.0 m/px) — replay verified: {replay_ok}\n")
    md.append(md_table(["id", "radius px", "GT depth_scale (height units)", "GT rim-to-floor relief (height units)",
                        "est depth_m", "est − depth_scale", "|est − depth_scale|", "est − relief"],
                       [[r["crater_id"], r["radius_px"], repr(r["gt_depth_scale"]), repr(r["gt_rim_to_floor_relief"]),
                         r["est_depth_m"], repr(r["error_vs_depth_scale"]), repr(r["abs_error_vs_depth_scale"]),
                         repr(r["error_vs_relief"])] for r in c2_rows]))
    md.append("\n" + "\n".join(f"- {k}: {v!r}" for k, v in c2.items() if k != "rows"))

    # ---------------- C2b depth estimator validity (pooled seeds, ground-truth boxes) ----------------
    factor = math.cos(math.radians(AZIMUTH)) + math.sin(math.radians(AZIMUTH))
    pool = {"est": [], "gt": [], "rad": [], "slope": []}
    n_box_formula, n_rows, n_unmeasurable = 0, 0, 0
    for seed in SEEDS:
        sy = synths[seed]
        spp = preprocess_pipeline(sy["image"], clip_limit=CLIP, tile_grid_size=(GRID, GRID), sigma=SIGMA)
        gdets = detect_ground_truth(spp["smoothed"], sy["craters"])["detections"]
        gt_by_id = {f"CR-{i + 1:02d}": c for i, c in enumerate(sy["craters"])}
        for r in estimate_crater_depths(sy["image"], gdets, THETA, AZIMUTH, PIXEL_SCALE)["rows"]:
            n_rows += 1
            if r["depth_m"] is None or not math.isfinite(r["depth_m"]):
                n_unmeasurable += 1
                continue
            side = r["x2"] - r["x1"]
            if abs(r["shadow_length_px"] - round((side - 1) * factor, 3)) < 0.0015:
                n_box_formula += 1
            pool["est"].append(r["depth_m"])
            pool["gt"].append(gt_by_id[r["crater_id"]]["true_depth"])
            pool["rad"].append(gt_by_id[r["crater_id"]]["radius_px"])
            pool["slope"].append(r["slope_estimate_deg"])
    c2b = {
        "seeds": SEEDS,
        "rows": n_rows,
        "not_measurable": n_unmeasurable,
        "pearson_est_vs_true_depth": float(np.corrcoef(pool["est"], pool["gt"])[0, 1]),
        "pearson_est_vs_radius_px": float(np.corrcoef(pool["est"], pool["rad"])[0, 1]),
        "slope_min_deg": min(pool["slope"]),
        "slope_max_deg": max(pool["slope"]),
        "slope_std_deg": float(np.std(pool["slope"])),
        "L_equals_(side-1)(cos+sin)": f"{n_box_formula}/{n_rows - n_unmeasurable}",
    }
    results["C2b"] = c2b
    md.append("\n### C2b depth estimator validity (seeds 42/7/123 pooled, ground-truth boxes, theta 35, azimuth 35)\n")
    md.append("\n".join(f"- {k}: {v!r}" for k, v in c2b.items()))

    # ---------------- C3 parameter sensitivity ----------------
    td_vals = [round(0.5 + 0.1 * i, 1) for i in range(46)]
    gear_vals = [round(1.0 + 0.1 * i, 1) for i in range(41)]
    c3: dict[str, Any] = {}
    for label, dets in det_sets.items():
        drows = estimate_crater_depths(syn["image"], dets, THETA, AZIMUTH, PIXEL_SCALE)["rows"]

        def counts(td: float, gear: float) -> tuple[int, int, int]:
            s = score_landing_safety(drows, td, gear, DENSITY, PIXEL_SCALE)["summary"]
            return s["safe"], s["caution"], s["hazard"]

        td_sweep = [[td, *counts(td, GEAR)] for td in td_vals]
        gear_sweep = [[g, *counts(TD, g)] for g in gear_vals]
        grid = {(td, g): counts(td, g) for td in td_vals for g in gear_vals}
        c3[label] = {
            "td_sweep": td_sweep,
            "gear_sweep": gear_sweep,
            "grid_distinct_outcomes": sorted({v for v in grid.values()}),
            "grid_corners": {f"td={td},gear={g}": grid[(td, g)] for td in (0.5, 5.0) for g in (1.0, 5.0)},
        }
        md.append(f"\n### C3 depth threshold sweep — seed 42, detections={label} (gear 2.6, density 85)\n")
        md.append(md_table(["Td m", "SAFE", "CAUTION", "HAZARD"], td_sweep))
        md.append(f"\n### C3 gear span sweep — seed 42, detections={label} (Td 1.8, density 85)\n")
        md.append(md_table(["gear m", "SAFE", "CAUTION", "HAZARD"], gear_sweep))
        md.append(f"\nFull 46×41 grid distinct (SAFE, CAUTION, HAZARD) outcomes: {c3[label]['grid_distinct_outcomes']}; "
                  f"corners: {c3[label]['grid_corners']}")
    results["C3"] = c3

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", OUT_JSON, "and", OUT_MD)


if __name__ == "__main__":
    main()
