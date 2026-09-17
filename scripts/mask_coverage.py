"""Report shadow-mask coverage for individual craters using the shipped depth code.

Usage:
    .venv/Scripts/python scripts/mask_coverage.py [seed] [CR-01 CR-05 ...]
"""

from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from modules.depth import compute_otsu_shadow_mask, crater_circle_mask, measure_shadow_length  # noqa: E402
from modules.detector import ground_truth_detections  # noqa: E402
from utils.synthetic import generate_synthetic_lunar_surface  # noqa: E402

AZ = 35.0


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    ids = sys.argv[2:] or ["CR-01", "CR-05", "CR-08"]
    kwargs = {}
    if os.environ.get("CAST_SHADOWS") == "1":
        kwargs = {"cast_shadows": True}
    syn = generate_synthetic_lunar_surface(size=512, seed=seed, **kwargs)
    dets = ground_truth_detections(syn["image"].shape, syn["craters"])
    factor = math.cos(math.radians(AZ)) + math.sin(math.radians(AZ))
    for det in dets:
        if det["crater_id"] not in ids:
            continue
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        roi = syn["image"][y1:y2, x1:x2]
        cx, cy, r = float(det["center_x"] - x1), float(det["center_y"] - y1), float(det["radius_px"])
        mask = compute_otsu_shadow_mask(roi, center=(cx, cy), radius_px=r)
        circle = crater_circle_mask(roi.shape, (cx, cy), r)
        n_mask = int(np.count_nonzero(mask))
        n_circle = int(np.count_nonzero(circle))
        outside = int(np.count_nonzero((mask > 0) & ~circle))
        n_outside_px = int(np.count_nonzero(~circle))
        L, _, _ = measure_shadow_length(mask, AZ)
        box_L = (x2 - x1 - 1) * factor
        print(
            f"{det['crater_id']}: mask/circle={n_mask / n_circle!r} mask/ROI={n_mask / mask.size!r} "
            f"mask px outside circle={outside}/{n_outside_px} | L={L!r} box_formula={box_L!r} "
            f"L==box_formula={abs(round(L, 3) - round(box_L, 3)) < 0.0015}"
        )


if __name__ == "__main__":
    main()
