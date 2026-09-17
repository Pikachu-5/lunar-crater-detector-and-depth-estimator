"""Sanity checks for the ray-cast shadow generator (item 3 of AUDIT_FINAL.md).

1. Shadows lengthen as the sun elevation drops.
2. Shadows fall on the expected side of each crater relative to the sun.
3. Shadow length across craters tracks true depth (the old shadow-free scene
   could not: r(true depth, radius) = -0.199, depth lived in brightness only).

Run:
    .venv/Scripts/python scripts/shadow_sanity.py
"""

from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from utils.synthetic import generate_synthetic_lunar_surface  # noqa: E402

SEEDS = [42, 7, 123]
AZ = 35.0
ELEVATIONS = [15.0, 30.0, 45.0, 60.0, 75.0]


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def crater_shadow_geometry(shadow: np.ndarray, crater: dict, az_deg: float) -> dict:
    """Shadow length and side for one crater, measured on the true shadow mask."""

    h, w = shadow.shape
    cx, cy, r = float(crater["center_x"]), float(crater["center_y"]), float(crater["radius_px"])
    reach = int(math.ceil(3.0 * r))
    x0, x1 = max(0, int(cx) - reach), min(w, int(cx) + reach + 1)
    y0, y1 = max(0, int(cy) - reach), min(h, int(cy) + reach + 1)
    sub = shadow[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Measure inside the crater circle only: shadow cast by the rim onto the
    # terrain OUTSIDE the crater is set by rim height, not by crater depth.
    sel = sub & (dist <= r)
    n = int(np.count_nonzero(sel))
    if n < 3:
        return {"n": n, "length_px": None, "mean_side": None}

    u = np.array([math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))])
    pts = np.stack([xx[sel] - cx, yy[sel] - cy], axis=1).astype(float)
    proj = pts @ u
    return {
        "n": n,
        "length_px": float(proj.max() - proj.min()),
        # >0 means the shadow sits on the side TOWARDS the sun, <0 anti-solar.
        "mean_side": float(np.mean(proj) / max(r, 1e-6)),
    }


def main() -> None:
    print("=== Check 1: shadow area vs sun elevation (seed 42, azimuth 35) ===")
    for elev in ELEVATIONS:
        syn = generate_synthetic_lunar_surface(size=512, seed=42, cast_shadows=True, sun_elevation_deg=elev)
        frac = float(np.count_nonzero(syn["shadow_mask"])) / syn["shadow_mask"].size
        lengths = [
            g["length_px"]
            for c in syn["craters"]
            if (g := crater_shadow_geometry(syn["shadow_mask"], c, AZ))["length_px"] is not None
        ]
        mean_len = float(np.mean(lengths)) if lengths else float("nan")
        print(f"  elevation {elev:4.1f} deg: shadow pixels {frac!r} of image, "
              f"mean per-crater shadow length {mean_len!r} px over {len(lengths)} craters")

    print("\n=== Check 2: which side of the crater the shadow falls on ===")
    print("  (projection of shadow pixels onto the direction TOWARDS the sun, in crater radii;")
    print("   negative = anti-solar side, positive = sun-facing side)")
    sides = []
    for seed in SEEDS:
        syn = generate_synthetic_lunar_surface(size=512, seed=seed, cast_shadows=True, sun_elevation_deg=35.0)
        for c in syn["craters"]:
            g = crater_shadow_geometry(syn["shadow_mask"], c, AZ)
            if g["mean_side"] is not None:
                sides.append(g["mean_side"])
    sides_arr = np.array(sides)
    print(f"  craters with shadow: {len(sides)}; mean side {float(sides_arr.mean())!r}; "
          f"min {float(sides_arr.min())!r}; max {float(sides_arr.max())!r}; "
          f"on anti-solar side: {int((sides_arr < 0).sum())}/{len(sides)}")

    print("\n=== Check 3: does shadow length track true depth? ===")
    variants = (
        ("shadow-free (legacy)", {}),
        ("ray-cast shadows, elevation 20 (generator default)", {"cast_shadows": True, "sun_elevation_deg": 20.0}),
        ("ray-cast shadows, elevation 35", {"cast_shadows": True, "sun_elevation_deg": 35.0}),
    )
    for label, kwargs in variants:
        depths, radii, lengths = [], [], []
        for seed in SEEDS:
            syn = generate_synthetic_lunar_surface(size=512, seed=seed, **kwargs)
            mask = syn["shadow_mask"]
            if mask is None:
                # Legacy scene has no shadow mask: measure the dark region instead,
                # i.e. pixels darker than the scene's own 20th percentile.
                img = syn["image"]
                mask = img < np.percentile(img, 20)
            for c in syn["craters"]:
                g = crater_shadow_geometry(mask, c, AZ)
                if g["length_px"] is None:
                    continue
                depths.append(c["true_depth"])
                radii.append(c["radius_px"])
                lengths.append(g["length_px"])
        print(f"  {label}: n={len(lengths)} "
              f"r(shadow length, true depth)={pearson(lengths, depths)!r} "
              f"r(shadow length, radius)={pearson(lengths, radii)!r} "
              f"r(true depth, radius)={pearson(depths, radii)!r}")


if __name__ == "__main__":
    main()
