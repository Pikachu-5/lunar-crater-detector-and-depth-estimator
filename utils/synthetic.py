"""Synthetic lunar surface generator for offline mission demos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class CraterSpec:
    """Parameter bundle describing one synthetic crater imprint.

    The crater is modeled as a radial depression with a narrow bright rim and
    directional illumination. This approximates how low-angle sunlight creates
    asymmetric brightness around circular impact structures on regolith.
    """

    crater_id: str
    cx: int
    cy: int
    radius: int
    depth_scale: float
    rim_scale: float


def _apply_crater_signature(
    height_map: np.ndarray,
    image_luma: np.ndarray,
    crater: CraterSpec,
    sun_angle_deg: float,
) -> None:
    """Imprint a crater depression and photometric highlight into arrays.

    Physics note:
    A bowl-shaped crater is approximated as a Gaussian depression in the local
    height field. A thin ring-shaped Gaussian is added to mimic uplifted rim.
    Brightness is modulated directionally to emulate sunlight incidence.

    Args:
        height_map: Continuous terrain elevation map updated in place.
        image_luma: Grayscale luminance raster updated in place.
        crater: Crater geometry and amplitude parameters.
        sun_angle_deg: Azimuth angle controlling illuminated rim direction.
    """

    h, w = height_map.shape
    x0 = max(0, crater.cx - crater.radius * 3)
    y0 = max(0, crater.cy - crater.radius * 3)
    x1 = min(w, crater.cx + crater.radius * 3)
    y1 = min(h, crater.cy + crater.radius * 3)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - crater.cx
    dy = yy - crater.cy
    dist = np.sqrt(dx * dx + dy * dy)

    bowl_sigma = crater.radius * 0.58
    rim_sigma = max(1.5, crater.radius * 0.13)

    bowl = -crater.depth_scale * np.exp(-(dist**2) / (2.0 * bowl_sigma * bowl_sigma))
    rim = crater.rim_scale * np.exp(-((dist - crater.radius) ** 2) / (2.0 * rim_sigma * rim_sigma))

    sun_rad = math.radians(sun_angle_deg)
    sun_vec = np.array([math.cos(sun_rad), math.sin(sun_rad)], dtype=np.float32)
    dist_safe = np.where(dist < 1.0, 1.0, dist)
    nx = dx / dist_safe
    ny = dy / dist_safe
    illum = np.clip(nx * sun_vec[0] + ny * sun_vec[1], -1.0, 1.0)

    directional_rim = rim * (0.55 + 0.45 * illum)
    shadow_term = bowl * (0.85 - 0.15 * illum)

    height_map[y0:y1, x0:x1] += bowl + 0.35 * rim
    image_luma[y0:y1, x0:x1] += shadow_term + directional_rim


def generate_synthetic_lunar_surface(
    size: int = 512,
    crater_count_range: tuple[int, int] = (8, 12),
    seed: int | None = 42,
    sun_angle_deg: float = 35.0,
) -> dict[str, Any]:
    """Generate a realistic grayscale lunar patch for full-pipeline demos.

    Computer vision note:
    Detection, photometric depth estimation, and hazard scoring all benefit from
    textures that resemble real orbital imagery. This function synthesizes
    regolith noise, crater depressions, and illumination cues to produce
    testable inputs without external data.

    Args:
        size: Width and height of the generated square image in pixels.
        crater_count_range: Inclusive minimum and maximum crater count.
        seed: Random seed for reproducible demos.
        sun_angle_deg: Sun azimuth used for directional highlights/shadows.

    Returns:
        Dictionary containing:
        - image: uint8 grayscale image of shape (size, size)
        - height_map: float32 relative elevation map
        - craters: list of crater descriptors
        - stats: mean/std/min/max intensity summary
    """

    rng = np.random.default_rng(seed)

    base = rng.normal(loc=120.0, scale=10.0, size=(size, size)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=1.8, sigmaY=1.8)

    height_map = np.zeros((size, size), dtype=np.float32)
    crater_count = int(rng.integers(crater_count_range[0], crater_count_range[1] + 1))

    craters: list[dict[str, Any]] = []
    for i in range(crater_count):
        radius = int(rng.integers(10, 61))
        cx = int(rng.integers(radius + 8, size - radius - 8))
        cy = int(rng.integers(radius + 8, size - radius - 8))

        crater = CraterSpec(
            crater_id=f"CR-{i+1:02d}",
            cx=cx,
            cy=cy,
            radius=radius,
            depth_scale=float(rng.uniform(25.0, 60.0)),
            rim_scale=float(rng.uniform(8.0, 18.0)),
        )

        _apply_crater_signature(height_map, base, crater, sun_angle_deg=sun_angle_deg)

        craters.append(
            {
                "crater_id": crater.crater_id,
                "center_x": crater.cx,
                "center_y": crater.cy,
                "radius_px": crater.radius,
                "sun_angle_deg": sun_angle_deg,
            }
        )

    texture = rng.normal(0.0, 5.5, size=(size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=0.9, sigmaY=0.9)

    image = base + texture
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)

    stats = {
        "mean": float(np.mean(image_u8)),
        "std": float(np.std(image_u8)),
        "min": int(np.min(image_u8)),
        "max": int(np.max(image_u8)),
    }

    return {
        "image": image_u8,
        "height_map": height_map,
        "craters": craters,
        "stats": stats,
    }


def generate_secondary_solar_view(
    image: np.ndarray,
    angle_shift_deg: float = 18.0,
) -> np.ndarray:
    """Create a second synthetic-looking view with shifted illumination.

    Physics note:
    Photometric methods infer depth from shadow length under known solar angle.
    A second image with a different sun direction provides complementary shadow
    geometry, reducing uncertainty when depth estimates are fused.

    Args:
        image: Base grayscale lunar image.
        angle_shift_deg: Magnitude of illumination direction change.

    Returns:
        uint8 grayscale image with altered directional shading.
    """

    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2.0
    cy = h / 2.0

    theta = math.radians(angle_shift_deg)
    grad = ((xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)) / max(w, h)
    grad = 18.0 * grad

    shifted = image.astype(np.float32) + grad.astype(np.float32)
    shifted = cv2.GaussianBlur(shifted, (0, 0), sigmaX=0.7, sigmaY=0.7)

    return np.clip(shifted, 0, 255).astype(np.uint8)
