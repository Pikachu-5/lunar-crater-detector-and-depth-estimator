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
    Brightness is modulated by sun azimuth only; there is no elevation term and
    no cast shadow.

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


def cast_shadow_mask(
    height_map: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    relief_px_per_unit: float = 0.25,
    max_steps: int | None = None,
) -> np.ndarray:
    """Ray-march the height field to find pixels the sun cannot reach.

    Physics note:
    This is an occlusion test, not an analytic shadow length. For every pixel a
    ray is marched towards the sun along the solar azimuth; the pixel is
    shadowed if terrain anywhere along that ray rises above the straight line
    leaving the pixel at the sun's elevation angle. Shadow geometry therefore
    emerges from the terrain itself (bowl, rim ring, neighbouring craters), and
    recovering depth from the result is a genuine inverse problem.

    Args:
        height_map: Elevation field in height units.
        sun_azimuth_deg: Direction TOWARDS the sun in image coordinates.
        sun_elevation_deg: Sun elevation above the horizontal, degrees.
        relief_px_per_unit: Vertical scale — pixels of height per height unit.
        max_steps: Maximum ray length in pixels (derived from the relief when None).

    Returns:
        Boolean mask, True where the sun is occluded.
    """

    h, w = height_map.shape
    elevation = float(np.clip(sun_elevation_deg, 0.5, 89.5))
    tan_e = math.tan(math.radians(elevation))
    azimuth = math.radians(sun_azimuth_deg)
    ux, uy = math.cos(azimuth), math.sin(azimuth)

    height_px = height_map.astype(np.float32) * float(relief_px_per_unit)
    relief = float(height_px.max() - height_px.min())
    if max_steps is None:
        # Beyond this distance nothing can still be occluding: even the tallest
        # relief in the scene no longer reaches the sun ray.
        max_steps = int(min(math.hypot(h, w), math.ceil(relief / max(tan_e, 1e-6)) + 2))
    max_steps = max(1, int(max_steps))

    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    shadow = np.zeros((h, w), dtype=bool)

    for step in range(1, max_steps + 1):
        map_x = xx + step * ux
        map_y = yy + step * uy
        sampled = cv2.remap(
            height_px,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1e9,
        )
        # Terrain at distance `step` blocks the sun when it rises above the ray.
        shadow |= sampled > (height_px + step * tan_e)

    return shadow


def generate_synthetic_lunar_surface(
    size: int = 512,
    crater_count_range: tuple[int, int] = (8, 12),
    seed: int | None = 42,
    sun_angle_deg: float = 35.0,
    cast_shadows: bool = False,
    sun_elevation_deg: float = 20.0,
    relief_px_per_unit: float = 1.0,
    shadow_darkness: float = 0.35,
    depth_to_radius_range: tuple[float, float] = (0.35, 0.75),
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
        sun_angle_deg: Sun azimuth (direction towards the sun) used for
            directional highlights and, when enabled, for shadow ray-casting.
        cast_shadows: When True, add geometric cast shadows computed by
            occlusion over the height field (see ``cast_shadow_mask``). When
            False (default) the legacy shadow-free scene is produced, byte for
            byte, so earlier audit results stay reproducible.
        sun_elevation_deg: Sun elevation above the horizontal for the ray-cast,
            used only when ``cast_shadows`` is True.
        relief_px_per_unit: Vertical scale for the ray-cast, in pixels of
            height per height-map unit.
        shadow_darkness: Brightness multiplier applied to occluded pixels.
        depth_to_radius_range: In ``cast_shadows`` mode the bowl depth is drawn
            as radius * U(range) instead of the legacy radius-independent
            U(25, 60). A Gaussian bowl only self-shadows when its wall slope
            exceeds the sun elevation (depth/radius > tan(elevation)/1.046) and
            the shadow only stays inside the crater when depth/radius <
            2*tan(elevation); with the legacy draw most craters fall outside
            that window, so no shadow forms at all. The ratio still varies
            independently of radius, so depth is not a function of size.

    Returns:
        Dictionary containing:
        - image: uint8 grayscale image of shape (size, size)
        - height_map: float32 relative elevation map
        - craters: list of crater descriptors. Each includes ``true_depth``
          (amplitude of the Gaussian bowl written into ``height_map``) and
          ``rim_height`` (rim amplitude). Both are in height_map units; the
          generator defines no metric scale, so they are not metres.
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

        if cast_shadows:
            depth_scale = float(radius * rng.uniform(*depth_to_radius_range))
        else:
            depth_scale = float(rng.uniform(25.0, 60.0))

        crater = CraterSpec(
            crater_id=f"CR-{i+1:02d}",
            cx=cx,
            cy=cy,
            radius=radius,
            depth_scale=depth_scale,
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
                "true_depth": crater.depth_scale,
                "rim_height": crater.rim_scale,
            }
        )

    shadow_mask = None
    if cast_shadows:
        # Geometric cast shadows from the height field. Bowl shading and the
        # texture noise below are kept; only occluded pixels are darkened.
        shadow_mask = cast_shadow_mask(
            height_map,
            sun_azimuth_deg=sun_angle_deg,
            sun_elevation_deg=sun_elevation_deg,
            relief_px_per_unit=relief_px_per_unit,
        )
        base = base.copy()
        base[shadow_mask] *= float(shadow_darkness)

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
        "shadow_mask": shadow_mask,
        "shadow_params": None
        if not cast_shadows
        else {
            "sun_azimuth_deg": float(sun_angle_deg),
            "sun_elevation_deg": float(sun_elevation_deg),
            "relief_px_per_unit": float(relief_px_per_unit),
            "shadow_darkness": float(shadow_darkness),
        },
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
