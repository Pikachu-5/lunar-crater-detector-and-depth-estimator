"""Image enhancement utilities for lunar crater analysis."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.2,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast using CLAHE.

    Computer vision note:
    CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local
    contrast while limiting noise amplification. Lunar surfaces have subtle
    intensity gradients near crater rims and shadows; CLAHE helps preserve those
    local cues for subsequent edge/circle detection and ROI segmentation.

    Mathematical formulation:
    1. Image is divided into non-overlapping tiles of size `tile_grid_size`.
    2. For each tile, a local histogram is computed and clipped at `clip_limit`
       (normalized to tile area). Excess counts above the clip limit are
       redistributed uniformly across all bins.
    3. A cumulative distribution function (CDF) is computed per tile:
       CDF(i) = sum_{j=0}^{i} clipped_hist(j) / total_pixels_in_tile
    4. Bilinear interpolation between neighboring tile CDFs eliminates
       block boundary artifacts, yielding smooth local contrast enhancement.

    Args:
        image: Input grayscale image.
        clip_limit: Contrast limiting threshold.
        tile_grid_size: Tile size for local histogram equalization.

    Returns:
        CLAHE-enhanced grayscale image.
    """

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_gaussian_smoothing(image: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Apply Gaussian smoothing to suppress high-frequency sensor noise.

    Computer vision note:
    Crater boundaries are low-to-mid frequency structures. Gaussian filtering
    attenuates high-frequency noise that can trigger false detections while
    keeping macro-geometry mostly intact.

    Mathematical formulation:
    G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

    The kernel size is automatically derived from sigma by OpenCV:
    ksize = round(sigma * 4.5) | 1  (rounded up to nearest odd integer)

    Convolution: smoothed(x,y) = sum over (u,v) of image(x-u, y-v) * G(u, v)

    Args:
        image: Input grayscale image.
        sigma: Gaussian kernel standard deviation in pixels.

    Returns:
        Smoothed grayscale image.
    """

    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def preprocess_pipeline(
    image: np.ndarray,
    clip_limit: float = 2.2,
    tile_grid_size: tuple[int, int] = (8, 8),
    sigma: float = 1.2,
) -> dict[str, Any]:
    """Run mission preprocessing chain and return all intermediates.

    Args:
        image: Raw grayscale lunar image.
        clip_limit: CLAHE contrast limit.
        tile_grid_size: CLAHE tile grid dimensions.
        sigma: Gaussian blur sigma.

    Returns:
        Dictionary with raw, enhanced, and smoothed image variants.
    """

    enhanced = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    smoothed = apply_gaussian_smoothing(enhanced, sigma=sigma)
    return {
        "raw": image,
        "enhanced": enhanced,
        "smoothed": smoothed,
        "params": {
            "clip_limit": clip_limit,
            "tile_grid_size": tile_grid_size,
            "sigma": sigma,
        },
    }


def compute_histogram(image: np.ndarray, bins: int = 64) -> dict[str, np.ndarray]:
    """Compute normalized grayscale histogram.

    Computer vision note:
    Histograms summarize global brightness distribution. During preprocessing,
    histogram spreading indicates improved contrast and often better separability
    between shadowed and illuminated terrain.

    Args:
        image: Grayscale image.
        bins: Number of histogram bins.

    Returns:
        Dictionary with bin edges, centers, and normalized counts.
    """

    counts, edges = np.histogram(image.flatten(), bins=bins, range=(0, 255))
    counts = counts.astype(np.float64)
    if np.sum(counts) > 0:
        counts /= np.sum(counts)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "counts": counts,
        "edges": edges,
        "centers": centers,
    }


def image_stats(image: np.ndarray, file_size_bytes: int | None = None) -> dict[str, Any]:
    """Compute instrument-style descriptive statistics for telemetry display.

    Args:
        image: Grayscale image.
        file_size_bytes: Optional file size from upload metadata.

    Returns:
        Dictionary with dimensions, file size, mean, standard deviation, and
        dynamic range metrics.
    """

    h, w = image.shape
    return {
        "width_px": int(w),
        "height_px": int(h),
        "file_size_kb": None if file_size_bytes is None else round(file_size_bytes / 1024.0, 2),
        "mean_intensity": round(float(np.mean(image)), 3),
        "std_intensity": round(float(np.std(image)), 3),
        "min_intensity": int(np.min(image)),
        "max_intensity": int(np.max(image)),
    }
