"""A* descent path planning over crater risk terrain."""

from __future__ import annotations

import heapq
import math
from typing import Any

import cv2
import numpy as np


def _downsample_score_map(score_map: np.ndarray, target_max_dim: int = 180) -> tuple[np.ndarray, float, float]:
    """Downsample score map for interactive A* planning latency.

    Path-planning note:
    A* complexity grows with node count. Downsampling preserves broad hazard
    structure while reducing computational cost for responsive UI updates.

    Args:
        score_map: Full-resolution score map in [0, 100].
        target_max_dim: Maximum width/height of planning grid.

    Returns:
        Tuple of (downsampled score map, sx, sy) where sx/sy map grid coords to
        original pixel coordinates.
    """

    h, w = score_map.shape
    scale = max(h, w) / float(target_max_dim)
    if scale <= 1.0:
        return score_map.copy(), 1.0, 1.0

    nw = max(32, int(round(w / scale)))
    nh = max(32, int(round(h / scale)))
    ds = cv2.resize(score_map, (nw, nh), interpolation=cv2.INTER_AREA)
    sx = w / float(nw)
    sy = h / float(nh)
    return ds.astype(np.float32), sx, sy


def _score_to_cost(score_map: np.ndarray) -> np.ndarray:
    """Convert safety scores into traversal costs for A*.

    Path-planning note:
    High safety scores should be cheap to traverse, while hazardous regions are
    assigned very large costs to discourage route passage.

    Args:
        score_map: Score map in [0, 100].

    Returns:
        Cost map for weighted grid search.
    """

    cost = 1.0 + (100.0 - np.clip(score_map, 0, 100)) / 18.0
    cost = cost.astype(np.float32)
    cost[score_map < 25.0] += 80.0
    return cost


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean heuristic for A* on 2D grids."""

    return math.hypot(a[0] - b[0], a[1] - b[1])


def _neighbors(x: int, y: int, h: int, w: int) -> list[tuple[int, int, float]]:
    """Enumerate 8-connected neighbors with movement distance weights."""

    out: list[tuple[int, int, float]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                step = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
                out.append((nx, ny, step))
    return out


def astar(cost_map: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Run weighted A* search on a 2D cost field.

    Args:
        cost_map: Traversal cost field.
        start: Start coordinate (x, y) on grid.
        goal: Goal coordinate (x, y) on grid.

    Returns:
        Ordered path coordinates from start to goal, or empty list if none.
    """

    h, w = cost_map.shape
    sx, sy = start
    gx, gy = goal

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, (sx, sy)))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {(sx, sy): 0.0}

    max_iters = h * w * 4
    iters = 0

    while open_heap and iters < max_iters:
        iters += 1
        _, current = heapq.heappop(open_heap)
        if current == (gx, gy):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        base = g_score[current]

        for nx, ny, step in _neighbors(cx, cy, h, w):
            tentative = base + step * float(cost_map[ny, nx])
            node = (nx, ny)
            if tentative < g_score.get(node, float("inf")):
                came_from[node] = current
                g_score[node] = tentative
                f = tentative + _heuristic(node, (gx, gy))
                heapq.heappush(open_heap, (f, node))

    return []


def _path_length(path: list[tuple[int, int]]) -> float:
    """Compute polyline length in grid units."""

    if len(path) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        dist += math.hypot(dx, dy)
    return dist


def _pick_goal(scored_rows: list[dict[str, Any]], image_shape: tuple[int, int]) -> tuple[int, int, float]:
    """Choose target landing zone from highest-scoring safe/caution crater.

    Strategy:
    Prefer SAFE zones first, otherwise CAUTION, otherwise highest score overall.

    Args:
        scored_rows: Crater rows with safety score and center coordinates.
        image_shape: Scene shape for fallback center target.

    Returns:
        (goal_x, goal_y, goal_score)
    """

    if not scored_rows:
        h, w = image_shape
        return w // 2, int(h * 0.85), 50.0

    priorities = ["SAFE", "CAUTION", "HAZARD"]
    for z in priorities:
        cand = [r for r in scored_rows if r["zone"] == z]
        if cand:
            best = max(cand, key=lambda r: float(r["safety_score"]))
            return int(best["center_x"]), int(best["center_y"]), float(best["safety_score"])

    best = max(scored_rows, key=lambda r: float(r["safety_score"]))
    return int(best["center_x"]), int(best["center_y"]), float(best["safety_score"])


def plan_descent_paths(
    score_map: np.ndarray,
    scored_rows: list[dict[str, Any]],
    pixel_scale_m: float = 1.0,
) -> dict[str, Any]:
    """Plan optimal and alternative descent trajectories using A*.

    Path-planning note:
    The route starts at top-center to emulate initial descent alignment and aims
    for the best-scoring candidate zone while avoiding low-score hazard regions
    represented as high traversal costs.

    Args:
        score_map: Terrain safety map in [0, 100].
        scored_rows: Crater safety rows.
        pixel_scale_m: Meters per original image pixel.

    Returns:
        Dictionary with primary path, alternatives, target coordinates, and
        mission metrics such as length and confidence.
    """

    h, w = score_map.shape
    ds_score, sx, sy = _downsample_score_map(score_map)
    cost = _score_to_cost(ds_score)

    start_orig = (w // 2, 4)
    goal_x, goal_y, goal_score = _pick_goal(scored_rows, image_shape=(h, w))

    start = (int(start_orig[0] / sx), int(start_orig[1] / sy))
    goal = (int(goal_x / sx), int(goal_y / sy))

    start = (np.clip(start[0], 0, cost.shape[1] - 1), np.clip(start[1], 0, cost.shape[0] - 1))
    goal = (np.clip(goal[0], 0, cost.shape[1] - 1), np.clip(goal[1], 0, cost.shape[0] - 1))

    primary_ds = astar(cost, start=start, goal=goal)

    def back_to_full(path_ds: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out = []
        for x, y in path_ds:
            out.append((int(round(x * sx)), int(round(y * sy))))
        return out

    primary = back_to_full(primary_ds)

    alt_paths: list[list[tuple[int, int]]] = []
    if primary_ds:
        alt_offsets = [(-8, 6), (7, 10), (-10, 14)]
        for ox, oy in alt_offsets:
            g2 = (
                int(np.clip(goal[0] + ox, 0, cost.shape[1] - 1)),
                int(np.clip(goal[1] + oy, 0, cost.shape[0] - 1)),
            )
            p2 = astar(cost, start=start, goal=g2)
            if len(p2) > 0:
                alt_paths.append(back_to_full(p2))

    length_px = _path_length(primary)
    length_m = float(length_px * pixel_scale_m)

    hazards_avoided = sum(1 for row in scored_rows if row["zone"] == "HAZARD")
    confidence = float(np.clip(goal_score + 0.2 * (100.0 - min(100.0, hazards_avoided * 8.0)), 40.0, 99.0))

    return {
        "start": start_orig,
        "goal": (goal_x, goal_y),
        "primary_path": primary,
        "alternative_paths": alt_paths,
        "path_length_m": round(length_m, 2),
        "obstacles_avoided": int(hazards_avoided),
        "landing_confidence": round(confidence, 2),
    }


def draw_paths_on_map(
    image: np.ndarray,
    primary_path: list[tuple[int, int]],
    alternative_paths: list[list[tuple[int, int]]],
) -> np.ndarray:
    """Overlay planned descent trajectories on hazard map imagery.

    Args:
        image: Base RGB or grayscale image.
        primary_path: Main path coordinates in pixels.
        alternative_paths: Alternative path sets.

    Returns:
        RGB image with path overlays.
    """

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()

    for path in alternative_paths:
        if len(path) >= 2:
            pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, (115, 181, 255), 1, cv2.LINE_AA)

    if len(primary_path) >= 2:
        pts = np.array(primary_path, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, (40, 140, 255), 2, cv2.LINE_AA)

    return canvas
