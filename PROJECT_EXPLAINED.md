# Autonomous Planetary Landing Site Analyzer - Full Technical Explanation

This document is a technical walkthrough of what the project computes, how data moves between modules, what arrays are created, what each slider changes, and how cache invalidation works.

## 1) System Purpose

The application ingests a lunar image, enhances contrast, detects crater candidates, estimates crater depth using photometric geometry, reconstructs a terrain depth field, scores landing safety, plans a descent path, and exports mission artifacts.

The pipeline is implemented as a 9-step Streamlit workflow in app.py and functional modules under modules/.

## 2) Core Data Model

At runtime, the main data object is a grayscale image array:

- Type: numpy.ndarray
- Shape: (H, W)
- Typical dtype: uint8
- Value range: 0..255

Major downstream arrays include:

- Enhanced image (CLAHE): uint8, shape (H, W)
- Smoothed image (Gaussian): uint8, shape (H, W)
- Shadow mask per crater ROI: uint8 binary mask with values {0, 255}
- Depth map: float32, shape (H, W)
- Safety score map: float32, shape (H, W), range 0..100
- Path cost map: float32, shape (h_ds, w_ds)

## 3) Session State and Pipeline Continuity

The app relies on st.session_state to persist outputs across Streamlit reruns.

Key state keys:

- raw_image
- preprocess
- detection
- cv_detection
- depth
- terrain
- scoring
- paths
- hazard_map

Important cache signatures:

- depth_signature: prevents stale depth reuse when theta, azimuth, pixel scale, or detections change.
- last_depth_slider_signature: tracks depth-related slider values.
- last_score_slider_signature: tracks scoring-related slider values.

When an upstream value changes, reset_downstream(start_step=...) clears all downstream artifacts to force recomputation.

## 4) Image Upload, Byte Decoding, and Resolution Mode

Primary upload is handled in step 1. The app supports:

- png, jpg, jpeg, tif, tiff

### 4.1 Byte decoding

Image bytes are decoded with:

- data = np.frombuffer(file_bytes, dtype=np.uint8)
- img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

### 4.2 Keep Original vs Auto-Resize toggle

The sidebar toggle Keep Original Upload Resolution controls whether large uploads are resized.

- Keep Original ON: decode and preserve full upload resolution.
- Keep Original OFF: if pixel count exceeds threshold, image is resized for stability.

Current resize threshold logic:

- max_pixels = 14,000,000
- if H * W > max_pixels:
  - scale = sqrt(max_pixels / (H * W))
  - new_w = round(W * scale)
  - new_h = round(H * scale)
  - cv2.resize(..., interpolation=cv2.INTER_AREA)

The app caches uploaded bytes and filename, then re-decodes from cache when this toggle changes, followed by downstream reset from step 3.

## 5) Step-by-Step Technical Computation

## Step 1: Mission briefing and image source

- Either uses synthetic image generator or uploaded file.
- On upload change, downstream artifacts are invalidated from preprocessing onward.

## Step 2: Raw acquisition and instrumentation

Computes telemetry-style metrics:

- width and height
- file size in KB
- mean and standard deviation of intensity
- min and max intensity

Histogram:

- np.histogram(image.flatten(), bins=64, range=(0, 255))
- normalized by total count for probability-like visualization.

## Step 3: Preprocessing and enhancement

The preprocess pipeline runs:

1. CLAHE
2. Gaussian smoothing

### 5.3.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE is local contrast enhancement per tile.

Parameters:

- clip_limit
- tile_grid_size

Computation concept:

- Partition image into tiles.
- For each tile, compute histogram h(k).
- Clip bins above limit and redistribute excess uniformly.
- Compute local CDF for intensity remapping.
- Blend neighboring tile mappings by bilinear interpolation.

Why this helps:

- Lunar rims and shadow boundaries are local features; global equalization is weaker.
- CLAHE boosts local separability for both YOLO/CV detection and shadow segmentation.

### 5.3.2 Gaussian blur

Gaussian smoothing suppresses high-frequency noise using kernel:

- G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

OpenCV computes a kernel based on sigma and performs convolution.

Why this helps:

- Stabilizes edge/circle extraction and reduces noisy false positives.

### 5.3.3 Before/after blend math

Fallback comparison mode uses per-pixel weighted blend:

- blended = cv2.addWeighted(before, 1 - alpha, after, alpha, 0)

This is direct pixel multiplication and accumulation:

- blended[i, j] = (1 - alpha) * before[i, j] + alpha * after[i, j]

## Step 4: Crater detection (manual YOLO and CV)

Two independent buttons run two detector families.

### 5.4.1 YOLO path

- Uses ultralytics YOLO model loaded from runs/crater_detector_SOTA/weights/best.pt
- Inference settings include imgsz=416, CPU execution, confidence threshold.
- Produces xyxy boxes and confidence arrays.
- Boxes are converted into normalized crater records with center, radius, diameter.

### 5.4.2 CV hybrid path

- Hough circles at multiple scales
- LoG blob detection
- Candidate merging and filtering

If image is large, CV path downsamples to max_detection_dim and later rescales detections back to full image coordinates.

Speed comparison section computes:

- elapsed time delta
- time ratio
- faster method label

## Step 5: Photometric depth estimation

For each detected crater bounding box:

1. Extract ROI (array slicing)
2. Segment shadow using Otsu on blurred ROI
3. Build anti-sun projection axis from azimuth
4. Project shadow pixels onto axis
5. Shadow length = max projection - min projection
6. Convert shadow length to depth

### 5.5.1 Otsu shadow mask

- Blur ROI slightly
- Otsu threshold chooses split value automatically
- mask = 255 where pixel is shadow class else 0
- Morphological open/close cleans mask noise

### 5.5.2 Shadow projection array math

Given shadow pixel coordinates:

- xs, ys = np.where(mask > 0)
- pts = np.stack([xs, ys], axis=1)

Azimuth vector:

- anti = radians((azimuth + 180) % 360)
- v = [cos(anti), sin(anti)]

Projection:

- proj = pts @ v

Shadow length in pixels:

- L = max(proj) - min(proj)

### 5.5.3 Depth formula

- depth_m = shadow_length_px * pixel_scale_m / tan(theta_incidence)

Where:

- theta is solar incidence angle from surface normal (depth scale control)
- azimuth controls arrow direction and projection axis

Also computes slope:

- radius_m = 0.5 * diameter_px * pixel_scale_m
- slope_deg = degrees(arctan2(depth_m, radius_m))

## Step 6: Terrain reconstruction and 3D rendering

The terrain model builds a continuous depth map from crater rows.

For each crater:

- Center (cx, cy), radius r, depth d
- sigma = r * 0.62
- Evaluate local Gaussian crater bowl on a bounded patch

Per-pixel crater contribution:

- crater(y, x) = d * exp(-dist_sq / (2 * sigma^2))

Accumulation:

- depth_map[y0:y1, x0:x1] += crater_patch

Then apply light Gaussian smoothing to depth_map.

### 5.6.1 3D downsample profile

The UI profile maps to target mesh size. Higher profile means larger render map and higher memory use.

The surface rendering uses downsampled depth map for WebGL smoothness and stability.

## Step 7: Landing safety scoring

Each crater receives a score from depth, diameter, and local density.

Current scoring uses continuous penalties:

- depth_scale = depth_threshold_m * 5
- depth_penalty = 45 * depth_m / (depth_m + depth_scale)

- gear_scale = landing_gear_span_m * 3
- diameter_penalty = 30 * diameter_m / (diameter_m + gear_scale)

- density_penalty = min(25, neighbors * 4)

Final score:

- score = clip(100 - depth_penalty - diameter_penalty - density_penalty, 0, 100)

Zone mapping:

- SAFE if score >= 70
- CAUTION if 40 <= score < 70
- HAZARD if score < 40

This stage also rasterizes crater influence into a dense score map.

### 5.7.1 Score map pixel blending math

For each crater influence field:

- influence = exp(-dist^2 / (2 * (radius * 1.15)^2))
- local = score_map * (1 - influence) + crater_score * influence
- score_map = minimum(score_map, local)

This is another key pixel multiplication and weighted blending operation.

## Step 8: Path planning (A*)

Path planning runs on a downsampled score map for speed.

### 5.8.1 Score to cost

- base cost = 1 + (100 - score) / 18
- extra hazard penalty added for very low score regions

### 5.8.2 A* search

- Graph: 8-connected grid
- g(n): cumulative traversal cost
- h(n): Euclidean heuristic to goal
- f(n) = g(n) + h(n)

Outputs:

- primary path
- several offset-goal alternative paths
- path length in meters: path_length_px * pixel_scale_m

## Step 9: Report export

Exports:

- PDF (ReportLab)
- PNG hazard map
- CSV crater table

The report includes score summary and recommended landing coordinates.

## 6) Slider-to-Computation Mapping

- Solar Incidence Angle theta
- Affects 1/tan(theta) term in depth formula.
- Impacts depth, slope, terrain map, scoring, path.

- Solar Azimuth phi
- Affects projection axis for shadow length measurement.
- Changes shadow arrow direction and can alter measured L depending on ROI shape.

- Pixel Scale (m/px)
- Converts pixel lengths to meters in depth and path length.
- Also affects crater diameter in meters for scoring.

- Depth Safety Threshold Td
- Affects depth penalty curve in scoring.

- Landing Gear Span
- Affects diameter penalty curve in scoring.

- Crater Density Radius
- Affects neighbor counts and density penalty in scoring.

- 3D Terrain Memory Profile
- Affects 3D surface target mesh size and rendering memory footprint.

- Keep Original Upload Resolution
- Controls whether very large uploads are resized before the full pipeline.
- Triggers downstream recomputation when changed.

## 7) Cache Invalidation and Recompute Rules

The app intentionally invalidates downstream data when upstream inputs change.

Examples:

- New upload or synthetic regeneration -> reset from step 3.
- Preprocess parameter change -> reset from step 4.
- Depth-related slider change -> reset from step 5.
- Scoring-related slider change -> reset from step 7.
- Upload resolution mode toggle -> re-decode cached upload bytes and reset from step 3.

This prevents stale visuals and stale metrics.

## 8) Memory and Performance Controls

- Streamlit server upload/message limits are configured in .streamlit/config.toml.
- Large-upload auto-resize protects memory and interaction latency.
- CV detector includes max dimension control before detection.
- Path planning downsamples score map for fast A*.
- 3D view uses profile-based downsample target.

## 9) Practical Presentation Talking Points

- The pipeline is transparent: every major stage exposes intermediate arrays and plots.
- Crater depth is physically grounded in shadow geometry, not a black-box depth network.
- Scoring and path planning are deterministic and auditable.
- Slider changes are wired to cache invalidation so outputs stay consistent with current controls.
