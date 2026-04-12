# Project Explained: Autonomous Planetary Landing Site Analyzer

This file describes the current implementation in app.py and modules/, including UI behavior, detector controls, slider meaning, and 3D memory profiles.

## 1) Repository map and responsibilities

### Root

- app.py
- Streamlit mission workflow controller.
- Session state lifecycle, step routing, sidebar controls, and navigation.
- Integrates all processing modules and UI widgets.

- README.md
- High-level overview, setup, and behavior summary.

- requirements.txt
- Runtime dependencies including streamlit, ultralytics, plotly, opencv, and reportlab.

### modules/

- modules/preprocess.py
- CLAHE enhancement, Gaussian smoothing, histogram and image stats.

- modules/detector.py
- Primary detector: YOLO inference using trained crater weights.
- Comparison detector: CV hybrid (Hough circles plus LoG blobs).
- Overlay drawing and normalized crater record creation.
- Default model path is runs/crater_detector_SOTA/weights/best.pt.

- modules/depth.py
- Photometric depth estimation from crater ROI shadows.
- Otsu shadow mask, shadow-length extraction, depth and slope estimates.
- Multi-angle fusion utility with IoU correspondence.

- modules/terrain3d.py
- Builds a depth map from crater rows.
- Creates heatmap, contour, and interactive 3D surface figures.
- 3D downsample logic supports larger targets than legacy 200-size rendering.

- modules/scorer.py
- Safety scoring for each crater.
- Zone labeling and score map generation.
- Hazard overlay generation and safety gauge support.

- modules/pathfinder.py
- A* descent path planning on the score-cost map.
- Returns primary and alternative routes with path metrics.

- modules/reporter.py
- PDF report generation.
- PNG export for hazard map and CSV export for crater table.

### utils/

- utils/synthetic.py
- Synthetic lunar scene generation and metadata.
- Secondary-angle synthetic view generator for fusion demos.

- utils/ui_components.py
- Shared visual system and mission widgets.
- HUD rendering, pipeline strip, metric cards, formula blocks, detection badges, and terminal panel.

## 2) Pipeline behavior by step

The app uses TOTAL_STEPS = 9 and persists intermediate outputs in st.session_state.

1. Mission Briefing
- Optional upload or synthetic default feed.
- Initializes mission context and launch action.

2. Raw Image Acquisition
- Raw feed display.
- Image stats table and intensity histogram.

3. Preprocessing and Enhancement
- CLAHE plus Gaussian smoothing.
- Side-by-side raw/enhanced/smoothed previews with histograms.
- Before/after comparison with robust fallback path.

4. Crater Detection (Manual)
- No auto-run on entry.
- Two explicit controls:
- Run YOLO Detection (primary downstream source).
- Run CV Detection (comparison source).
- Displays crater counts, elapsed times, overlays, and speed comparison delta/ratio when both runs exist.

5. Photometric Depth Estimation
- Requires YOLO detections.
- If YOLO has not run, the UI prompts the operator to run YOLO first.
- Computes per-crater depth and slope, with ROI diagnostics.

6. 3D Terrain Reconstruction
- Builds depth map from crater depth rows.
- Renders heatmap, contour, and interactive 3D surface.
- Uses profile-based 3D mesh target sizing and memory estimate display.

7. Landing Safety Scoring
- Scores each crater using depth, diameter feasibility, and local density.
- Produces SAFE/CAUTION/HAZARD summary and hazard map.

8. Descent Path Planning
- Runs A* planning on score-derived traversal costs.
- Produces primary and alternative routes and recommended landing coordinates.

9. Mission Report Export
- Final crater table and hazard map.
- Export package: PDF plus PNG plus CSV.

## 3) Sidebar mission controls and why they matter

- Solar Incidence Angle theta
- Changes shadow geometry used in depth estimation.

- Depth Safety Threshold Td (m)
- Sets safety strictness for depth-related hazard penalties.

- Landing Gear Span (m)
- Controls diameter feasibility relative to lander footprint.

- Crater Density Radius (px)
- Controls neighborhood radius used in local roughness risk.

- Pixel Scale (m/px)
- Converts pixel distances to metric units used across modules.

- 3D Terrain Memory Profile
- Balanced (256 MB)
- High Fidelity (512 MB) default
- Max Fidelity (1 GB)
- Adaptive
- Influences surface mesh downsample target and render memory usage.

- Enable Multi-Angle Fusion
- Allows optional second-view depth fusion in Step 5.

## 4) Detector behavior details

- YOLO path
- Implemented in modules/detector.py via detect_craters().
- Uses trained crater model at runs/crater_detector_SOTA/weights/best.pt.
- Output stored as st.session_state.detection.
- This is the downstream source for depth/terrain/scoring/path/report.

- CV path
- Implemented via detect_craters_cv() in modules/detector.py.
- Uses Hough plus LoG crater candidates and quality filtering.
- Output stored as st.session_state.cv_detection.
- Used for visual and speed comparison in Step 4.

## 5) Before/after comparison behavior

- Primary mode
- Uses streamlit-image-comparison interactive swipe when available.

- Fallback mode
- If unavailable or runtime-failing, app switches to:
- side-by-side RAW/SMOOTHED view
- blend slider for continuous visual interpolation
- explicit operator notice in UI and telemetry log

## 6) Navigation behavior

- The app uses bottom-only navigation arrows for all steps.
- The previous duplicated top-and-bottom navigation layout has been removed.

## 7) Blur operations currently used

### Operational pipeline blurs

1. Preprocess smoothing
- modules/preprocess.py
- Gaussian blur with configurable sigma (sidebar/step controls).

2. CV detector pre-smoothing
- modules/detector.py
- Gaussian blur before Hough candidate generation.

3. Depth ROI smoothing before Otsu thresholding
- modules/depth.py
- Gaussian blur to stabilize shadow segmentation.

4. Terrain depth-map smoothing
- modules/terrain3d.py
- Gaussian blur after crater depression aggregation.

### Synthetic generation blurs

5. Base synthetic terrain smoothing
- utils/synthetic.py

6. Synthetic texture smoothing
- utils/synthetic.py

7. Secondary view smoothing
- utils/synthetic.py

## 8) Notes on large files and memory

- Large uploads can increase reconstruction and rendering cost.
- 3D surface rendering now uses profile-based target sizing and can exceed the old 200-size mesh behavior.
- High Fidelity (512 MB) is the default profile and is intended for better detail retention on larger scenes.
