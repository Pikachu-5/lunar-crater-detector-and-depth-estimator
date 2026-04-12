# Autonomous Planetary Landing Site Analyzer

Interactive 9-step Streamlit mission console for lunar landing-site analysis, with full visibility into preprocessing, crater detection, depth estimation, terrain reconstruction, safety scoring, and descent planning.

## What the app does

- Runs a guided 9-step analysis flow from image acquisition to final export.
- Uses trained YOLO crater weights from runs/crater_detector_SOTA/weights/best.pt.
- Supports separate manual detector execution in Step 4:
- Run YOLO Detection (primary downstream detector).
- Run CV Detection (Hough plus LoG comparator).
- Shows YOLO/CV speed comparison after both are run.
- Uses bottom-only navigation controls for cleaner progression.
- Uses resilient before/after comparison:
- Interactive swipe widget when streamlit-image-comparison works.
- Automatic side-by-side plus blend fallback when unavailable.
- Includes 3D terrain memory profiles with higher-capacity defaults:
- Balanced (256 MB)
- High Fidelity (512 MB) default
- Max Fidelity (1 GB)
- Adaptive

## Sidebar sliders explained

- Solar Incidence Angle theta: controls shadow-to-depth sensitivity.
- Depth Safety Threshold Td (m): max crater depth still treated as safe.
- Landing Gear Span (m): diameter feasibility threshold for touchdown stability.
- Crater Density Radius (px): neighborhood size used for local hazard density penalties.
- Pixel Scale (m/px): converts pixels to meters for depth, scoring, and path length.

## Pipeline steps

1. Mission briefing and upload/synthetic feed setup.
2. Raw image telemetry and histogram.
3. Preprocessing (CLAHE plus Gaussian) with parameter controls.
4. Manual crater detection (YOLO and/or CV), overlays, and speed comparison.
5. Photometric depth estimation from YOLO detections.
6. 2D and 3D terrain reconstruction with profile-based mesh sizing.
7. Landing safety scoring and hazard map generation.
8. A* descent path planning.
9. Mission report export (PDF, PNG, CSV).

## Project structure

- app.py
- modules/preprocess.py
- modules/detector.py
- modules/depth.py
- modules/terrain3d.py
- modules/scorer.py
- modules/pathfinder.py
- modules/reporter.py
- utils/synthetic.py
- utils/ui_components.py

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- For synthetic mode, crater metadata can be used to keep demo output deterministic.
- Depth, terrain, and scoring depend on YOLO detections. If YOLO has not been run, those steps will prompt you to run it first.
