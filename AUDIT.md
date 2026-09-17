# AUDIT — Lunar Crater Detector + Depth Estimator

Audit date: 2026-09-17. Repository state: branch `main`, HEAD `20bd8d9` ("updated project documentation"), clean working tree before this audit (the audit adds only `AUDIT.md`, `scripts/audit_run.py`, `scripts/audit_results.json`, `scripts/audit_tables.md`).

Conventions used here:
- `file:line` refers to the file as of HEAD `20bd8d9`.
- Where docs and code disagree, **the code wins**; both are quoted.
- **PLACEHOLDER** = a literal in source shown to the user as if it were measured.
- Numbers from runs are copied verbatim from `scripts/audit_results.json` / `scripts/audit_tables.md` (Python `repr`, full precision). Values the project itself rounds (e.g. `depth.py:164-169` rounds table columns to 3 dp) are reported as the project emits them.

Environment for all runs: Python 3.12.10, Windows-11-10.0.26200, 13th Gen Intel Core i7-13620H (10 cores / 16 logical), numpy 2.5.3, OpenCV 5.0.0, torch 2.14.0+cpu, ultralytics 8.4.155, scikit-image 0.26.0. `torch.get_num_threads()` read at script start (after `modules.detector` imported ultralytics) returned **1**; whether the Streamlit process uses the same thread count: could not determine (not measured inside Streamlit).

---

## PART A — STATIC AUDIT

### A1. Solar angle convention

**Widget** — `app.py:555`
```python
theta = st.slider("Solar Incidence Angle θ", min_value=10, max_value=80, value=35, step=1)
```
- Label: `"Solar Incidence Angle θ"`
- Help text: **none** — the slider has no `help=` argument. The adjacent caption (`app.py:556`) reads: *"Controls illumination geometry for shadow-based depth. Larger angles usually increase estimated depth."* The "What this controls" expander (`app.py:576`) says: *"Solar incidence angle changes shadow-to-depth conversion sensitivity."*
- min 10, max 80, default 35, step 1 (integer).

**Variable chain**
| # | Name | Location |
|---|---|---|
| 1 | `theta` (slider return) | `app.py:555` |
| 2 | also enters `depth_slider_signature[0]` as `round(float(theta), 3)` | `app.py:612` |
| 3 | dict key `"theta"` in the returned params | `app.py:655` |
| 4 | `params["theta"]` passed as `theta_deg=` to `ensure_depth` | `app.py:1003`, `1115`, `1190`, `1332`, `1410` |
| 5 | `theta_deg` (parameter of `ensure_depth`; docstring: "Solar incidence angle.") | `app.py:375`, `379`; enters `signature` at `app.py:390` |
| 6 | keyword `solar_incidence_angle_deg=theta_deg` → `estimate_crater_depths` | `app.py:406` |
| 7 | `solar_incidence_angle_deg` (parameter; docstring "User-specified solar incidence angle.") | `depth.py:101`, `115` |
| 8 | keyword `solar_incidence_angle_deg=` → `depth_from_shadow` | `depth.py:140-144` |
| 9 | `solar_incidence_angle_deg` → local `theta` | `depth.py:75`, `94` |
| — | also echoed into each row as `solar_angle_deg` and `solar_incidence_deg` | `depth.py:165-166` |
| — | displayed caption `tan(theta) = …` | `app.py:1019` |
| — | multi-angle fusion second pass uses `np.clip(params["theta"] + 15, 10, 80)` | `app.py:1087` |

**Final formula** — `depth.py:94-95`
```python
theta = math.radians(np.clip(solar_incidence_angle_deg, 1.0, 89.0))
return float(shadow_length_px * pixel_scale_m * math.tan(theta))
```
i.e. `depth_m = L_px · s · tan(θ)`, θ clipped to [1°, 89°]. The formula string shown in the UI log is `"depth_m = shadow_length_px * pixel_scale_m * tan(theta)"` (`depth.py:194`, logged at `app.py:1014`). The docstring at `depth.py:82` also says `depth ~= shadow_length * tan(theta), where theta is solar incidence angle` — the docstring contradicts itself in the same way as the slider.

**Geometry.** A step of height *d* lit by the Sun at elevation *e* (measured from the horizontal) casts a shadow of horizontal length *L = d / tan e*, so *d = L · tan e*. Incidence *i* is measured from the surface normal, *i = 90° − e*, so *d = L · tan(90° − i) = L / tan i*.

**Verdict (definitive):**
- The running code multiplies by `tan(θ)`, which is the correct relation **only if θ is solar ELEVATION (from the horizontal)**.
- The label "Solar Incidence Angle", the `ensure_depth` docstring, the `depth.py` parameter names and docstrings all call it **INCIDENCE**. With incidence the correct formula would divide by `tan`. **The formula is not consistent with the label.**
- `PROJECT_EXPLAINED.md:221` states `depth_m = shadow_length_px * pixel_scale_m / tan(theta_incidence)`, and `PROJECT_EXPLAINED.md:225` "theta is solar incidence angle from surface normal"; `PROJECT_EXPLAINED.md:330` "Affects 1/tan(theta) term". That document is internally consistent physics, but **it is not what the code does. The code wins: the code multiplies by tan.**
- The UI caption (`app.py:556`, "Larger angles usually increase estimated depth") matches the code's behaviour (tan is increasing on [1°, 89°]), not the label.
- **Implication:** for the code as it runs, the correct label for the slider is **"Solar Elevation Angle"** (angle of the Sun above the horizon). Alternatively the label can stay "incidence" only if `depth.py:95` is changed to divide by `tan`. The two conventions coincide only at θ = 45° (see C1 table).

C1 confirms empirically that estimated depth rises monotonically with the slider value (×tan). Caveat on what C1 can and cannot show: the synthetic renderer has no elevation parameter — its `sun_angle_deg` is an *azimuth* (`synthetic.py:85`, `99`, `67-74`) — and θ is never used by the image or the shadow measurement, so the measured shadow length is identical at every θ. The sweep therefore settles *which convention the code implements*; it cannot validate either convention against physics.

Related finding (measured, see B3/C2): the "shadow length" is not tracking a shadow on the synthetic scenes. For 8 of 9 craters in seed 42 (synthetic-metadata boxes), `shadow_length_px` equals `(box side − 1) · (cos 35° + sin 35°)` = `(side − 1) · 1.392728480640038` to the 3 dp the table reports (e.g. CR-01, side 34: 33 × 1.392728… = 45.960; CR-05, side 112: 111 × 1.392728… = 154.593). The exception is CR-08 (side 70 → 96.098 predicted, 93.641 measured). The Otsu mask (`depth.py:27-33`) reaches opposite corners of the ROI, so `measure_shadow_length` (`depth.py:57-68`) returns the ROI's extent along the azimuth axis. Pearson r between shadow length and crater radius = 0.9998390831810087 (C2). Result: the slope column is nearly constant (61.907°–62.653°) for every crater.

### A2. Model identity (`best.pt`)

Loaded with ultralytics 8.4.155 (`YOLO('best.pt')`, `ultralytics.utils.torch_utils.model_info` / `get_flops`):

| Property | Value | Source |
|---|---|---|
| Variant | **YOLO11m** | `ckpt['model'].yaml['yaml_file'] = 'yolo11m.yaml'`, `scale = 'm'`; `train_args.model = 'yolo11m.pt'` |
| Layers (unfused, ultralytics summary) | 232 | `model_info` |
| Top-level blocks / all `nn.Module`s | 24 / 410 | `len(model.model)`, `len(list(model.modules()))` |
| Parameters | 20,053,779 | `model_info`; `sum(p.numel())` agrees |
| GFLOPs @640 (ultralytics default report) | 68.3033088 | `model_info` |
| GFLOPs @416 (the size the app uses) | 28.828166016 | `get_flops(model, 416)` |
| Classes | 1 | `model.names` |
| Class names | `{0: '0'}` (class is literally named `"0"`, not "crater") | `model.names` |
| Input size (training and app inference) | 416 | `train_args.imgsz = 416`; `detector.py:128` `imgsz=416` |
| Checkpoint date / ultralytics version at save | `2026-04-10T23:02:50.373141` / `8.4.37` | ckpt metadata |
| Training data path | `/kaggle/working/crater_data.yaml` | `train_args.data` |
| Training device | `'0,1'` (two CUDA GPUs) | `train_args.device` |
| Epochs configured / completed | 200 / 136 rows in `train_results` (last epoch 136; patience 40) | `train_args`, `train_results` |

**Comparison**
- `modules/detector.py:1` and `:483` docstrings: "YOLO11m" — **matches** the checkpoint. Also UI strings `app.py:52`, `886`, `895`, `955`, `997`, `utils/ui_components.py:608`: "YOLO11m" — matches.
- `train_crater_yolo.py:61`: `BASE_MODEL = "yolo11s.pt"` (comment "9.4M params") and docstrings `train_crater_yolo.py:2`, `5`, `111`: "YOLO11s" — **MISMATCH**. `README.md:70` "trains a YOLO11s-based crater detector" — **MISMATCH**.
- Further evidence `best.pt` was not produced by `train_crater_yolo.py`: script vs checkpoint `train_args` — epochs 150 vs 200 (`:68`), patience 30 vs 40 (`:71`), `degrees` 15.0 vs 180.0 (`:83`), `hsv_h` 0.015 vs 0.0 (`:80`), `hsv_s` 0.2 vs 0.0 (`:81`), `hsv_v` 0.4 vs 0.3 (`:82`), Intel XPU device (`:28-45`) vs `'0,1'`, local `LU3M6TGT_yolo_format/data.yaml` vs `/kaggle/working/crater_data.yaml`; checkpoint also has `copy_paste 0.1`, `cache 'ram'`, `name 'crater_detector_SOTA'`, none of which appear in the script.

**Metrics embedded in the checkpoint (training-time metadata, NOT re-measured by this audit — see Part D):** `ckpt['train_metrics']` = precision 0.77976, recall 0.78214, mAP50 0.8471, mAP50-95 0.64392 (fitness 0.64392). In `train_results` these are the values of epoch 96 (the epoch with maximum mAP50-95). The highest mAP50 in the history is 0.85921 at epoch 37. These were computed on the Kaggle validation split, which is not available locally and cannot be reproduced here.

### A3. Placeholder numbers

| # | File:line | Literal / construction | Where shown & how labelled | Status |
|---|---|---|---|---|
| 1 | `modules/detector.py:528` | `"map50_proxy": 0.847` (YOLO path) | Terminal log `app.py:347-349`: `"[DETECTOR] >> Source: yolo \| Craters detected: N \| mAP@0.5: 0.847 \| Time: …"` | **PLACEHOLDER.** Constant, independent of the image and of whether anything was detected. It equals the checkpoint's stored training-time `metrics/mAP50(B)` 0.8471 truncated to 3 dp (A2), but it is not computed at runtime and there is no ground truth for an uploaded image. |
| 2 | `modules/detector.py:509` | `"map50_proxy": 0.99` (synthetic-metadata path) | Same log line, rendered `mAP@0.5: 0.990` with `Source: synthetic-meta` | **PLACEHOLDER.** This is the value shown in the app's **default** flow (synthetic feed), where no detector runs at all (see #4). |
| 3 | `modules/detector.py:575` | `"map50_proxy": 0.82` (CV hybrid) | Not displayed anywhere (`run_cv_detection`, `app.py:369-372`, does not log it) | PLACEHOLDER, unused. |
| 4 | `modules/detector.py:463` | `confidence=0.98` for every synthetic-metadata "detection" | Overlay labels (`draw_detections`, `detector.py:603`), **"YOLO Detection Table (used for downstream pipeline)"** (`app.py:968-974`) and bar chart **"YOLO Confidence Score Distribution"** (`app.py:976-984`); overlay captioned `"YOLO11m — N craters"` (`app.py:955`) with a hard-coded YOLO badge (`app.py:953`) | **PLACEHOLDER** presented as YOLO confidence. In the default synthetic flow `run_detection` passes `hint_craters` (`app.py:326-338`) so `detect_craters` returns ground-truth metadata (`detector.py:501-510`) and YOLO is never called. |
| 5 | `modules/detector.py:210`, `243`, `377`, `292` | CV "confidence" = `0.6 + 0.2·min(1, r/80)` (Hough), `0.52 + 0.35·min(1, r/40)` (LoG), then `0.35·conf + 0.65·quality`, `+0.05` per merge, clipped to [0.05, 0.99] | Overlay labels on the CV image (`app.py:361-365`, `detector.py:603`) | Heuristic score from hard-coded weights, not a calibrated probability. Measured range 0.8983192179997763–0.99 (B2). |
| 6 | `modules/pathfinder.py:237` | `confidence = clip(goal_score + 0.2·(100 − min(100, 8·n_hazard)), 40, 99)` | `"Confidence: xx.x%"` under "RECOMMENDED LANDING COORDINATES" (`app.py:1386`); log `"LZ confidence: xx%"` (`app.py:1399`) | Heuristic with literal constants and a hard floor of 40 % / cap of 99 %; not a probability. |
| 7 | `modules/pathfinder.py:236` | `hazards_avoided = count(zone == "HAZARD")` | Log `"Obstacles avoided: N"` (`app.py:1398`) | Not measured from the path: it counts every HAZARD crater whether or not the route goes near it. |
| 8 | `modules/depth.py:289-291` | `(std(single) − std(fused)) / std(single) · 100` across craters | Log `"[FUSION] >> Depth uncertainty reduced by x.xx%"` (`app.py:1100`) | Mislabelled: this is the change in spread of depths *across different craters*, not uncertainty of any estimate. |
| 9 | `app.py:1087-1088`, `utils/synthetic.py:168`/`192` | Second "view" = θ+15°, φ+25°, and a linear brightness gradient of 18 levels (`angle_shift_deg=18.0`) | Fusion table (`app.py:1096-1097`) | Simulated second view, not a second observation. |
| 10 | `utils/ui_components.py:465`, `modules/reporter.py:163` | `"TELEMETRY: NOMINAL"` | HUD badge on every page; PDF report line | Hard-coded status text (non-numeric). |
| 11 | `app.py:449` | `points · 48.0 bytes` | `"estimated render memory: x.x MB"` (`app.py:1131`); selectbox labels "(256 MB)", "(512 MB)", "(1 GB)" (`app.py:588`) | Estimate from a constant (48 bytes/point); the MB figures in the option labels are literals not tied to the estimate. |
| 12 | `app.py:855` | `max(3, round(σ·4.5) \| 1)` | `"kernel ≈ 5×5 px"` (at σ=1.2) in the Gaussian formula panel | Does not reproduce OpenCV. Measured: with `ksize=(0,0)` on uint8, the result for σ=1.2 is reproduced only by ksize ≥ 7 (5 does not match); for σ=4.0 only ksize 25 matches (formula gives 19). OpenCV's automatic size for 8-bit is ≈ 6σ+1. `preprocess.py:59` docstring repeats the 4.5σ rule. |
| 13 | `modules/scorer.py:218` | `score_map` background `82.0` | Feeds A* cost and path (`pathfinder.py:56-58`) | Assumed score of all non-crater terrain; not displayed directly. |

Real measurements shown in the UI (not placeholders): `YOLO TIME` / `CV TIME` cards (`app.py:923-928`, from `time.perf_counter()` at `detector.py:499/522`, `548/569`), crater counts, image statistics (`preprocess.py:146-155`).

### A4. Parameter dump

Sidebar (`render_sidebar`, `app.py:531-663`):

| Control | Widget / label | Type | Min | Max | Default | Step | File:line |
|---|---|---|---|---|---|---|---|
| Solar angle | `st.slider("Solar Incidence Angle θ")` | int | 10 | 80 | 35 | 1 | `app.py:555` |
| Solar azimuth | `st.slider("Solar Azimuth φ")` | int | 0 | 359 | 35 | 1 | `app.py:558` |
| Depth threshold | `st.slider("Depth Safety Threshold Td (m)")` | float | 0.5 | 5.0 | 1.8 | 0.1 | `app.py:561` |
| Gear span | `st.slider("Landing Gear Span (m)")` | float | 1.0 | 5.0 | 2.6 | 0.1 | `app.py:564` |
| Density radius | `st.slider("Crater Density Radius (px)")` | int | 20 | 200 | 85 | 5 | `app.py:567` |
| Pixel scale | `st.slider("Pixel Scale (m/px)")` | float | 0.2 | 4.0 | 1.0 | 0.1 | `app.py:570` |
| 3D terrain profile | `st.selectbox("3D Terrain Memory Profile")` | enum | — | — | "High Fidelity (512 MB)" (`app.py:134`) | — | `app.py:586-595` |
| Multi-angle fusion | `st.toggle("ENABLE MULTI-ANGLE FUSION")` | bool | — | — | False | — | `app.py:642` |
| Regenerate synthetic | `st.button("↻ Regenerate Synthetic Surface")` | button | — | — | — | — | `app.py:645` |

Other controls (Step 3 / Step 4):

| Control | Widget / label | Min | Max | Default | Step | File:line |
|---|---|---|---|---|---|---|
| CLAHE grid size | `st.slider("CLAHE Grid Size")` | 2 | 16 | 8 (`app.py:129`) | 1 | `app.py:787-795` |
| CLAHE clip limit | `st.slider("CLAHE Clip Limit")` | 1.0 | 5.0 | 2.2 (`app.py:127`) | 0.1 | `app.py:797-805` |
| Gaussian σ | `st.slider("Gaussian Sigma (σ)")` | 0.5 | 4.0 | 1.2 (`app.py:131`) | 0.1 | `app.py:807-815` |
| Blend RAW↔SMOOTHED (fallback only) | `st.slider` | 0 | 100 | 50 | 1 | `app.py:769` |
| **Detection confidence** | **no widget** — hard-coded `conf_threshold=0.35` | — | — | 0.35 | — | `app.py:336`, `1078`, `1082`; function defaults `detector.py:104`, `478` |

Hard-coded pipeline constants that behave like parameters: YOLO `imgsz=416`, `device="cpu"`, `max_det=300` (`detector.py:128-130`); CV `max_detection_dim=896` (`detector.py:534`); upload resize `max_pixels = 14_000_000` with a 1024 px floor (`app.py:193-197`); A* grid `target_max_dim=180` (`pathfinder.py:13`); A* start `(w//2, 4)` (`pathfinder.py:202`); alternative goal offsets `[(-8, 6), (7, 10), (-10, 14)]` grid cells (`pathfinder.py:223`); zone cut-offs SAFE ≥ 70, CAUTION ≥ 40 (`scorer.py:55-59`); fusion IoU 0.25 (`app.py:1092`); synthetic image size 512 (`app.py:83`, `646`).

Documented but non-existent control: "Keep Original Upload Resolution" toggle (`PROJECT_EXPLAINED.md:66-71`, `353-355`) — not in the code; resizing is unconditional (`app.py:193-199`).

### A5. Architecture map

Dispatch: `main()` (`app.py:1523-1563`) calls `render_sidebar()` on every rerun, then exactly one step function based on `st.session_state.current_step`.

`reset_downstream(start_step)` (`app.py:204-227`) clears: ≤3 `preprocess`; ≤4 `detection`, `cv_detection`; ≤5 `depth`, `fusion`, `depth_signature`; ≤6 `terrain`; ≤7 `scoring`, `hazard_map`, `last_score_slider_signature`; ≤8 `paths`.

Global invalidation triggers (sidebar): depth-slider signature `(θ, φ, pixel_scale)` change → `reset_downstream(5)` (`app.py:629-633`); else scoring signature `(Td, gear, density, pixel_scale)` change → `reset_downstream(7)` (`app.py:636-639`); terrain profile change → `terrain = None` (`app.py:596-598`); Regenerate → new synthetic image + `reset_downstream(3)` (`app.py:645-652`).

| Step | Computes | Module functions called | Writes to session state | Displays | Invalidation |
|---|---|---|---|---|---|
| 1 Briefing `app.py:666-706` | Optional upload decode to grayscale, resize if > 14 MP | `decode_upload_to_gray` (`app.py:171`); UI helpers `render_typewriter_block`, `render_pipeline_flow` | `raw_image`, `image_name`, `file_size_bytes`, `synthetic_meta=None`; on launch `mission_started`, `completed_steps∪{1}`, `current_step=2` | Pipeline strip, uploader, active image, launch button | `reset_downstream(3)` whenever `uploader is not None` (`app.py:691`) — this runs on **every rerun of step 1** while a file sits in the uploader, not only on change |
| 2 Acquisition `app.py:709-734` | Image statistics, 64-bin histogram | `image_stats`, `compute_histogram` (via `histogram_figure`, `app.py:276`) | `completed_steps∪{2}`, log | Raw image, stats table, histogram | none |
| 3 Preprocess `app.py:775-880` | CLAHE → Gaussian blur | `preprocess_pipeline` (`preprocess.py:74`) | `pp_grid_size`, `pp_clip_limit`, `pp_sigma`, `preprocess`, `completed_steps∪{3}` | 3 sliders, formula panels, RAW/CLAHE/SMOOTHED images + histograms, before/after widget | On slider change `reset_downstream(4)` (`app.py:829`). `preprocess_pipeline` is re-run on every rerun of this step regardless (`app.py:832`) |
| 4 Detection `app.py:883-984` | YOLO (or synthetic metadata) and/or CV hybrid, on the **smoothed** image | `run_detection` → `ensure_preprocess`, `detect_craters(conf 0.35, hint_craters if synthetic)`, `draw_detections`; `run_cv_detection` → `detect_craters_cv`, `draw_detections` | `detection` (incl. `overlay`), `cv_detection`, `completed_steps∪{4}` | Count/time cards, speed comparison, two overlays, YOLO table and confidence chart | No explicit reset. Depth is invalidated indirectly through `depth_signature`, whose geometry key only covers the first 20 detections (`app.py:388`) |
| 5 Depth `app.py:987-1105` | Per-ROI Otsu shadow, projected length, depth, slope — on the **raw** image (`app.py:404`) | `ensure_depth` → `estimate_crater_depths`; optional fusion: `generate_secondary_solar_view`, `preprocess_pipeline` (module defaults, not the Step-3 sliders, `app.py:1070`), `detect_craters`, `estimate_crater_depths`, `fuse_depth_estimates` | `depth`, `depth_signature`, `completed_steps∪{5}`, `fusion` | tan(θ) caption, ROI viewer (first 8), depth table, depth bar chart, fusion table | `reset_downstream(6)` if signature changed (`app.py:400-401`) |
| 6 Terrain `app.py:1108-1170` | Gaussian-bowl depth map, heatmap, contour, 3D surface | `ensure_depth`, `ensure_terrain` → `build_depth_map`, `make_heatmap_figure`, `make_surface_figure`, `make_contour_figure` | `terrain`, `completed_steps∪{6}` | Caption with mesh size/memory estimate, heatmap, contour, 3D surface, auto-rotate | `ensure_terrain` has no cache check (`app.py:453-471`): rebuilds on every rerun of this step |
| 7 Scoring `app.py:1183-1251` | Per-crater score, zones, hazard map, score raster | `ensure_depth`, `ensure_scoring` → `score_landing_safety`, `annotate_hazard_map`, `build_score_map` | `scoring` (incl. `score_map`), `hazard_map`, `completed_steps∪{7}` | Score cards, zone counts, hazard map, gauge | `ensure_scoring` has no cache check (`app.py:474-501`): recomputed on every rerun of steps 7, 8, 9 |
| 8 Path `app.py:1325-1400` | A* over downsampled cost map + 3 offset-goal alternatives | `ensure_depth`, `ensure_scoring`, `ensure_paths` → `plan_descent_paths`, `draw_paths_on_map` | `paths` (incl. `overlay`), `completed_steps∪{8}` | Plotly path overlay, animation, goal pixel, landing confidence, path length, overlay image | `ensure_paths` has no cache check (`app.py:504-528`) |
| 9 Report `app.py:1403-1480` | PDF, PNG, CSV | `ensure_depth`, `ensure_scoring`, `ensure_paths`, `build_mission_pdf`, `image_to_png_bytes`, `crater_rows_to_csv` | `completed_steps∪{9}` | Final table, hazard map, landing zone, 3 download buttons | none |

Other structural facts: `mission_status_from_findings` (`app.py:253-273`) sets RED if `hazard > safe`. `render_pipeline_flow` has 8 labels (`ui_components.py:491-500`) for 9 steps (`app.py:44`), so its labels are shifted by one relative to the step numbers. Step 4's "Run YOLO Detection" button in synthetic mode never runs YOLO (`app.py:326-338`, `detector.py:501-510`).

---

## PART B — INSTRUMENTED RUNS

Script: `scripts/audit_run.py` (imports `modules.*` and `utils.synthetic` directly; no Streamlit). Raw output: `scripts/audit_results.json`; tables: `scripts/audit_tables.md`. The whole script ran in 26.139 s wall-clock.

Setup mirrors the app's defaults: synthetic 512×512 image; CLAHE clip 2.2, grid 8, σ 1.2; conf 0.35; θ 35, φ 35, Td 1.8, gear 2.6, density 85 px, pixel scale 1.0; terrain mesh target 512 (what `_terrain_profile_target_size("High Fidelity (512 MB)", (512,512))` returns). Detection runs on the smoothed image, depth on the raw image, as in `app.py:324`, `404`.

Seeds: 42, 7, 123. **Seed 42 is the scene the app always shows**: `app.py:83` and `app.py:646` call `generate_synthetic_lunar_surface(size=512)` without a seed, and the default is `seed=42` (`synthetic.py:84`). Measured: the default call is bitwise identical to `seed=42` (`app_default_call_equals_seed42: true`), and the generator is deterministic (`synthetic_determinism_seed42: true`).

Every stage runs through two detection paths:
- **yolo** — `detect_craters(..., hint_craters=None)`: real YOLO inference, which is what an uploaded image gets.
- **synthetic_meta** — `detect_craters(..., hint_craters=synth["craters"])`: what the app actually does in its default synthetic mode (no inference; ground-truth boxes, confidence 0.98).

### B1. Per-stage wall-clock timing

5 timed runs per path on seed 42, after one excluded warm-up run per path (the first YOLO call includes loading weights through `lru_cache`, `detector.py:26`). Stage boundaries: `preprocess` = `preprocess_pipeline`; `detection` = `detect_craters` (YOLO `predict` + post-processing); `depth` = `estimate_crater_depths`; `terrain_depth_map` = `build_depth_map`; `terrain_figures` = the three Plotly figure builders (together they equal `ensure_terrain`); `scoring` = `score_landing_safety` + `annotate_hazard_map` + `build_score_map` (= `ensure_scoring`); `astar` = `plan_descent_paths` (primary + 3 alternatives, without `draw_paths_on_map`). Streamlit rendering and serialisation are not included. The standard deviation is the sample stdev (n−1).

#### B1 timing (seconds, seed 42, 5 runs after 1 excluded warm-up)

| detection path | stage | mean | stdev (sample) | raw runs |
|---|---|---|---|---|
| yolo | preprocess | 0.0011238799983402715 | 0.0003805060449260085 | 0.001013999994029291, 0.0009585999941918999, 0.0009107000078074634, 0.0018011000065598637, 0.0009349999891128391 |
| yolo | detection | 0.08189003999868874 | 0.0031355692369807895 | 0.08359020001080353, 0.08577259999583475, 0.07763729999714997, 0.08231869999144692, 0.08013139999820851 |
| yolo | depth | 0.0017990200023632497 | 8.21052858025457e-05 | 0.0018641000060597435, 0.0018429000047035515, 0.0016687999886926264, 0.0017671000096015632, 0.0018522000027587637 |
| yolo | terrain_depth_map | 0.003740780003136024 | 0.00022609396706392944 | 0.003978699998697266, 0.003987400006735697, 0.003507400004309602, 0.0035923999967053533, 0.0036380000092322007 |
| yolo | terrain_figures | 0.017837420001160353 | 0.000890179360834939 | 0.017227300006197765, 0.019407100000535138, 0.017647199987550266, 0.017426200007321313, 0.017479300004197285 |
| yolo | terrain_total | 0.021578200004296378 | 0.0010176561885993903 | 0.02120600000489503, 0.023394500007270835, 0.021154599991859868, 0.021018600004026666, 0.021117300013429485 |
| yolo | scoring | 0.1052418800041778 | 0.0021660784037622117 | 0.1065330000128597, 0.10753700000350364, 0.10183250000409316, 0.10484240000369027, 0.10546449999674223 |
| yolo | astar | 0.18397008000174536 | 0.00842376868265919 | 0.17765739999595098, 0.17890559999796096, 0.1984020999952918, 0.18408450001152232, 0.1808008000080008 |
| synthetic_meta | preprocess | 0.0010151599970413371 | 0.00011375923101891395 | 0.001034899993101135, 0.0009233000018866733, 0.0009375999943586066, 0.0009766000002855435, 0.0012033999955747277 |
| synthetic_meta | detection | 0.00028533999575302004 | 2.7228989223117463e-05 | 0.00033319999056402594, 0.00026699999580159783, 0.00028079999901819974, 0.00027459999546408653, 0.0002710999979171902 |
| synthetic_meta | depth | 0.0012610800040420145 | 6.615358951571968e-05 | 0.0012106000067433342, 0.0013718000118387863, 0.0012251000007381663, 0.0012254000030225143, 0.0012724999978672713 |
| synthetic_meta | terrain_depth_map | 0.0026631999964592977 | 7.67116349288451e-05 | 0.002642099992954172, 0.0026885999977821484, 0.0027280999929644167, 0.0027176000003237277, 0.002539599998272024 |
| synthetic_meta | terrain_figures | 0.014799360002507455 | 0.0008450530725733725 | 0.015489600002183579, 0.015186200005700812, 0.013830900003085844, 0.01554710000345949, 0.013942999998107553 |
| synthetic_meta | terrain_total | 0.017462559998966752 | 0.0008714730386918626 | 0.01813169999513775, 0.01787480000348296, 0.01655899999605026, 0.01826470000378322, 0.016482599996379577 |
| synthetic_meta | scoring | 0.09075820000143722 | 0.0058163196423589225 | 0.08641200000420213, 0.08822230000805575, 0.0868285000033211, 0.10041309999360237, 0.09191509999800473 |
| synthetic_meta | astar | 0.8142462000047089 | 0.016245796670516486 | 0.8096073000051547, 0.7922283000079915, 0.8196834999980638, 0.8369947000028333, 0.8127172000095015 |

Warm-up run (excluded; includes YOLO weight load on first call): {"yolo": {"wall_total": 1.9452794999961043, "preprocess": 0.0012165999942226335, "detection": 1.430902199994307, "depth": 0.022414799997932278, "terrain_depth_map": 0.0038804999931016937, "terrain_figures": 0.13206129999889527, "terrain_total": 0.13594179999199696, "scoring": 0.1087132999964524, "astar": 0.24601220000477042}, "synthetic_meta": {"wall_total": 0.922672100001364, "preprocess": 0.000943200007895939, "detection": 0.00026259999140165746, "depth": 0.0012053999962517992, "terrain_depth_map": 0.0031432999967364594, "terrain_figures": 0.014051299993298016, "terrain_total": 0.017194599990034476, "scoring": 0.09175039999536239, "astar": 0.8111435999890091}}

What the timing shows: in the app's default synthetic flow, "detection" takes 0.00028533999575302004 s on average because no model runs. A* is the slowest stage on both paths; its cost depends on how far away the chosen goal is (synthetic_meta goal (405, 387) vs yolo goal (492, 100), see B5).

### B2. Detection output on 3 seeds (YOLO vs CV baseline vs synthetic metadata)

"Matched GT" is an extra measurement against the synthetic ground-truth boxes: greedy one-to-one matching at IoU ≥ 0.5, with GT boxes built by the project's own `_to_detection_record` from the generator's centre/radius. It is synthetic ground truth, not lunar ground truth. The CV baseline runs on the same smoothed image as YOLO (`detect_craters_cv`, as `app.py:360` does).

#### B2 detection (defaults: CLAHE 2.2/8, sigma 1.2, conf 0.35)

| seed | GT craters | detector | count | conf min | conf median | conf max | diam px min | diam px max | matched GT @IoU>=0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 9 | yolo | 10 | 0.5229388475418091 | 0.7407869100570679 | 0.8692474365234375 | 17.0 | 127.0 | 9/9 |
| 42 | 9 | cv_hybrid | 10 | 0.9032651081096981 | 0.9649123748143513 | 0.99 | 38.0 | 238.0 | 3/9 |
| 42 | 9 | synthetic_meta | 9 | 0.98 | 0.98 | 0.98 | 34.0 | 116.0 | 9/9 |
| 7 | 11 | yolo | 8 | 0.48157837986946106 | 0.7826994359493256 | 0.8730961084365845 | 25.5 | 113.0 | 8/11 |
| 7 | 11 | cv_hybrid | 10 | 0.9362029892603556 | 0.9562515699089675 | 0.99 | 46.0 | 230.5 | 5/11 |
| 7 | 11 | synthetic_meta | 11 | 0.98 | 0.98 | 0.98 | 22.0 | 108.0 | 11/11 |
| 123 | 10 | yolo | 10 | 0.5181504487991333 | 0.6831502020359039 | 0.8704314231872559 | 22.0 | 131.0 | 10/10 |
| 123 | 10 | cv_hybrid | 10 | 0.8983192179997763 | 0.9542991148657636 | 0.99 | 46.0 | 226.0 | 6/10 |
| 123 | 10 | synthetic_meta | 10 | 0.98 | 0.98 | 0.98 | 20.0 | 118.0 | 10/10 |

YOLO vs CV on identical input: seed 42 — YOLO 10, CV 10; seed 7 — YOLO 8, CV 10; seed 123 — YOLO 10, CV 10. The counts are similar, but CV boxes match far fewer ground-truth craters (3/9, 5/11, 6/10 vs YOLO 9/9, 8/11, 10/10). CV diameters reach 238.0 px on a 512 px image. Every CV run reported `scale=1.000` (no downsampling at 512 px).

### B3. Depth output

Per-crater tables for seed 42 on both detection paths. `Otsu T` and `shadow px` (mask pixel count after morphology) come from re-running `compute_otsu_shadow_mask` on the same ROI. The table values (`shadow length px`, `depth m`, `slope deg`) are as emitted by `depth.py`, which rounds them to 3 dp (`depth.py:164-169`).

#### B3 depth tables (seed 42, theta 35, azimuth 35, 1.0 m/px)

##### synthetic_meta

| id | shadow length px | depth m | slope deg | confidence | diameter px | Otsu T | shadow px | zero-depth reason |
|---|---|---|---|---|---|---|---|---|
| CR-01 | 45.96 | 32.182 | 62.155 | 0.98 | 34.0 | 135.0 | 689 |  |
| CR-02 | 73.815 | 51.686 | 62.418 | 0.98 | 54.0 | 158.0 | 1683 |  |
| CR-03 | 79.386 | 55.586 | 62.448 | 0.98 | 58.0 | 173.0 | 1771 |  |
| CR-04 | 90.527 | 63.388 | 62.498 | 0.98 | 66.0 | 179.0 | 2224 |  |
| CR-05 | 154.593 | 108.247 | 62.646 | 0.98 | 112.0 | 137.0 | 7252 |  |
| CR-06 | 160.164 | 112.148 | 62.653 | 0.98 | 116.0 | 177.0 | 6996 |  |
| CR-07 | 154.593 | 108.247 | 62.646 | 0.98 | 112.0 | 131.0 | 7280 |  |
| CR-08 | 93.641 | 65.568 | 61.907 | 0.98 | 70.0 | 100.0 | 2572 |  |
| CR-09 | 45.96 | 32.182 | 62.155 | 0.98 | 34.0 | 175.0 | 677 |  |

rows=9, skipped (no row)=0, zero depth=0

##### yolo

| id | shadow length px | depth m | slope deg | confidence | diameter px | Otsu T | shadow px | zero-depth reason |
|---|---|---|---|---|---|---|---|---|
| CR-01 | 164.342 | 115.073 | 62.658 | 0.8692474365234375 | 119.0 | 179.0 | 7413 |  |
| CR-02 | 100.276 | 70.214 | 62.533 | 0.8522818684577942 | 73.0 | 180.0 | 3154 |  |
| CR-03 | 83.564 | 58.512 | 62.469 | 0.8309051394462585 | 61.0 | 175.0 | 2008 |  |
| CR-04 | 77.993 | 54.611 | 62.441 | 0.7702561616897583 | 57.0 | 162.0 | 1884 |  |
| CR-05 | 93.886 | 65.74 | 62.481 | 0.7589907646179199 | 68.5 | 100.0 | 2254 |  |
| CR-06 | 142.877 | 100.044 | 62.649 | 0.7225830554962158 | 103.5 | 142.0 | 4983 |  |
| CR-07 | 50.138 | 35.107 | 62.213 | 0.7092379927635193 | 37.0 | 179.0 | 833 |  |
| CR-08 | 175.484 | 122.875 | 62.671 | 0.6920771598815918 | 127.0 | 137.0 | 9992 |  |
| CR-09 | 47.353 | 33.157 | 62.175 | 0.6427695751190186 | 35.0 | 138.0 | 736 |  |
| CR-10 | 14.254 | 9.98 | 49.58 | 0.5229388475418091 | 17.0 | 204.0 | 117 |  |

rows=10, skipped (no row)=0, zero depth=0

##### Zero-depth count across seeds

| seed | detections | rows | zero depth | skipped | reasons |
|---|---|---|---|---|---|
| 42 | yolo | 10 | 0 | 0 | - |
| 42 | synthetic_meta | 9 | 0 | 0 | - |
| 7 | yolo | 8 | 0 | 0 | - |
| 7 | synthetic_meta | 11 | 0 | 0 | - |
| 123 | yolo | 10 | 0 | 0 | - |
| 123 | synthetic_meta | 10 | 0 | 0 | - |

**Zero-depth result:** across all 6 runs (3 seeds × 2 detection paths, 58 craters), **0 craters returned zero depth and 0 were skipped**, so no failure reasons were observed. From the code, only these paths can produce zero or missing depth: (a) ROI ≤ 2 px on an axis → crater skipped with no row (`depth.py:133-134`); (b) fewer than 3 mask pixels after Otsu + open/close → length 0 (`depth.py:53-55`); (c) all mask pixels share one projection value → length 0 (`depth.py:68`). Otsu itself cannot "fail" in `cv2.threshold`; a uniform ROI gives an empty mask, which ends up in case (b).

### B4. Scoring output

#### B4 scoring (Td 1.8, gear 2.6, density 85 px, 1.0 m/px)

| seed | detections | n | score min | median | max | mean | SAFE | CAUTION | HAZARD | overall_score |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | yolo | 10 | 25.81 | 33.705 | 55.77 | 35.387 | 0 | 2 | 8 | 35.39 |
| 42 | synthetic_meta | 9 | 22.41 | 30.83 | 40.43 | 32.601111111111116 | 0 | 2 | 7 | 32.6 |
| 7 | yolo | 8 | 27.21 | 31.490000000000002 | 40.37 | 32.4475 | 0 | 1 | 7 | 32.45 |
| 7 | synthetic_meta | 11 | 25.32 | 29.77 | 38.59 | 30.566363636363636 | 0 | 0 | 11 | 30.57 |
| 123 | yolo | 10 | 25.15 | 33.43 | 41.6 | 33.202 | 0 | 2 | 8 | 33.2 |
| 123 | synthetic_meta | 10 | 25.77 | 33.87 | 42.59 | 33.755 | 0 | 2 | 8 | 33.75 |

No crater reached SAFE (score ≥ 70) on any seed or detection path at the default parameters.

### B5. Pathfinding output

"Path nodes" = number of waypoints in the primary path (full-resolution coordinates, one per A* grid cell visited by the path). A* runs on a 180×180 grid (`pathfinder.py:13`, 512/180 ≈ 2.844 px per cell). `landing_confidence` is the heuristic from A3 #6, listed only for completeness.

#### B5 pathfinding

| seed | detections | route found | path length m | path nodes | alternatives ok | alt nodes | start px | goal px | A* grid (h x w) | landing_confidence (heuristic) |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | yolo | True | 274.24 | 83 | 3/3 | [75, 90, 73] | (256, 4) | (492, 100) | 180x180 | 62.97 |
| 42 | synthetic_meta | True | 508.12 | 156 | 3/3 | [151, 169, 153] | (256, 4) | (405, 387) | 180x180 | 49.23 |
| 7 | yolo | True | 212.64 | 62 | 3/3 | [68, 72, 76] | (256, 4) | (164, 177) | 180x180 | 49.17 |
| 7 | synthetic_meta | True | 212.64 | 62 | 3/3 | [68, 72, 76] | (256, 4) | (164, 177) | 180x180 | 40.99 |
| 123 | yolo | True | 53.94 | 18 | 3/3 | [24, 28, 33] | (256, 4) | (244, 52) | 180x180 | 48.8 |
| 123 | synthetic_meta | True | 50.94 | 17 | 3/3 | [23, 27, 31] | (256, 4) | (243, 51) | 180x180 | 49.79 |

A route was found in 6/6 runs, and 3/3 alternatives succeeded in every run (18/18).

---

## PART C — SENSITIVITY EXPERIMENTS

### C1. Solar angle sweep

The image is fixed (seed 42). θ goes from 5° to 85° in 10° steps, with φ 35 and 1.0 m/px. θ = 5° and 85° are outside the slider's 10–80 range but inside `depth_from_shadow`'s clip [1, 89]. "Code" column = the unmodified `depth.py`. "Alt" column = the same pipeline with `modules.depth.depth_from_shadow` monkey-patched to `L · s / tan(θ)`; everything else, including the 3 dp rounding, is unchanged.

#### C1 solar angle sweep — seed 42, detections=synthetic_meta, azimuth 35, 1.0 m/px

| theta deg | tan(theta) | mean shadow length px | mean depth m (code: × tan) | mean depth m (alt: ÷ tan) | n craters |
|---|---|---|---|---|---|
| 5 | 0.08748866352592401 | 99.84877777777778 | 8.735666666666667 | 1141.275222222222 | 9 |
| 15 | 0.2679491924311227 | 99.84877777777778 | 26.754444444444445 | 372.6401111111111 | 9 |
| 25 | 0.4663076581549986 | 99.84877777777778 | 46.56033333333333 | 214.1261111111111 | 9 |
| 35 | 0.7002075382097097 | 99.84877777777778 | 69.9148888888889 | 142.59855555555555 | 9 |
| 45 | 0.9999999999999999 | 99.84877777777778 | 99.84877777777778 | 99.84877777777778 | 9 |
| 55 | 1.4281480067421144 | 99.84877777777778 | 142.59855555555555 | 69.9148888888889 | 9 |
| 65 | 2.1445069205095586 | 99.84877777777778 | 214.1261111111111 | 46.56033333333333 | 9 |
| 75 | 3.7320508075688776 | 99.84877777777778 | 372.6401111111111 | 26.754444444444445 | 9 |
| 85 | 11.430052302761348 | 99.84877777777778 | 1141.275222222222 | 8.735666666666667 | 9 |

#### C1 solar angle sweep — seed 42, detections=yolo, azimuth 35, 1.0 m/px

| theta deg | tan(theta) | mean shadow length px | mean depth m (code: × tan) | mean depth m (alt: ÷ tan) | n craters |
|---|---|---|---|---|---|
| 5 | 0.08748866352592401 | 95.0167 | 8.3129 | 1086.0461 | 10 |
| 15 | 0.2679491924311227 | 95.0167 | 25.459600000000002 | 354.60720000000003 | 10 |
| 25 | 0.4663076581549986 | 95.0167 | 44.3071 | 203.764 | 10 |
| 35 | 0.7002075382097097 | 95.0167 | 66.5313 | 135.698 | 10 |
| 45 | 0.9999999999999999 | 95.0167 | 95.0167 | 95.0167 | 10 |
| 55 | 1.4281480067421144 | 95.0167 | 135.698 | 66.5313 | 10 |
| 65 | 2.1445069205095586 | 95.0167 | 203.764 | 44.3071 | 10 |
| 75 | 3.7320508075688776 | 95.0167 | 354.60720000000003 | 25.459600000000002 | 10 |
| 85 | 11.430052302761348 | 95.0167 | 1086.0461 | 8.3129 | 10 |

**Result:** the mean measured shadow length stays the same at every θ (99.84877777777778 px synthetic_meta; 95.0167 px yolo), so θ only rescales depth. In the code as written, mean depth **rises** monotonically with θ (8.735666666666667 → 1141.275222222222 m for synthetic_meta). The 1/tan column is its mirror image around 45°, where the two agree (99.84877777777778 m). The behaviour matches **elevation** semantics (×tan), not the "incidence" label (A1).

### C2. Ground-truth check (synthetic ground truth — NOT real lunar terrain)

**Important limitations, stated before the numbers:**
1. `utils/synthetic.py` does **not** return the true depth. The returned `craters` entries hold only `crater_id`, `center_x`, `center_y`, `radius_px`, `sun_angle_deg` (`synthetic.py:134-142`); `depth_scale` (`synthetic.py:128`) is drawn and then thrown away. The audit script recovers it by replaying the generator's RNG draws in the same order (`replay_crater_specs`). The replay is **verified**: re-imprinting the replayed craters produces a `height_map` bitwise equal to the generator's (`replay_verified_bitwise_equal_height_map: True`).
2. The ground truth is in **arbitrary height units**. The generator defines no metres-per-height-unit or metres-per-pixel scale, so "error in metres" is **dimensionally undefined**. The numbers below compare `depth_m` at pixel scale 1.0 against the height units numerically, as requested. Treat them as a consistency check, not an accuracy figure.
3. The renderer does not cast geometric shadows. Image brightness gets `bowl · (0.85 − 0.15·illum) + directional_rim` added (`synthetic.py:74-78`), with no dependence on sun elevation, so there is no physically correct shadow length to recover.
4. Two ground-truth definitions are given: `depth_scale` (amplitude of the Gaussian bowl, `synthetic.py:64`) and rim-to-floor relief measured on the `height_map` (max height at 0.9–1.1 r minus min height within 0.3 r; this includes the rim term `0.35·rim` and overlap from neighbouring craters).

Setup: seed 42, synthetic_meta boxes (the app's default flow, which match ground-truth craters 1:1 by index), θ 35, φ 35, 1.0 m/px.

#### C2 ground truth (seed 42, synthetic_meta detections, theta 35, 1.0 m/px) — replay verified: True

| id | radius px | GT depth_scale (height units) | GT rim-to-floor relief (height units) | est depth_m | est − depth_scale | |est − depth_scale| | est − relief |
|---|---|---|---|---|---|---|---|
| CR-01 | 17 | 54.35409181740661 | 48.619232177734375 | 32.182 | -22.17209181740661 | 22.17209181740661 | -16.437232177734373 |
| CR-02 | 27 | 40.10082694214742 | 36.74742126464844 | 51.686 | 11.585173057852579 | 11.585173057852579 | 14.938578735351562 |
| CR-03 | 29 | 25.322605527577533 | 25.66147232055664 | 55.586 | 30.263394472422465 | 30.263394472422465 | 29.924527679443358 |
| CR-04 | 33 | 26.284537170549584 | 25.52847671508789 | 63.388 | 37.103462829450415 | 37.103462829450415 | 37.85952328491211 |
| CR-05 | 56 | 56.92744308861561 | 50.56437301635742 | 108.247 | 51.31955691138439 | 51.31955691138439 | 57.68262698364258 |
| CR-06 | 58 | 28.868372376793225 | 28.423799514770508 | 112.148 | 83.27962762320678 | 83.27962762320678 | 83.72420048522949 |
| CR-07 | 56 | 50.93171324141389 | 55.66045379638672 | 108.247 | 57.31528675858611 | 57.31528675858611 | 52.58654620361328 |
| CR-08 | 35 | 50.598478071510016 | 66.78636932373047 | 65.568 | 14.969521928489982 | 14.969521928489982 | -1.218369323730471 |
| CR-09 | 17 | 27.905392547346114 | 27.401552200317383 | 32.182 | 4.276607452653888 | 4.276607452653888 | 4.780447799682619 |

- replay_verified_bitwise_equal_height_map: True
- MAE_vs_depth_scale: 34.69830253905036
- mean_signed_error_vs_depth_scale: 29.77117102407111
- n_over: 8
- n_under: 1
- MAE_vs_rim_to_floor_relief: 33.239116963704426
- mean_signed_error_vs_relief: 29.315649963378902
- pearson_est_vs_depth_scale: 0.22621023094987164
- pearson_est_vs_radius_px: 0.9998390451381755
- pearson_shadow_len_vs_radius_px: 0.9998390831810087

**Result:** the estimate is **systematically biased high**: 8 of 9 craters over, 1 under; mean signed error +29.77117102407111 against `depth_scale`, +29.315649963378902 against rim-to-floor relief. MAE = 34.69830253905036 (vs `depth_scale`) and 33.239116963704426 (vs relief). The estimate barely tracks true depth (Pearson r = 0.22621023094987164) and almost perfectly tracks crater radius (r = 0.9998390451381755), which agrees with the A1 finding that the measured "shadow length" is the box extent along the azimuth axis.

### C3. Parameter sensitivity (depth threshold, gear span)

Seed 42, θ 35, φ 35, density 85 px, 1.0 m/px. Each 1-D sweep holds the other parameter at its default (Td 1.8 / gear 2.6) and covers the full slider range at slider step 0.1. The full 46×41 grid was also evaluated, and its distinct outcomes are listed.

#### C3 depth threshold sweep — seed 42, detections=synthetic_meta (gear 2.6, density 85)

| Td m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 0.5 | 0 | 0 | 9 |
| 0.6 | 0 | 0 | 9 |
| 0.7 | 0 | 0 | 9 |
| 0.8 | 0 | 0 | 9 |
| 0.9 | 0 | 0 | 9 |
| 1.0 | 0 | 0 | 9 |
| 1.1 | 0 | 0 | 9 |
| 1.2 | 0 | 0 | 9 |
| 1.3 | 0 | 0 | 9 |
| 1.4 | 0 | 0 | 9 |
| 1.5 | 0 | 0 | 9 |
| 1.6 | 0 | 0 | 9 |
| 1.7 | 0 | 2 | 7 |
| 1.8 | 0 | 2 | 7 |
| 1.9 | 0 | 2 | 7 |
| 2.0 | 0 | 2 | 7 |
| 2.1 | 0 | 2 | 7 |
| 2.2 | 0 | 2 | 7 |
| 2.3 | 0 | 2 | 7 |
| 2.4 | 0 | 2 | 7 |
| 2.5 | 0 | 2 | 7 |
| 2.6 | 0 | 2 | 7 |
| 2.7 | 0 | 2 | 7 |
| 2.8 | 0 | 2 | 7 |
| 2.9 | 0 | 2 | 7 |
| 3.0 | 0 | 2 | 7 |
| 3.1 | 0 | 2 | 7 |
| 3.2 | 0 | 2 | 7 |
| 3.3 | 0 | 2 | 7 |
| 3.4 | 0 | 2 | 7 |
| 3.5 | 0 | 3 | 6 |
| 3.6 | 0 | 3 | 6 |
| 3.7 | 0 | 3 | 6 |
| 3.8 | 0 | 3 | 6 |
| 3.9 | 0 | 3 | 6 |
| 4.0 | 0 | 3 | 6 |
| 4.1 | 0 | 3 | 6 |
| 4.2 | 0 | 3 | 6 |
| 4.3 | 0 | 3 | 6 |
| 4.4 | 0 | 3 | 6 |
| 4.5 | 0 | 3 | 6 |
| 4.6 | 0 | 4 | 5 |
| 4.7 | 0 | 4 | 5 |
| 4.8 | 0 | 4 | 5 |
| 4.9 | 0 | 4 | 5 |
| 5.0 | 0 | 4 | 5 |

#### C3 gear span sweep — seed 42, detections=synthetic_meta (Td 1.8, density 85)

| gear m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 1.0 | 0 | 0 | 9 |
| 1.1 | 0 | 0 | 9 |
| 1.2 | 0 | 0 | 9 |
| 1.3 | 0 | 0 | 9 |
| 1.4 | 0 | 0 | 9 |
| 1.5 | 0 | 0 | 9 |
| 1.6 | 0 | 0 | 9 |
| 1.7 | 0 | 0 | 9 |
| 1.8 | 0 | 0 | 9 |
| 1.9 | 0 | 0 | 9 |
| 2.0 | 0 | 0 | 9 |
| 2.1 | 0 | 0 | 9 |
| 2.2 | 0 | 0 | 9 |
| 2.3 | 0 | 0 | 9 |
| 2.4 | 0 | 2 | 7 |
| 2.5 | 0 | 2 | 7 |
| 2.6 | 0 | 2 | 7 |
| 2.7 | 0 | 2 | 7 |
| 2.8 | 0 | 2 | 7 |
| 2.9 | 0 | 2 | 7 |
| 3.0 | 0 | 2 | 7 |
| 3.1 | 0 | 2 | 7 |
| 3.2 | 0 | 2 | 7 |
| 3.3 | 0 | 2 | 7 |
| 3.4 | 0 | 2 | 7 |
| 3.5 | 0 | 2 | 7 |
| 3.6 | 0 | 2 | 7 |
| 3.7 | 0 | 2 | 7 |
| 3.8 | 0 | 2 | 7 |
| 3.9 | 0 | 2 | 7 |
| 4.0 | 0 | 2 | 7 |
| 4.1 | 0 | 2 | 7 |
| 4.2 | 0 | 2 | 7 |
| 4.3 | 0 | 2 | 7 |
| 4.4 | 0 | 2 | 7 |
| 4.5 | 0 | 2 | 7 |
| 4.6 | 0 | 2 | 7 |
| 4.7 | 0 | 2 | 7 |
| 4.8 | 0 | 2 | 7 |
| 4.9 | 0 | 2 | 7 |
| 5.0 | 0 | 2 | 7 |

Full 46×41 grid distinct (SAFE, CAUTION, HAZARD) outcomes: [(0, 0, 9), (0, 2, 7), (0, 3, 6), (0, 4, 5), (0, 5, 4)]; corners: {'td=0.5,gear=1.0': (0, 0, 9), 'td=0.5,gear=5.0': (0, 0, 9), 'td=5.0,gear=1.0': (0, 3, 6), 'td=5.0,gear=5.0': (0, 5, 4)}

#### C3 depth threshold sweep — seed 42, detections=yolo (gear 2.6, density 85)

| Td m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 0.5 | 0 | 1 | 9 |
| 0.6 | 0 | 1 | 9 |
| 0.7 | 0 | 1 | 9 |
| 0.8 | 0 | 1 | 9 |
| 0.9 | 0 | 1 | 9 |
| 1.0 | 0 | 1 | 9 |
| 1.1 | 0 | 1 | 9 |
| 1.2 | 0 | 1 | 9 |
| 1.3 | 0 | 1 | 9 |
| 1.4 | 0 | 1 | 9 |
| 1.5 | 0 | 1 | 9 |
| 1.6 | 0 | 1 | 9 |
| 1.7 | 0 | 1 | 9 |
| 1.8 | 0 | 2 | 8 |
| 1.9 | 0 | 2 | 8 |
| 2.0 | 0 | 3 | 7 |
| 2.1 | 0 | 3 | 7 |
| 2.2 | 0 | 3 | 7 |
| 2.3 | 0 | 3 | 7 |
| 2.4 | 0 | 3 | 7 |
| 2.5 | 0 | 3 | 7 |
| 2.6 | 0 | 3 | 7 |
| 2.7 | 0 | 3 | 7 |
| 2.8 | 0 | 3 | 7 |
| 2.9 | 0 | 3 | 7 |
| 3.0 | 0 | 3 | 7 |
| 3.1 | 0 | 3 | 7 |
| 3.2 | 0 | 3 | 7 |
| 3.3 | 0 | 3 | 7 |
| 3.4 | 0 | 3 | 7 |
| 3.5 | 0 | 3 | 7 |
| 3.6 | 0 | 3 | 7 |
| 3.7 | 0 | 3 | 7 |
| 3.8 | 0 | 4 | 6 |
| 3.9 | 0 | 4 | 6 |
| 4.0 | 0 | 4 | 6 |
| 4.1 | 0 | 5 | 5 |
| 4.2 | 0 | 5 | 5 |
| 4.3 | 0 | 5 | 5 |
| 4.4 | 0 | 5 | 5 |
| 4.5 | 0 | 5 | 5 |
| 4.6 | 0 | 5 | 5 |
| 4.7 | 0 | 5 | 5 |
| 4.8 | 0 | 5 | 5 |
| 4.9 | 0 | 5 | 5 |
| 5.0 | 0 | 5 | 5 |

#### C3 gear span sweep — seed 42, detections=yolo (Td 1.8, density 85)

| gear m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 1.0 | 0 | 1 | 9 |
| 1.1 | 0 | 1 | 9 |
| 1.2 | 0 | 1 | 9 |
| 1.3 | 0 | 1 | 9 |
| 1.4 | 0 | 1 | 9 |
| 1.5 | 0 | 1 | 9 |
| 1.6 | 0 | 1 | 9 |
| 1.7 | 0 | 1 | 9 |
| 1.8 | 0 | 1 | 9 |
| 1.9 | 0 | 1 | 9 |
| 2.0 | 0 | 1 | 9 |
| 2.1 | 0 | 1 | 9 |
| 2.2 | 0 | 1 | 9 |
| 2.3 | 0 | 1 | 9 |
| 2.4 | 0 | 1 | 9 |
| 2.5 | 0 | 1 | 9 |
| 2.6 | 0 | 2 | 8 |
| 2.7 | 0 | 2 | 8 |
| 2.8 | 0 | 2 | 8 |
| 2.9 | 0 | 2 | 8 |
| 3.0 | 0 | 3 | 7 |
| 3.1 | 0 | 3 | 7 |
| 3.2 | 0 | 3 | 7 |
| 3.3 | 0 | 3 | 7 |
| 3.4 | 0 | 3 | 7 |
| 3.5 | 0 | 3 | 7 |
| 3.6 | 0 | 3 | 7 |
| 3.7 | 0 | 3 | 7 |
| 3.8 | 0 | 3 | 7 |
| 3.9 | 0 | 3 | 7 |
| 4.0 | 0 | 3 | 7 |
| 4.1 | 0 | 3 | 7 |
| 4.2 | 0 | 3 | 7 |
| 4.3 | 0 | 3 | 7 |
| 4.4 | 0 | 3 | 7 |
| 4.5 | 0 | 3 | 7 |
| 4.6 | 0 | 3 | 7 |
| 4.7 | 0 | 3 | 7 |
| 4.8 | 0 | 3 | 7 |
| 4.9 | 0 | 3 | 7 |
| 5.0 | 0 | 3 | 7 |

Full 46×41 grid distinct (SAFE, CAUTION, HAZARD) outcomes: [(0, 0, 10), (0, 1, 9), (0, 2, 8), (0, 3, 7), (0, 4, 6), (0, 5, 5), (0, 6, 4), (1, 5, 4)]; corners: {'td=0.5,gear=1.0': (0, 0, 10), 'td=0.5,gear=5.0': (0, 1, 9), 'td=5.0,gear=1.0': (0, 4, 6), 'td=5.0,gear=5.0': (1, 5, 4)}

**Result:** both sliders change the counts monotonically: raising Td or gear span moves craters from HAZARD to CAUTION. SAFE stays 0 everywhere except one outcome, (1, 5, 4), which appears only on the yolo boxes and only in the full grid (at Td 5.0, gear 5.0). Why, from the code: estimated depths are 9.98–122.875 m (B3) while the depth penalty's half-saturation point is `5·Td` ≤ 25 m (`scorer.py:97-98`), so the depth penalty alone is usually close to its 45-point cap. The diameter penalty (`scorer.py:101-102`) and density penalty (`scorer.py:105`) then push most scores below 40.

---

## PART D — VALIDATION

`LU3M6TGT_yolo_format/` is **not present** locally (`ls: cannot access 'LU3M6TGT_yolo_format': No such file or directory`; the folder is gitignored at `.gitignore:13`). `yolo val` was **not run**.

**No genuine detection metrics (precision, recall, mAP@0.5, mAP@0.5:0.95) can be produced without the dataset.** The `map50_proxy` literals (0.847 / 0.99 / 0.82, A3) are placeholders and are not substitutes. The training-time numbers stored inside `best.pt` (A2) were computed by the original Kaggle run on a validation split this audit cannot access, so they cannot be verified here. If cited, they must be attributed to checkpoint metadata, not to this audit.

---

## Discrepancies between code and documentation

(Code wins in every case.)

1. **Solar angle convention.** Label "Solar Incidence Angle θ" (`app.py:555`) and docstrings `app.py:379`, `depth.py:75`, `82`, `87`, `115` say *incidence*; the formula `depth.py:95` multiplies by `tan(θ)`, which is correct only for *elevation*. `PROJECT_EXPLAINED.md:221`, `225`, `330` say `/ tan(theta_incidence)` / "1/tan". Code: `× tan`. Measured in C1.
2. **Model variant.** `train_crater_yolo.py:2`, `5`, `61`, `111` and `README.md:70` say YOLO11s; `best.pt` is YOLO11m (232 layers, 20,053,779 params). The checkpoint's `train_args` (epochs 200, patience 40, degrees 180, Kaggle 2-GPU, `/kaggle/working/crater_data.yaml`) do not match `train_crater_yolo.py:60-104`, so `best.pt` was not produced by the included script.
3. **"YOLO-only" detection claim.** `detector.py:485-486` "runs YOLO inference exclusively" and `README.md:102` "depend on YOLO detections". In the default synthetic mode `detect_craters` returns ground-truth metadata with confidence 0.98 (`detector.py:501-510`, `app.py:326-338`), and the UI still labels the result YOLO11m (`app.py:895`, `953`, `955`, `968`, `978`).
4. **mAP shown in the log is a literal** (`detector.py:509`, `528`, `575` → `app.py:349`), not a measurement.
5. **"Keep Original Upload Resolution" toggle** is described in `PROJECT_EXPLAINED.md:66-82`, `353-355` (and its cache rule in section 7). It does not exist; resizing is unconditional (`app.py:193-199`). The doc's resize formula also omits the `max(1024, …)` floor (`app.py:196-197`).
6. **"Regenerate Synthetic Surface"** (`app.py:645-646`) regenerates the identical seed-42 image, because no seed is passed and the default is 42 (`synthetic.py:84`); measured bitwise identical.
7. **Shadow segmentation input.** `PROJECT_EXPLAINED.md:132` says CLAHE helps "shadow segmentation"; depth runs on `raw_image`, not the preprocessed image (`app.py:404`, `1085`).
8. **Azimuth described as cosmetic.** The captions at `app.py:559` ("rotates the depth arrow annotation") and `app.py:577` ("used for ROI diagnostics") understate it: φ sets the projection axis and therefore the shadow length and the depth (`depth.py:57-68`). `PROJECT_EXPLAINED.md:335` gets this right.
9. **Gaussian kernel size.** `preprocess.py:59` and the UI panel `app.py:855` claim OpenCV uses `round(σ·4.5)|1`. Measured: OpenCV's automatic kernel for uint8 is larger (σ=4.0 reproduced only by ksize 25; formula says 19).
10. **Synthetic ground truth.** The `synthetic.py:90-93` / `98` docstrings present the generator as supplying crater descriptors for testing, but true depth is not returned (`synthetic.py:134-142`). The "sun angle" there is an azimuth, not an elevation or incidence (`synthetic.py:47`, `85`, `99`).
11. **Heuristic values labelled as measurements:** "Confidence %" (`pathfinder.py:237` → `app.py:1386`), "Obstacles avoided" = count of HAZARD craters (`pathfinder.py:236` → `app.py:1398`), "Depth uncertainty reduced by %" = spread across craters (`depth.py:289-291` → `app.py:1100`).
12. **Pipeline strip** has 8 labels (`ui_components.py:491-500`) for a 9-step workflow (`app.py:44`, `48-58`).
13. **Class name.** The model's only class is named `"0"` (A2), not "crater"; nothing in the docs mentions this.
