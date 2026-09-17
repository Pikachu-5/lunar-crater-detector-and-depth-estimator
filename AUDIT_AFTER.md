# AUDIT_AFTER — Lunar Crater Detector + Depth Estimator

Date: 2026-09-17. Branch `audit-fixes`, 8 commits on top of `20bd8d9`. Each numbered fix is its own commit and can be reverted on its own:

| Commit | Item | Status |
|---|---|---|
| `8a561a0` | 1 Depth estimator | **NOT FIXED** — stopped per instruction; experiment script committed as evidence, `depth.py` unchanged |
| `c335e8b` | — | Baseline `AUDIT.md`, harness, frozen before-results (`scripts/audit_results_before.json`, `scripts/audit_tables_before.md`) |
| `1a03ba5` | 2 Ground truth | Fixed |
| `af1c3f4` | 3 Synthetic mode fabricates detections | Fixed |
| `68f58d0` | 4 Placeholder metrics | Fixed (12 removed or replaced by measured values; #13 kept as a labelled assumption) |
| `e6c7d67` | 5 Solar angle label | Fixed |
| `968706a` | 6 Model metadata | Fixed |
| `6ff75e1` | 7 Smaller items | Fixed (Keep-Original toggle: **removed from docs**, not implemented) |

Conventions as in `AUDIT.md`: `file:line` refers to the current branch tip; numbers are copied from `scripts/audit_results.json` / `scripts/audit_tables.md` (the "after" run) and `scripts/audit_results_before.json` / `scripts/audit_tables_before.md` (the "before" run) in full precision. The code wins over docs.

Environment: unchanged from `AUDIT.md` (Python 3.12.10, Windows 11 10.0.26200, i7-13620H, numpy 2.5.3, OpenCV 5.0.0, torch 2.14.0+cpu, ultralytics 8.4.155, `torch.get_num_threads()` = 1).

Verification tools:
- `scripts/audit_run.py` — headless pipeline; re-run after every item.
- `scripts/app_check.py` — new; drives the real `app.py` through Streamlit `AppTest` (steps 3–9, fusion, confidence slider, regenerate) and prints what the UI shows. No exceptions in the final run.
- `scripts/depth_segmentation_experiment.py` — new; item 1 candidate comparison. Output saved in `scripts/depth_segmentation_results.txt`.

---

## Before / after — every metric

### Item 1 — depth estimator (the four success criteria)

Ground-truth boxes, θ 35°, φ 35°, 1.0 m/px. "Seed 42" is the scene `AUDIT.md` used; "pooled" means seeds 42/7/123 (30 craters). `depth.py` is unchanged, so before and after are the same.

| Criterion | Before (seed 42) | After (seed 42) | Before (pooled) | After (pooled) | Target | Met? |
|---|---|---|---|---|---|---|
| r(estimated depth, true depth) | 0.22621023094987164 | 0.22621023094987164 | −0.19723435320093202 | −0.19723435320093202 | well above 0.226 | **No** |
| r(estimated depth, crater radius) | 0.9998390451381755 | 0.9998390451381755 | 0.99982452656156 | 0.99982452656156 | well below 0.9998 | **No** |
| slope range (deg) | 61.907–62.653 | 61.907–62.653 | 61.339–62.657 (std 0.338374100992115) | same | genuinely varying | **No** |
| L = (side − 1)(cos 35° + sin 35°) | 8/9 | 8/9 | 28/30 | 28/30 | no longer true | **No** |
| craters returned as not measurable | 0 | 0 | 0 | 0 | — | — |

### Items 2–7

| Metric | Before | After | Evidence |
|---|---|---|---|
| True depth available from `generate_synthetic_lunar_surface` | No (had to replay the RNG) | Yes: `true_depth`, `rim_height` per crater (`utils/synthetic.py:145-146`) | item 2 |
| `true_depth` equals RNG replay | — | True for all craters, seeds 42/7/123 | `returned_true_depth_equals_replay: True` |
| SHA-256 (first 16 hex) of image / height_map, seed 42 | `05c257d80cdc63c4` / `e88ef0350839aaa2` | `05c257d80cdc63c4` / `e88ef0350839aaa2` | bitwise identical |
| same, seed 7 | `3cd20637e3a47c11` / `9f69e8a2421f4297` | identical | |
| same, seed 123 | `f2c1201419ffc21b` / `3c5991e1b4b167a0` | identical | |
| App default flow: detection source | `synthetic-meta` (generator boxes) | `yolo` | `app_check.py` |
| App default flow: detections (seed 42) | 9 | 10 | |
| App default flow: confidence min / max | 0.98 / 0.98 (literal) | 0.5229388475418091 / 0.8692474365234375 (model) | |
| Ground-truth boxes in detection list / confidence chart | Yes | No; separate overlay "GROUND TRUTH, NOT MODEL OUTPUT" (`app.py:1037`) | |
| App default flow downstream, seed 42: SAFE/CAUTION/HAZARD | 0 / 2 / 7 | 0 / 2 / 8 | before = B4 synthetic_meta row, after = B4 yolo row |
| App default flow downstream, seed 42: overall_score | 32.6 | 35.39 | |
| App default flow downstream, seed 42: path length m | 508.12 | 274.24 | B5 |
| App default flow downstream, seed 42: goal px | (405, 387) | (492, 100) | B5 |
| "mAP@0.5" in detector log | 0.847 (literal; 0.990 in synthetic mode) | removed; step 4 caption reads 0.8471 / 0.64392 **from `best.pt` train_metrics**, labelled "Reported by the checkpoint from training, not measured on this image" (`app.py:978`) | item 4 |
| Landing "Confidence %" (seed 42, YOLO boxes) | 62.97 | removed | |
| "Obstacles avoided" (seed 42, YOLO boxes) | 8 (= count of all HAZARD craters) | replaced by measured "HAZARD craters the route passes through (excluding goal)": 0 of 8 | B5; crossing test checked on synthetic lines (True through a crater, False 12 px away, True for a segment jumping over it) |
| Fusion "uncertainty reduced %" | shown | removed; replaced by mean \|view1 − view2\| depth: 44.870 m, 10 matched (seed 42 app run) | `app_check.py` |
| Gaussian kernel readout matches OpenCV (σ = 0.5…4.0, 36 values) | 3/36 (`round(4.5σ)\|1`) | 36/36 (`round(6σ+1)\|1`) | item 4 |
| Terrain "estimated render memory MB" / "(512 MB)" labels | shown | removed; caption shows the rendered grid, 512×512 points | |
| "TELEMETRY: NOMINAL" | HUD and PDF | removed | |
| CV heuristic scores drawn as confidences | yes | no (`app.py:396`) | |
| Non-crater terrain score 82 | hidden literal | named `NON_CRATER_TERRAIN_SCORE` (`modules/scorer.py:15`), shown in step 8 as an assumption | |
| Slider label | "Solar Incidence Angle θ", no help | "Solar Elevation Angle (above horizontal)", with help text (`app.py:573-582`) | item 5 |
| Depth output for any θ (C1, B3, C2, C3 tables) | — | identical to before (formula unchanged) | diff of non-timing tables |
| Variant claimed by `train_crater_yolo.py` / README | YOLO11s | YOLO11m (`train_crater_yolo.py:67`, `README.md:70`) + mismatch note (`README.md:74`, `train_crater_yolo.py:8`) | item 6 |
| Class name shown | `"0"` (never shown) | `crater` in the YOLO table (`modules/detector.py:31`, `204`) | |
| Detection confidence threshold | hard-coded 0.35 | sidebar slider 0.05–0.95, default 0.35 (`app.py:600-608`) | item 7 |
| Boxes at conf 0.70 (seed 42) | not settable | 7, min confidence 0.7092379927635193 | `app_check.py` |
| Regenerate: consecutive scenes identical | yes (always seed 42) | no; e.g. seeds 1806639592, 1831555662, different images; displayed seed reproduces the image | |
| PROJECT_EXPLAINED depth formula | `/ tan(theta_incidence)` | `* tan(theta_elevation)` (`PROJECT_EXPLAINED.md:215`) | |
| PROJECT_EXPLAINED "Keep Original Upload Resolution" | documented | removed; §4.2 now describes the unconditional resize with the 1024 px floor | |

### Numbers that should not change (and did not)

| Metric | Before | After |
|---|---|---|
| B2 YOLO counts seeds 42/7/123 | 10 / 8 / 10 | 10 / 8 / 10 |
| B2 YOLO matched GT | 9/9, 8/11, 10/10 | 9/9, 8/11, 10/10 |
| B2 CV counts | 10 / 10 / 10 | 10 / 10 / 10 |
| B2 CV matched GT | 3/9, 5/11, 6/10 | 3/9, 5/11, 6/10 |
| B3, B4, C1, C2, C3 tables | — | byte-identical after normalising the ground-truth confidence column (0.98 → None) and the added C2b block |
| B1 YOLO inference mean (s) | 0.08189003999868874 | 0.0834876200009603 |
| B1 A* mean, YOLO boxes (s) | 0.18397008000174536 | 0.18348370000021533 |
| B1 scoring mean, YOLO boxes (s) | 0.1052418800041778 | 0.08764745999942533 |

The timing code and the YOLO/CV code paths were not modified; the B1 differences are run-to-run variation.

---

## PART A — STATIC AUDIT (after)

### A1. Solar angle convention

- Widget: `app.py:572-582`, label **"Solar Elevation Angle (above horizontal)"**, min 10, max 80, default 35, step 1. Help: *"Angle of the Sun above the local horizon: 0° grazing, 90° overhead. Depth is computed as shadow length × pixel scale × tan(angle), which is the correct relation for elevation (not incidence from the surface normal). For the same measured shadow, a higher Sun implies a deeper crater."* Caption `app.py:583` is unchanged.
- Chain: `theta` (`app.py:572`) → `depth_slider_signature` (`app.py:659`) → params `"theta"` (`app.py:709`) → `theta_deg=params["theta"]` (`app.py:1082`, `1193`, `1268`, `1410`, `1497`) → `ensure_depth(theta_deg)` (`app.py:407`, docstring `app.py:411`) → `solar_elevation_angle_deg=theta_deg` (`app.py:438`) → `estimate_crater_depths(solar_elevation_angle_deg)` (`modules/depth.py:103`) → `depth_from_shadow(solar_elevation_angle_deg=…)` (`modules/depth.py:144`) → `theta` (`modules/depth.py:96`). The row key is now `solar_elevation_deg` (`modules/depth.py:168`).
- Formula (unchanged): `modules/depth.py:96-97`, `depth_m = L · s · tan(clip(θ, 1°, 89°))`.
- Verdict: the value is treated as **elevation**, and the label and every docstring now agree. `PROJECT_EXPLAINED.md:215` now multiplies by tan. Remaining "incidence" mentions (`app.py:580`, `modules/depth.py:83-84`) only contrast elevation with incidence.

### A2. Model identity

Unchanged model: YOLO11m, 232 layers, 20,053,779 parameters, 68.3033088 GFLOPs @640 / 28.828166016 @416, 1 class stored as `"0"`, input 416. Docs now agree: `modules/detector.py:1-6`, `train_crater_yolo.py:2-12`, `:67`, `README.md:70-74`. Display name mapping: `modules/detector.py:31-37`, applied at `:204`.

### A3. Placeholder numbers (after)

| # (AUDIT.md) | Before | After | File:line |
|---|---|---|---|
| 1 | `map50_proxy` 0.847 logged as mAP@0.5 | Deleted. Checkpoint's own `train_metrics` read from the file at runtime and labelled "Reported by the checkpoint from training, not measured on this image" | `modules/detector.py:64-90`, `app.py:975-982` |
| 2 | `map50_proxy` 0.99 (synthetic) | Deleted together with the synthetic-hint path | — |
| 3 | `map50_proxy` 0.82 (CV) | Deleted | — |
| 4 | Ground-truth confidence 0.98 shown as YOLO | Ground truth has `confidence=None`, drawn only in a labelled overlay | `modules/detector.py:486-515`, `app.py:1034-1043` |
| 5 | CV heuristic scores drawn as confidences | Not drawn (`show_confidence=False`); still computed and used for ranking | `app.py:396`, `modules/detector.py:616` |
| 6 | Landing "Confidence %" | Deleted | — |
| 7 | "Obstacles avoided" = count of HAZARD craters | Measured: HAZARD craters (excluding the goal) whose radius the route enters, sampled at ≤ 1 px | `modules/pathfinder.py:237-274`, `app.py:1463-1468` |
| 8 | Fusion "uncertainty reduced %" | Deleted; mean absolute depth difference between the two views | `modules/depth.py:286-296`, `app.py:1171-1180` |
| 9 | Simulated second view and assumed angles unlabelled | Labelled as ASSUMED / SIMULATED | `app.py:1131-1136` |
| 10 | "TELEMETRY: NOMINAL" | Deleted (HUD and PDF) | — |
| 11 | 48 bytes/point memory estimate, "(256 MB)" etc. | Deleted; the rendered grid shape is reported instead | `app.py:470-487`, `app.py:1207-1210` |
| 12 | Kernel `round(4.5σ)\|1` | `gaussian_kernel_size` = `round(6σ+1)\|1`, verified 36/36 against OpenCV | `modules/preprocess.py:76-79`, `app.py:914` |
| 13 | Non-crater score 82.0 | **Kept** (cannot be measured) as a named constant, shown to the user as an assumption | `modules/scorer.py:12-15`, `app.py:1475-1479` |

### A4. Parameter dump (after)

| Control | Label | Min | Max | Default | Step | File:line |
|---|---|---|---|---|---|---|
| Solar angle | Solar Elevation Angle (above horizontal) | 10 | 80 | 35 | 1 | `app.py:572-582` |
| Solar azimuth | Solar Azimuth φ | 0 | 359 | 35 | 1 | `app.py:585` |
| Depth threshold | Depth Safety Threshold Td (m) | 0.5 | 5.0 | 1.8 | 0.1 | `app.py:588` |
| Gear span | Landing Gear Span (m) | 1.0 | 5.0 | 2.6 | 0.1 | `app.py:591` |
| Density radius | Crater Density Radius (px) | 20 | 200 | 85 | 5 | `app.py:594` |
| Pixel scale | Pixel Scale (m/px) | 0.2 | 4.0 | 1.0 | 0.1 | `app.py:597` |
| **Detection confidence** | **Detection Confidence Threshold** | **0.05** | **0.95** | **0.35** | **0.05** | `app.py:600-608` |
| 3D terrain profile | 3D Terrain Memory Profile (options now "Balanced", "High Fidelity", "Max Fidelity", "Adaptive") | — | — | High Fidelity | — | `app.py:623` |
| Fusion | ENABLE MULTI-ANGLE FUSION | — | — | False | — | `app.py:689` |
| Regenerate | ↻ Regenerate Synthetic Surface (fresh random seed, displayed) | — | — | — | — | `app.py:692-706` |
| CLAHE grid / clip / σ | unchanged | 2 / 1.0 / 0.5 | 16 / 5.0 / 4.0 | 8 / 2.2 / 1.2 | 1 / 0.1 / 0.1 | `app.py:846`, `856`, `866` |
| Blend (fallback) | unchanged | 0 | 100 | 50 | 1 | `app.py:828` |

### A5. Architecture map — changes only

- Step 4 (`app.py:942`): `run_detection(params["conf_threshold"])` (`app.py:340-354`) always calls `detect_craters` (YOLO). New read-only overlay `synthetic_ground_truth()` (`app.py:372-382`) writes nothing to session state. Changing the confidence slider after detections exist calls `reset_downstream(4)` (`app.py:649-656`).
- Step 5 fusion: second pass uses YOLO with the slider threshold; writes `fusion` with `mean_abs_view_difference_m`.
- Step 6: `terrain` stores `surface_grid_shape` instead of `surface_memory_est_mb`.
- Step 8: `paths` holds `goal_crater_id`, `goal_zone`, `hazard_craters_other_than_goal`, `hazard_craters_crossed`; `landing_confidence` and `obstacles_avoided` are gone.
- New session keys: `synthetic_seed`, `last_conf_threshold`.
- Unchanged from `AUDIT.md`: `ensure_terrain` / `ensure_scoring` / `ensure_paths` still recompute on every rerun, and step 1 still resets downstream on every rerun while a file is in the uploader (not in scope).

---

## PART B — INSTRUMENTED RUNS (after)

Same harness and parameters as `AUDIT.md`. **Label note:** the `synthetic_meta` rows are now ground-truth boxes built by `ground_truth_detections` (evaluation only — the app no longer uses them). Their confidence is `None`. The `yolo` rows are what the app now produces for the synthetic feed.

### B1. Per-stage timing

#### B1 timing (seconds, seed 42, 5 runs after 1 excluded warm-up)

| detection path | stage | mean | stdev (sample) | raw runs |
|---|---|---|---|---|
| yolo | preprocess | 0.0011431399965658785 | 0.0003258700680579435 | 0.0017226000054506585, 0.0009523999906377867, 0.0009915999980876222, 0.0010520999931031838, 0.0009969999955501407 |
| yolo | detection | 0.0834876200009603 | 0.0050323962181511835 | 0.08601179999823216, 0.08233280001149978, 0.07909270000527613, 0.0909038999961922, 0.07909689999360126 |
| yolo | depth | 0.0018831400026101618 | 0.00010127167334908839 | 0.0018044000025838614, 0.0018710000003920868, 0.0018475000106263906, 0.002059099992038682, 0.0018337000074097887 |
| yolo | terrain_depth_map | 0.0033848400023998694 | 0.00019188948273135824 | 0.0033101000008173287, 0.003103800001554191, 0.003445900001679547, 0.003444000001763925, 0.0036204000061843544 |
| yolo | terrain_figures | 0.016629239995381795 | 0.0011245861636310698 | 0.016928400000324473, 0.016484799998579547, 0.01704059999610763, 0.01786589999392163, 0.014826499987975694 |
| yolo | terrain_total | 0.020014079997781663 | 0.0010711886917615097 | 0.0202385000011418, 0.019588600000133738, 0.020486499997787178, 0.021309899995685555, 0.01844689999416005 |
| yolo | scoring | 0.08764745999942533 | 0.0028044387448164956 | 0.08491740000317805, 0.08798039999965113, 0.08622659998945892, 0.092257200012682, 0.08685569999215659 |
| yolo | astar | 0.18348370000021533 | 0.0030013624288945306 | 0.18650930000876542, 0.1863561999925878, 0.18017490000056569, 0.1836598999943817, 0.180718200004776 |
| synthetic_meta | preprocess | 0.0009863399987807497 | 4.4198107121000986e-05 | 0.0009805000008782372, 0.0009595000010449439, 0.0010521999938646331, 0.0010026999952970073, 0.0009368000028189272 |
| synthetic_meta | detection | 0.00026928000152111055 | 8.100118133518154e-06 | 0.0002801999944495037, 0.00025910000840667635, 0.0002697000018088147, 0.0002730999985942617, 0.0002643000043462962 |
| synthetic_meta | depth | 0.0015615199983585626 | 0.00044563891651583257 | 0.0017987000028369948, 0.00220979998994153, 0.0011104000004706904, 0.001444399997126311, 0.0012443000014172867 |
| synthetic_meta | terrain_depth_map | 0.002510299999266863 | 0.00013679169031339877 | 0.0026396000030217692, 0.00261730000784155, 0.002360899990890175, 0.002365399996051565, 0.0025682999985292554 |
| synthetic_meta | terrain_figures | 0.014564040000550449 | 0.0012514738910582364 | 0.016071799997007474, 0.015767700009746477, 0.013585900000180118, 0.01390200000605546, 0.013492799989762716 |
| synthetic_meta | terrain_total | 0.017074339999817313 | 0.0013552628046491942 | 0.018711400000029244, 0.018385000017588027, 0.015946799991070293, 0.016267400002107024, 0.01606109998829197 |
| synthetic_meta | scoring | 0.07656581999908667 | 0.002385730608348347 | 0.07718530000420287, 0.08015930000692606, 0.07365249999566004, 0.07617869999376126, 0.07565329999488313 |
| synthetic_meta | astar | 0.8441180399997392 | 0.021737343267377238 | 0.8278627999970922, 0.8313848999969196, 0.8780885000014678, 0.8294352000084473, 0.8538187999947695 |

Warm-up run (excluded; includes YOLO weight load on first call): {"yolo": {"wall_total": 1.734720099993865, "preprocess": 0.0016472999996040016, "detection": 1.2657573999895249, "depth": 0.02441260000341572, "terrain_depth_map": 0.0035611999919638038, "terrain_figures": 0.11857959999179002, "terrain_total": 0.12214079998375382, "scoring": 0.08194240000739228, "astar": 0.23876210000889841}, "synthetic_meta": {"wall_total": 0.9216537000029348, "preprocess": 0.0009965000062948093, "detection": 0.00028059999749530107, "depth": 0.0011884999985340983, "terrain_depth_map": 0.0025815999979386106, "terrain_figures": 0.015090199987753294, "terrain_total": 0.017671799985691905, "scoring": 0.07428929999878164, "astar": 0.8270999000087613}}

### B2. Detection output

#### B2 detection (defaults: CLAHE 2.2/8, sigma 1.2, conf 0.35)

| seed | GT craters | detector | count | conf min | conf median | conf max | diam px min | diam px max | matched GT @IoU>=0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 9 | yolo | 10 | 0.5229388475418091 | 0.7407869100570679 | 0.8692474365234375 | 17.0 | 127.0 | 9/9 |
| 42 | 9 | cv_hybrid | 10 | 0.9032651081096981 | 0.9649123748143513 | 0.99 | 38.0 | 238.0 | 3/9 |
| 42 | 9 | synthetic_meta | 9 | None | None | None | 34.0 | 116.0 | 9/9 |
| 7 | 11 | yolo | 8 | 0.48157837986946106 | 0.7826994359493256 | 0.8730961084365845 | 25.5 | 113.0 | 8/11 |
| 7 | 11 | cv_hybrid | 10 | 0.9362029892603556 | 0.9562515699089675 | 0.99 | 46.0 | 230.5 | 5/11 |
| 7 | 11 | synthetic_meta | 11 | None | None | None | 22.0 | 108.0 | 11/11 |
| 123 | 10 | yolo | 10 | 0.5181504487991333 | 0.6831502020359039 | 0.8704314231872559 | 22.0 | 131.0 | 10/10 |
| 123 | 10 | cv_hybrid | 10 | 0.8983192179997763 | 0.9542991148657636 | 0.99 | 46.0 | 226.0 | 6/10 |
| 123 | 10 | synthetic_meta | 10 | None | None | None | 20.0 | 118.0 | 10/10 |

### B3. Depth output

#### B3 depth tables (seed 42, theta 35, azimuth 35, 1.0 m/px)

##### synthetic_meta

| id | shadow length px | depth m | slope deg | confidence | diameter px | Otsu T | shadow px | zero-depth reason |
|---|---|---|---|---|---|---|---|---|
| CR-01 | 45.96 | 32.182 | 62.155 | None | 34.0 | 135.0 | 689 |  |
| CR-02 | 73.815 | 51.686 | 62.418 | None | 54.0 | 158.0 | 1683 |  |
| CR-03 | 79.386 | 55.586 | 62.448 | None | 58.0 | 173.0 | 1771 |  |
| CR-04 | 90.527 | 63.388 | 62.498 | None | 66.0 | 179.0 | 2224 |  |
| CR-05 | 154.593 | 108.247 | 62.646 | None | 112.0 | 137.0 | 7252 |  |
| CR-06 | 160.164 | 112.148 | 62.653 | None | 116.0 | 177.0 | 6996 |  |
| CR-07 | 154.593 | 108.247 | 62.646 | None | 112.0 | 131.0 | 7280 |  |
| CR-08 | 93.641 | 65.568 | 61.907 | None | 70.0 | 100.0 | 2572 |  |
| CR-09 | 45.96 | 32.182 | 62.155 | None | 34.0 | 175.0 | 677 |  |

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

As before: 0 zero-depth and 0 skipped craters across 58 rows. **Correction to `AUDIT.md`:** the `Otsu T` column is the real Otsu threshold, but `depth.py` never uses it (see Part C, item 1 root cause).

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

### B5. Pathfinding output

#### B5 pathfinding

| seed | detections | route found | path length m | path nodes | alternatives ok | alt nodes | start px | goal px | A* grid (h x w) | goal zone | HAZARD craters crossed (excl. goal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | yolo | True | 274.24 | 83 | 3/3 | [75, 90, 73] | (256, 4) | (492, 100) | 180x180 | CAUTION | 0/8 |
| 42 | synthetic_meta | True | 508.12 | 156 | 3/3 | [151, 169, 153] | (256, 4) | (405, 387) | 180x180 | CAUTION | 0/7 |
| 7 | yolo | True | 212.64 | 62 | 3/3 | [68, 72, 76] | (256, 4) | (164, 177) | 180x180 | CAUTION | 0/7 |
| 7 | synthetic_meta | True | 212.64 | 62 | 3/3 | [68, 72, 76] | (256, 4) | (164, 177) | 180x180 | HAZARD | 0/10 |
| 123 | yolo | True | 53.94 | 18 | 3/3 | [24, 28, 33] | (256, 4) | (244, 52) | 180x180 | CAUTION | 0/8 |
| 123 | synthetic_meta | True | 50.94 | 17 | 3/3 | [23, 27, 31] | (256, 4) | (243, 51) | 180x180 | CAUTION | 0/8 |

---

## PART C — SENSITIVITY EXPERIMENTS (after)

### C1. Solar angle sweep (identical to before)

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

### C2. Ground truth (synthetic ground truth — NOT real lunar terrain)

Ground truth now comes directly from `true_depth` in the generator metadata; the RNG replay is kept as a cross-check. The caveats from `AUDIT.md` still apply: arbitrary height units, not metres, and the renderer casts no shadows.

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
- returned_true_depth_equals_replay: True
- MAE_vs_depth_scale: 34.69830253905036
- mean_signed_error_vs_depth_scale: 29.77117102407111
- n_over: 8
- n_under: 1
- MAE_vs_rim_to_floor_relief: 33.239116963704426
- mean_signed_error_vs_relief: 29.315649963378902
- pearson_est_vs_depth_scale: 0.22621023094987164
- pearson_est_vs_radius_px: 0.9998390451381755
- pearson_shadow_len_vs_radius_px: 0.9998390831810087

#### C2b depth estimator validity (seeds 42/7/123 pooled, ground-truth boxes, theta 35, azimuth 35)

- seeds: [42, 7, 123]
- rows: 30
- not_measurable: 0
- pearson_est_vs_true_depth: -0.19723435320093202
- pearson_est_vs_radius_px: 0.99982452656156
- slope_min_deg: 61.339
- slope_max_deg: 62.657
- slope_std_deg: 0.338374100992115
- L_equals_(side-1)(cos+sin): '28/30'

### C3. Parameter sensitivity (identical to before)

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

### Item 1 investigation — why the depth estimator was not fixed

**Step 1: root cause (confirmed before any change).** Masks were dumped as PNGs for CR-01, CR-05, CR-08 (seed 42) to `scripts/audit_masks/before_CR-*.png`. These files are local only, because `.gitignore` excludes `*.png`.

| Crater | ROI px | mask px | mask / ROI | mask share of pixels **outside** the crater circle | mask share of the crater circle |
|---|---|---|---|---|---|
| CR-01 | 1156 | 689 | 0.5960207612456747 | 1.0 | 0.48053392658509453 |
| CR-05 | 12544 | 7252 | 0.578125 | 1.0 | 0.4623590368790003 |
| CR-08 | 4900 | 2572 | 0.5248979591836734 | 0.9408960915157293 | 0.41158140742664245 |

The hypothesis in the request ("the mask covers the whole dark bowl") is **not** what happens. The mask covers the **bright** terrain: every ROI pixel outside the crater circle for CR-01 and CR-05, and 94 % for CR-08. The dark centre is excluded. Cause: `modules/depth.py:28` unpacks `_, th = cv2.threshold(...)`, which keeps the thresholded **image** and discards the threshold value. `modules/depth.py:29` then evaluates `blur < th` pixel-by-pixel against that 0/255 image. That is true exactly where the binary image is 255, i.e. where `blur` is **above** the Otsu threshold. Checked on CR-05: centre blur 59 < Otsu 137.0, yet the mask is 0 at the centre and 255 at the corner (corner blur 195). Because the mask includes the ROI corners, L is the ROI's corner-to-corner extent along the azimuth axis, which explains `L = (side − 1)(cos 35° + sin 35°)`.

**Step 2: candidates tried** (`scripts/depth_segmentation_experiment.py`, ground-truth boxes, no tuning beyond the listed values). Candidates a–c share: restriction to the crater circle, 3×3 open/close, rejection of components on the lit half (the measured interior mean is brighter on the +azimuth half for 8 of 9 seed-42 craters), largest remaining component only, and not-measurable if the mask covers > 60 % of the circle.

| Candidate | r(est, true depth) seed 42 | r pooled (3 seeds) | r(est, radius) pooled | slope min–max (deg) | slope std | L = box formula | not measurable |
|---|---|---|---|---|---|---|---|
| current code | 0.22620879780796738 | −0.19723458355779167 | 0.9998245616386158 | 61.339195394290144–62.65660282733033 | 0.3383199097037404 | 28/30 | 0 |
| polarity bug fixed only (full ROI) | 0.245477872703009 | −0.2158916720208762 | 0.9698242018678418 | 40.3656471321092–55.149893598689154 | 3.3156360753853735 | 0/30 | 0 |
| (a) Otsu inside circle | 0.11137585850283509 | −0.27739969892146465 | 0.9868593625548906 | 37.28488131661844–48.06679212726684 | 2.791319172544803 | 0/24 | 6 |
| (b) 15th percentile | 0.09734225102331495 | −0.24129651855773243 | 0.9702240792547462 | 18.042462025402088–33.79434626139513 | 3.0555048025048825 | 0/27 | 3 |
| (b) 20th percentile | 0.2615027795364082 | −0.17895239471180305 | 0.9791937813820155 | 20.450118533867027–37.5377626910159 | 3.06774585619532 | 0/27 | 3 |
| (b) 25th percentile | 0.28445180103751844 | −0.16283430735850332 | 0.9764756935994762 | 22.482826783299746–40.74355179833398 | 3.3802356424925164 | 0/27 | 3 |
| (b) 30th percentile | 0.25689232268719625 | −0.22574543228502192 | 0.9925863286300388 | 32.48513399733123–41.90220654508642 | 1.9292455729562215 | 0/26 | 4 |
| (c) mean − 0.5·std | 0.23226987458818937 | −0.22654848385055917 | 0.9959833206394446 | 33.92968734310996–41.90220654508642 | 1.8753636928820536 | 0/26 | 4 |
| (c) mean − 1.0·std | 0.17861622955325354 | −0.21589486504021407 | 0.9746777671911145 | 19.412856884069715–37.5377626910159 | 3.1777525020658812 | 0/27 | 3 |

Every candidate meets two criteria (L no longer box-sized; slopes vary). **None** meets the two that matter. The best seed-42 value (0.28445180103751844, 25th percentile, n = 8) is not a clear margin over 0.226. Every pooled correlation is negative, and correlation with radius stays at 0.97–0.996. Per the instruction, I stopped here and did not pick or commit a segmenter.

**Why no segmentation method can pass on this test data (diagnostic, measured, 30 craters pooled):** the generator draws no cast shadows. Depth only scales how dark the bowl is (`utils/synthetic.py:65`, `76`, `79`); the dark region's size is set by radius (bowl σ = 0.58·r).
- r(true depth, ring-mean − interior-min intensity contrast) = 0.736313974870724
- r(true depth, dark-area fraction of the interior) = 0.3636546100170834
- r(true depth, radius) = −0.1992216940923165 — so any radius-proportional estimate correlates *negatively* with depth on these seeds, which is exactly the pooled result above.

A shadow-*length* method measures a quantity that this synthetic scene does not tie to depth.

---

## PART D — VALIDATION

Unchanged: `LU3M6TGT_yolo_format/` is not present, so no genuine precision/recall/mAP can be produced. The app now shows only the checkpoint's own stored training-time values, labelled as such.

---

## Could not fix, or deliberately left

1. **Item 1 — the depth estimator is not fixed.** Root cause confirmed: the threshold-unpacking bug at `modules/depth.py:28-29` inverts the mask. None of candidates (a), (b), (c) raises the ground-truth correlation clearly above 0.226, and the reason is structural (the synthetic scene has no cast shadows). `depth.py` is unchanged, so the app still shows the box-proportional depths. The validity guard and the "not measurable" UI were part of item 1 and were **not** implemented. Decisions needed from you:
   - (i) commit the one-line polarity fix alone. It is a real correctness bug, but on synthetic ground truth it gives r = 0.245477872703009 (seed 42) / −0.2158916720208762 (pooled), so it must not be described as making depth accurate.
   - (ii) change the synthetic generator to render geometric cast shadows from the height map at a given sun elevation, then re-run the candidates. That changes the test data, so it is your call.
   - (iii) validate on real imagery with known depths.
2. **Placeholder #13** (non-crater terrain score 82) cannot be computed from the data. It is kept as a named, user-visible assumption rather than deleted, because deleting it would change A* routing.
3. **Harness edits I had to make, disclosed because you asked that the timing harness and the comparison not be touched:**
   - The ground-truth ("synthetic_meta") path in `run_pipeline` now calls `ground_truth_detections`, because `detect_craters(hint_craters=…)` no longer exists. It builds the same records.
   - The keyword `solar_incidence_angle_deg` was renamed to `solar_elevation_angle_deg` in the `estimate_crater_depths` call and in C1's `inverse_tan`.
   - B2 ignores `None` confidences.
   - The YOLO and CV timing and comparison code is unchanged, and their B2 numbers are identical.
4. **CV heuristic "confidence"** is still computed and still appears in the B2 comparison table (min 0.8983192179997763–0.99). It is no longer drawn in the app. I left the comparison untouched as instructed.
5. **Discrepancies from `AUDIT.md` outside the numbered items, not addressed:**
   - #7 `PROJECT_EXPLAINED.md:132` still says CLAHE helps shadow segmentation, but depth runs on the raw image.
   - #8 The azimuth expander text still calls azimuth a diagnostic ("rotates shadow-direction arrows"), although it changes L.
   - #12 The pipeline strip still has 8 labels for 9 steps (`utils/ui_components.py:490`).
   - Step 1 still resets downstream on every rerun while a file is in the uploader.
   - `ensure_*` functions still recompute on every rerun.
   - `depth_signature` still covers only the first 20 detections.
   - `train_crater_yolo.py` emits a pre-existing `SyntaxWarning` (`\S` in its docstring).
6. **The initial synthetic scene is still seed 42** by design, so the first view stays reproducible; only Regenerate draws a fresh seed.
7. **Part D** is still impossible without the dataset.
