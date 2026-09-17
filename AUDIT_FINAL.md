# AUDIT_FINAL — Lunar Crater Detector + Depth Estimator

Date: 2026-09-18. Branch `audit-fixes`, merged into `main` (not pushed). Every numbered item is one commit and can be reverted on its own.

| Commit | Item | Status |
|---|---|---|
| `6020a75` | 1 Polarity bug | Fixed (defect corrected; accuracy unchanged, as expected) |
| `494d6bf` | 2 Validity guard / "not measurable" | Fixed |
| `ea43a7b` | 3 Ray-cast shadows in the generator | Fixed; **no depth approach clearly wins, so nothing was tuned** |
| `744c689` | 4 Non-crater terrain score | Fixed (measured roughness replaces the constant) |
| `73acd2b` | 5 Remaining audit findings | Fixed |

Verification tools (all re-run after every item):
- `scripts/audit_run.py` → `scripts/audit_results.json`, `scripts/audit_tables.md`
- `scripts/app_check.py` → drives the real `app.py` through Streamlit `AppTest`
- `scripts/shadow_sanity.py` → generator sanity checks (item 3)
- `scripts/depth_segmentation_experiment.py` → `scripts/depth_segmentation_results.txt`
- `scripts/mask_coverage.py` → per-crater mask coverage

Baseline for the "before" column: `scripts/audit_results_before.json` / `scripts/audit_tables_before.md` (frozen at commit `c335e8b`).

**Scene note, important when reading every number below.** The audit harness (B1–B5, C1–C3) still runs on the *legacy shadow-free* generator so earlier results stay reproducible. That scene contains no cast shadows at all, so after item 3 corrected the shadow-side convention the estimator correctly reports almost every crater there as not measurable. The estimator's real behaviour is in **C4**, which runs on ray-cast scenes, and in the app, which now generates ray-cast scenes.

---

## Every metric, before and after

### Depth estimator

| Metric | Before (AUDIT_AFTER) | After | Where |
|---|---|---|---|
| Mask polarity | selects pixels ABOVE the Otsu threshold (bright terrain) | selects pixels BELOW it, inside the crater circle | `modules/depth.py:56-79` |
| Mask coverage outside the crater circle (CR-01 / CR-05 / CR-08, seed 42) | 100% / 100% / 94.09% | 0/257, 0/2701, 0/1049 pixels — none | `scripts/mask_coverage.py` |
| Mask coverage as a fraction of the crater circle (CR-01 / CR-05 / CR-08) | 0.4805 / 0.4624 / 0.4116 (of a mask that was mostly outside the crater) | 0.41935483870967744 / 0.4226353753936808 / 0.5440145416774864 | same |
| L = (box side − 1)(cos 35° + sin 35°) | 28/30 craters | 0/30 | C2b |
| Measured L, CR-01 / CR-05 / CR-08 | 45.96 / 154.593 / 93.641 px (box geometry) | 20.89092720960057 / 75.94236472488836 / 54.970712348300154 px | `scripts/mask_coverage.py` |
| r(estimated depth, true depth), pooled, legacy scene | −0.19723435320093202 | −0.2502917291387287 after item 1; not computable after item 2 (n=2) | C2b |
| r(estimated depth, radius), pooled, legacy scene | 0.99982452656156 | 0.9844454878682677 after item 1 | C2b |
| Slope spread, pooled, legacy scene | 61.339–62.657° (std 0.338374100992115) | 37.285–48.067° (std 2.743234303801912) after item 1 | C2b |
| **r(estimated depth, true depth) on ray-cast scenes, YOLO boxes** | could not be measured (no cast shadows existed) | **0.8995673087987358** (18 of 26 craters) | C4 |
| r(estimated depth, radius) on ray-cast scenes, YOLO boxes | — | 0.9694402063001051 | C4 |
| Slope spread on ray-cast scenes, YOLO boxes | — | 18.496–26.648° (std 2.1958330437838023) | C4 |

### Validity / "not measurable" (item 2)

| Metric | Before | After |
|---|---|---|
| Craters returned as not measurable | 0 (a failed measurement returned 0.0 m) | legacy scene: **25/28** YOLO boxes, **28/30** ground-truth boxes; ray-cast scene: **8/26** YOLO boxes (18 measured), 28/30 ground-truth boxes |
| Zero-depth craters carved into the terrain model | yes | excluded entirely |
| Zone for an unmeasured crater | scored as depth 0 → tended to SAFE | `UNKNOWN`, never SAFE |
| Zone summary keys | safe/caution/hazard | safe/caution/hazard/**unknown** |
| Seed 42, YOLO boxes, zone counts (legacy scene) | 0 / 2 / 8 | 0 / 0 / 0 / 10 unknown |
| Seed 7, YOLO boxes | 0 / 1 / 7 | 0 / 0 / 2 / 6 unknown |
| Seed 123, YOLO boxes | 0 / 2 / 8 | 0 / 1 / 0 / 9 unknown |
| Overall score, seed 42 YOLO (legacy) | 35.39 | 73.08 (no depth penalty is applied when depth is unknown) |
| Depth/slope shown in UI for a failed measurement | `0.0` m | `not measurable`, plus a reason column and an "N of M craters not measurable" banner |
| PDF/report cells | `0.0` | `not measurable` |

### Generator (item 3)

| Metric | Before | After |
|---|---|---|
| Cast shadows | none | ray-marched occlusion over the height field (`cast_shadow_mask`) |
| r(shadow length, true depth) | −0.16159495492145676 | **0.8474407494773224** (elevation 20°), 0.5232095191607821 (elevation 35°) |
| r(shadow length, radius) | 0.9967409204684531 | 0.8983969957226128 (elevation 20°) |
| r(true depth, radius) | −0.1992216940923165 | 0.8636430715431034 (depth now drawn as radius × U(0.35, 0.75)) |
| Legacy mode reproducibility | — | image and height_map SHA-256 identical for seeds 42/7/123 and the default call |
| Shadow side convention in `depth.py` | assumed anti-solar | corrected to the up-sun rim, from measurement (0 of 29 craters anti-solar) |

### Terrain score (item 4)

| Seed | Mode | score map min / mean / max / std | Path length m | Path nodes |
|---|---|---|---|---|
| 42 | constant 82 | 67.73999786376953 / 77.93296813964844 / 81.99793243408203 / 3.6558799743652344 | 274.24 | 83 |
| 42 | roughness | 0.0 / 76.8014907836914 / 99.99482727050781 / 27.787796020507812 | 273.52 | 83 |
| 7 | constant 82 | 32.31999969482422 / 73.98768615722656 / 81.99990844726562 / 10.067140579223633 | 435.38 | 137 |
| 7 | roughness | 0.0 / 75.88999938964844 / 99.99400329589844 / 26.94521713256836 | 455.35 | 138 |
| 123 | constant 82 | 41.63999938964844 / 77.80872344970703 / 82.0 / 4.4072418212890625 | 59.74 | 18 |
| 123 | roughness | 0.0 / 78.11991882324219 / 99.99996948242188 / 26.460041046142578 | 63.03 | 20 |

**SAFE/CAUTION/HAZARD counts do not change with item 4, and cannot:** zones come from per-crater depth/diameter/density scoring, not from the terrain map. What changes is the surface A* plans over, so routes shift (seed 7: 435.38 m → 455.35 m).

### Unchanged on purpose

| Metric | Before | After |
|---|---|---|
| B2 YOLO counts, seeds 42/7/123 | 10 / 8 / 10 | 10 / 8 / 10 |
| B2 YOLO matched ground truth @IoU≥0.5 | 9/9, 8/11, 10/10 | 9/9, 8/11, 10/10 |
| B2 CV counts / matched | 10/10/10; 3/9, 5/11, 6/10 | identical |
| B1 YOLO inference mean | 0.08189003999868874 s | 0.0834876200009603 s (run-to-run variation; code untouched) |
| C1 solar sweep, C3 sensitivity | — | unchanged where craters remain measurable |

---

## Generator sanity checks (item 3)

Run: `scripts/shadow_sanity.py`. Azimuth 35°, seeds 42/7/123.

**1. Shadows lengthen as the sun drops** (seed 42):

| Sun elevation | Shadow pixels (fraction of image) | Mean per-crater shadow length | Craters with shadow |
|---|---|---|---|
| 15° | 0.10688400268554688 | 52.71304003421827 px | 9 |
| 30° | 0.042331695556640625 | 30.047000053076967 px | 9 |
| 45° | 0.00186920166015625 | 17.302123226731524 px | 5 |
| 60° | 3.814697265625e-06 | — | 0 |
| 75° | 0.0 | — | 0 |

**2. Which side the shadow falls on.** Projection of shadow pixels onto the direction towards the sun, in crater radii: mean **+0.5335978777716529**, min +0.24212375938144248, max +0.7992861543737081, and **0 of 29 craters** have their interior shadow on the anti-solar side. Inside a crater the shadow lies against the **up-sun rim** and extends down-sun across the floor; only terrain *outside* the rim is shadowed anti-solar. `depth.py` had assumed anti-solar, so its side test was inverted; item 3 corrects it and cites this measurement.

**3. Shadow length now tracks depth:**

| Scene | n | r(shadow length, true depth) | r(shadow length, radius) | r(true depth, radius) |
|---|---|---|---|---|
| legacy shadow-free | 30 | −0.16159495492145676 | 0.9967409204684531 | −0.1992216940923165 |
| ray-cast, elevation 20° | 30 | **0.8474407494773224** | 0.8983969957226128 | 0.8636430715431034 |
| ray-cast, elevation 35° | 29 | 0.5232095191607821 | 0.46333362636382297 | 0.871465211036753 |

No analytic `L = d/tan(θ)` fill is used anywhere: `cast_shadow_mask` marches a ray per pixel and tests occlusion against the height field, so shadow geometry emerges from the terrain.

---

## Depth approaches on the new shadow data

Ground-truth boxes (detection error excluded), pooled over seeds 42/7/123, estimator θ = the scene's true sun elevation. `partial` = correlation with true depth after removing the shared radius trend. **A radius-only estimator would score r_gt = r(true depth, radius) = 0.8636 on this scene**, so raw `r_gt` alone is not evidence.

| Method | n measured | seed42 r_gt | pooled r_gt | r(est, radius) | partial | slope range (std) |
|---|---|---|---|---|---|---|
| shipped (with validity guards) | 2/30 | — | — | — | — | 20.46–20.96 (0.250) |
| shipped, area guard disabled (diagnostic) | 2/30 | — | — | — | — | 20.46–20.96 (0.250) |
| a0_full (no circle restriction) | 30/30 | 0.6809962023635832 | 0.6386696275860081 | 0.6905352049523888 | 0.11599460437788794 | 20.46–43.66 (7.301) |
| (a) Otsu inside circle | 18/30 | 0.8895107060223886 | 0.7913936514414457 | 0.9176248646313153 | 0.27656299082098945 | 19.17–28.09 (2.416) |
| (b) 15th percentile | 25/30 | 0.23187317387065343 | 0.5215603402413567 | 0.7583735288908031 | −0.5040749790174073 | 2.50–17.48 (4.537) |
| (b) 20th percentile | 26/30 | 0.8488314099107275 | 0.7746891919981528 | 0.9185517371884986 | −0.20343587973595326 | 2.15–20.16 (3.916) |
| (b) 25th percentile | 27/30 | 0.8872325018331964 | 0.8211903626304242 | 0.8114449867600044 | **0.4083662197626925** | 1.16–22.47 (4.464) |
| (b) 30th percentile | 27/30 | 0.3226126104593227 | 0.5924227495583875 | 0.6587532016199084 | 0.04198498051816391 | 1.04–23.29 (4.691) |
| (c) mean − 0.5σ | 28/30 | 0.9199145730058701 | **0.8721562895417037** | 0.9292661777667388 | 0.32747935936426675 | 14.15–30.92 (3.229) |
| (c) mean − 1.0σ | 17/30 | −0.19844917595675138 | 0.18245758087910938 | 0.40090538324343744 | −0.22868991734017144 | 2.96–24.05 (6.327) |

On the legacy shadow-free scene for comparison: `a0_full` pooled r_gt −0.2158916720208762; every circle-restricted candidate measures ≤ 5 of 30 craters, so its correlations rest on 4–5 points and mean nothing.

**Verdict: no approach clearly wins, so nothing was kept or tuned.** The best raw correlation (c, mean − 0.5σ: 0.8722) barely exceeds the 0.8636 a radius-only estimator would score, and after removing the radius trend the best partial correlation is 0.41 (b, 25th percentile) on 27 points. The shipped method from items 1–2 stays as it is.

---

## Not-measurable counts

| Scene | Boxes | Craters | Measured | Not measurable | Dominant reasons |
|---|---|---|---|---|---|
| legacy (shadow-free) | YOLO | 28 | 3 | **25** | no shadow component against the up-sun rim (the scene has no cast shadow) |
| legacy (shadow-free) | ground truth | 30 | 2 | **28** | 24 × no component against the up-sun rim; 3 × touches ROI boundary; 1 × >60% area |
| ray-cast, elevation 20° | YOLO | 26 | 18 | **8** | 5 × touches ROI boundary; 3 × >60% of crater area |
| ray-cast, elevation 20° | ground truth | 30 | 2 | **28** | 16 × touches ROI boundary; 12 × >60% of crater area |

Per seed on the legacy scene (ground-truth boxes) immediately after item 2, before the item-3 side correction: seed 42 → 2/9, seed 7 → 6/11, seed 123 → 3/10 (11 of 30 pooled).

---

## What remains unfixed, and why

1. **The two validity guards reject valid deep-crater measurements when the box hugs the crater.** On ray-cast scenes with ground-truth boxes the shipped estimator measures only 2 of 30: 16 craters fail "component touches the ROI boundary" and 12 fail ">60% of the crater area". Both are the thresholds you specified in item 2, and both are legitimately triggered here: a bowl-shaped crater at low sun has a shadow that genuinely fills most of the bowl and reaches the edge of a box drawn exactly around the crater. With YOLO boxes, which are looser, the same code measures 18 of 26. I did **not** tune either threshold. The fix is a design decision for you:
   - expand the ROI beyond the detection box (e.g. 1.3×) so "touches the boundary" means genuinely truncated rather than "the box is tight"; or
   - test truncation against the crater circle instead of the ROI rectangle; or
   - raise the area fraction for bowl-shaped craters, accepting that a shadow may legitimately cover most of the interior.
   A diagnostic run with the area guard disabled still measures 2 of 30, so the boundary rule is the binding constraint.
2. **Depth accuracy is still only demonstrated correlationally, and mostly against radius.** On ray-cast data the shipped estimator reaches r = 0.8996 with true depth, but r = 0.9694 with radius, and true depth itself correlates 0.8636 with radius in that scene. The estimator is consistent with a depth-tracking method but this data cannot cleanly separate depth from size. A scene with depth drawn independently of radius would need a sun elevation chosen per crater, since the self-shadowing and containment windows both scale with depth/radius.
3. **The legacy shadow-free scene is now reported as almost entirely not measurable** (25/28 and 28/30). That is the correct answer for a scene with no cast shadows, but it means the B-series tables in `scripts/audit_tables.md` mostly show UNKNOWN zones. The B-series deliberately still runs on the legacy scene so the earlier numbers stay comparable.
4. **The harness's timed pipeline still uses the constant terrain score.** Item 4's roughness map is measured in an untimed block (C5) so the B1 timing stage keeps measuring identical work, as you asked. The app uses the roughness map.
5. **The CV heuristic "confidence"** is still computed and still appears in the B2 comparison table; it is not shown in the app. The comparison was left untouched as instructed.
6. **No genuine detection metrics.** `LU3M6TGT_yolo_format/` is still absent, so precision/recall/mAP cannot be measured; the app shows only the checkpoint's stored training values, labelled as such.
7. **Multi-angle fusion remains a simulated second view** (same image plus a brightness gradient, assumed θ+15°/φ+25°). It is labelled in the UI but is not a real second observation.

---

# Addendum — ROI expansion and an honest radius-only comparison

Date: 2026-09-18, commit `7b40b65` on `main`. Scripts: `scripts/roi_comparison.py` (→ `scripts/roi_comparison_results.txt`), `scripts/audit_run.py` block C4.

## What changed

`estimate_crater_depths` now crops a square analysis window of ±1.4 × crater radius about the crater centre (`ROI_WINDOW_RADIUS_FACTOR`, `modules/depth.py`), clipped to image bounds, instead of using the detection box. The crater circle used for masking keeps the detection radius — only the observable window grows. **The >60% area threshold and the boundary rule are unchanged**; the boundary test now means "the shadow left the window", not "the box was drawn tight".

## Ray-cast data: coverage and failure reasons

| Boxes | ROI | Measured / detected | Coverage | Boundary failures | >60% area failures |
|---|---|---|---|---|---|
| ground truth | box (old) | 2 / 30 | 0.06666666666666667 | 16 | 12 |
| ground truth | window (new) | 19 / 30 | 0.6333333333333333 | **0** | 11 |
| YOLO | box (old) | 18 / 26 | 0.6923076923076923 | 5 | 3 |
| YOLO | window (new) | 23 / 26 | 0.8846153846153846 | **0** | 3 |

The boundary guard no longer fires at all. Every remaining failure is the >60% area guard, which is the threshold you asked to leave alone: a bowl crater at low sun genuinely has a shadow covering 60.9%–81.7% of its interior.

## Correlations and slope spread

| Boxes | ROI | r(estimate, true depth) | r(estimate, radius) | Slope spread (std) |
|---|---|---|---|---|
| ground truth | box | not computable (n=2) | not computable | 20.456750414379716–20.95693833543677 (0.2500939605285275) |
| ground truth | window | 0.831327912297354 | 0.9358898950821479 | 19.16727773985974–28.091120345969784 (2.421973871469177) |
| YOLO | box | 0.8995703292607853 | 0.9694400346997122 | 18.49593827811613–26.647717209120316 (2.1957625636829534) |
| YOLO | window | 0.8346774071496863 | 0.9338802524982774 | 18.49593827811614–32.84101468693158 (3.0021074162556483) |

## The honest comparison

Radius-only = depth predicted from crater radius alone, ignoring the shadow entirely. Pearson r of any linear function of radius equals r(radius, true depth), so that is the baseline. Everything below is computed **on exactly the craters that were measured**, and coverage is stated so a high correlation on an easy subset cannot pass as a result.

| Boxes | ROI | Coverage | Shadow r | Radius-only r (same craters) | Difference | Partial r given radius (p) | Radius-only r (all detected) |
|---|---|---|---|---|---|---|---|
| ground truth | box | 2/30 | n/a | n/a | n/a | n/a | 0.8636430715431034 |
| ground truth | window | 19/30 | 0.831327912297354 | 0.8291224217811244 | **+0.0022054905162296468** | 0.28108224687631833 (p = 0.2585243788167261) | 0.8636430715431034 |
| YOLO | box | 18/26 | 0.8995703292607853 | 0.8682422753601329 | **+0.03132805390065241** | 0.47537604794193633 (p = 0.05378833089122232) | 0.8764990870379938 |
| YOLO | window | 23/26 | 0.8346774071496863 | 0.8730120392220289 | **−0.038334632072342556** | 0.1111774806114964 (p = 0.6223227140301517) | 0.8764990870379938 |

### Headline

**The shadow measurement does not beat a radius-only estimator on the same craters.** The best case is +0.0022 (ground-truth boxes) and the configuration with the best coverage is −0.0383, i.e. *worse* than simply predicting depth from crater size. No partial correlation reaches significance (p = 0.26, 0.054, 0.62).

The `yolo/box` row is exactly the trap you flagged: at 18/26 coverage it looked like the best configuration (+0.031, partial 0.475, p = 0.054). Widening the window to measure 23/26 of the same craters drops it to −0.038 with partial 0.111. The five craters the boundary guard used to reject were the ones where the shadow carried information beyond size; with them included, the shadow signal is no better than the radius. The earlier headline in this document — "r = 0.8996 on ray-cast data" — should be read with its radius-only baseline of 0.8682 next to it, and with the coverage caveat.

So: the ROI fix is a real improvement in **coverage** (2/30 → 19/30 and 18/26 → 23/26) and it makes the boundary guard mean what it was meant to mean. It is **not** evidence that the depth estimator works. On this test data, shadow-length depth estimation is statistically indistinguishable from guessing depth from crater diameter.

## Side effects on the legacy shadow-free scene

Pooled not-measurable counts improve slightly but the scene stays mostly unmeasurable, which is the correct answer where no shadow was ever cast: YOLO boxes 25/28 → 23/28, ground-truth boxes 28/30 → 25/30. C2b (legacy, ground-truth boxes) now has 5 measured craters instead of 2: r(estimate, true depth) −0.39861994556559466, r(estimate, radius) 0.6992320833316884, slope 41.352–47.72°.

## Still unfixed

1. The >60% area guard still rejects 11 of 30 ground-truth-box craters and 3 of 26 YOLO-box craters on ray-cast data, all legitimately deep bowls. Unchanged as instructed; raising it is a design decision.
2. Depth accuracy remains unproven — see the headline above.
3. Everything else listed in the previous "What remains unfixed" section stands.
