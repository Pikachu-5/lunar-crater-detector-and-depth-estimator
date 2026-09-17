### B1 timing (seconds, seed 42, 5 runs after 1 excluded warm-up)

| detection path | stage | mean | stdev (sample) | raw runs |
|---|---|---|---|---|
| yolo | preprocess | 0.0010951600008411333 | 0.0001348113404880606 | 0.0011174999963259324, 0.0009742000111145899, 0.001216100004967302, 0.0012300999951548874, 0.0009378999966429546 |
| yolo | detection | 0.09600478000065778 | 0.008233572070016435 | 0.09142630000133067, 0.09046059999673162, 0.10990760001004674, 0.0910297999944305, 0.09719960000074934 |
| yolo | depth | 0.003724460004013963 | 0.00014460659278408525 | 0.003723000001627952, 0.0037721999979112297, 0.003687600008561276, 0.003919700000551529, 0.0035198000114178285 |
| yolo | terrain_depth_map | 0.0011121800023829564 | 0.00017294827842382298 | 0.000963500002399087, 0.0010268000041833147, 0.0011215000122319907, 0.0010443999926792458, 0.001404700000421144 |
| yolo | terrain_figures | 0.01909291999763809 | 0.004508303801270349 | 0.017049399990355596, 0.017013299991958775, 0.016803500009700656, 0.02714669999841135, 0.01745169999776408 |
| yolo | terrain_total | 0.02020510000002105 | 0.004480150928011467 | 0.018012899992754683, 0.01804009999614209, 0.017925000021932647, 0.028191099991090596, 0.018856399998185225 |
| yolo | scoring | 0.09275941999512724 | 0.007127596732786772 | 0.08773659999133088, 0.08619999999064021, 0.10270129999844357, 0.09774910000851378, 0.08941009998670779 |
| yolo | astar | 0.20358384000137447 | 0.005222635009841932 | 0.20809109999390785, 0.1946304000011878, 0.20395060000009835, 0.20591410000633914, 0.20533300000533927 |
| synthetic_meta | preprocess | 0.00114631999458652 | 0.00023840743543850526 | 0.0015032000083010644, 0.0012812999921152368, 0.0010082999942824244, 0.000977099989540875, 0.0009616999886929989 |
| synthetic_meta | detection | 0.0003237600001739338 | 7.339282771654218e-05 | 0.0002725000085774809, 0.00034389999927952886, 0.00044409999100025743, 0.00028419999580364674, 0.0002741000062087551 |
| synthetic_meta | depth | 0.0033374999999068677 | 0.0006021142202423488 | 0.0028310999914538115, 0.0029476000054273754, 0.003297000002930872, 0.004354200005764142, 0.003257599993958138 |
| synthetic_meta | terrain_depth_map | 0.0009869599976809694 | 0.00023651983795785298 | 0.0008756000024732202, 0.0007212999917101115, 0.000876799997058697, 0.0013011999981245026, 0.0011598999990383163 |
| synthetic_meta | terrain_figures | 0.01618468000087887 | 0.002205833732581738 | 0.014339000001200475, 0.01445149999926798, 0.016330600003129803, 0.01601190000656061, 0.019790399994235486 |
| synthetic_meta | terrain_total | 0.01717163999855984 | 0.0023518571376783567 | 0.015214600003673695, 0.015172799990978092, 0.0172074000001885, 0.01731310000468511, 0.020950299993273802 |
| synthetic_meta | scoring | 0.07926934000279288 | 0.0050745835333115726 | 0.07209520001197234, 0.07949059999373276, 0.07755110001016874, 0.0812659000075655, 0.0859438999905251 |
| synthetic_meta | astar | 0.8680963199964026 | 0.05455342223379367 | 0.7780222000001231, 0.879390499991132, 0.8800883999938378, 0.9268799999990733, 0.8761004999978468 |

Warm-up run (excluded; includes YOLO weight load on first call): {"yolo": {"wall_total": 1.8373853999946732, "preprocess": 0.001479700003983453, "detection": 1.3526540999882855, "depth": 0.04059639999468345, "terrain_depth_map": 0.0008859999943524599, "terrain_figures": 0.11998810000659432, "terrain_total": 0.12087410000094678, "scoring": 0.07473799999570474, "astar": 0.24699390000023413}, "synthetic_meta": {"wall_total": 0.9134437999891816, "preprocess": 0.0009204999951180071, "detection": 0.0002783000090857968, "depth": 0.0030583000043407083, "terrain_depth_map": 0.0015147999947657809, "terrain_figures": 0.01545389999228064, "terrain_total": 0.01696869998704642, "scoring": 0.07243220000236761, "astar": 0.8196579000068596}}

### B2 detection (defaults: CLAHE 2.2/8, sigma 1.2, conf 0.35)

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

### B3 depth tables (seed 42, theta 35, azimuth 35, 1.0 m/px)


#### synthetic_meta

| id | shadow length px | depth m | slope deg | confidence | diameter px | Otsu T | shadow px | not-measurable reason |
|---|---|---|---|---|---|---|---|---|
| CR-01 | not measurable | not measurable | not measurable | None | 34.0 | 148.0 | 386 | no shadow component against the up-sun rim of the crater |
| CR-02 | not measurable | not measurable | not measurable | None | 54.0 | 165.0 | 947 | no shadow component against the up-sun rim of the crater |
| CR-03 | not measurable | not measurable | not measurable | None | 58.0 | 175.0 | 1457 | no shadow component against the up-sun rim of the crater |
| CR-04 | not measurable | not measurable | not measurable | None | 66.0 | 179.0 | 1908 | no shadow component against the up-sun rim of the crater |
| CR-05 | not measurable | not measurable | not measurable | None | 112.0 | 147.0 | 4160 | no shadow component against the up-sun rim of the crater |
| CR-06 | not measurable | not measurable | not measurable | None | 116.0 | 177.0 | 5720 | no shadow component against the up-sun rim of the crater |
| CR-07 | not measurable | not measurable | not measurable | None | 112.0 | 140.0 | 4296 | no shadow component against the up-sun rim of the crater |
| CR-08 | 54.971 | 38.491 | 47.72 | None | 70.0 | 120.0 | 2095 |  |
| CR-09 | not measurable | not measurable | not measurable | None | 34.0 | 179.0 | 445 | no shadow component against the up-sun rim of the crater |

rows=9, skipped (no row)=0, not measurable=8

#### yolo

| id | shadow length px | depth m | slope deg | confidence | diameter px | Otsu T | shadow px | not-measurable reason |
|---|---|---|---|---|---|---|---|---|
| CR-01 | not measurable | not measurable | not measurable | 0.8692474365234375 | 119.0 | 177.0 | 6956 | shadow mask covers 61.6% of the crater area (> 60%): ROI histogram was not bimodal |
| CR-02 | not measurable | not measurable | not measurable | 0.8522818684577942 | 73.0 | 179.0 | 2239 | no shadow component against the up-sun rim of the crater |
| CR-03 | not measurable | not measurable | not measurable | 0.8309051394462585 | 61.0 | 174.0 | 1631 | no shadow component against the up-sun rim of the crater |
| CR-04 | not measurable | not measurable | not measurable | 0.7702561616897583 | 57.0 | 165.0 | 1180 | no shadow component against the up-sun rim of the crater |
| CR-05 | 57.591 | 40.326 | 49.658 | 0.7589907646179199 | 68.5 | 119.0 | 2174 |  |
| CR-06 | not measurable | not measurable | not measurable | 0.7225830554962158 | 103.5 | 146.0 | 4867 | no shadow component against the up-sun rim of the crater |
| CR-07 | not measurable | not measurable | not measurable | 0.7092379927635193 | 37.0 | 179.0 | 548 | no shadow component against the up-sun rim of the crater |
| CR-08 | not measurable | not measurable | not measurable | 0.6920771598815918 | 127.0 | 143.0 | 5574 | no shadow component against the up-sun rim of the crater |
| CR-09 | not measurable | not measurable | not measurable | 0.6427695751190186 | 35.0 | 149.0 | 467 | no shadow component against the up-sun rim of the crater |
| CR-10 | not measurable | not measurable | not measurable | 0.5229388475418091 | 17.0 | 202.0 | 89 | no shadow component against the up-sun rim of the crater |

rows=10, skipped (no row)=0, not measurable=9

#### Not-measurable count across seeds

| seed | detections | rows | not measurable | skipped | reasons |
|---|---|---|---|---|---|
| 42 | yolo | 10 | 9 | 0 | no shadow component against the up-sun rim of the crater; shadow mask covers 61.6% of the crater area (> 60%): ROI histogram was not bimodal |
| 42 | synthetic_meta | 9 | 8 | 0 | no shadow component against the up-sun rim of the crater |
| 7 | yolo | 8 | 6 | 0 | no shadow component against the up-sun rim of the crater |
| 7 | synthetic_meta | 11 | 8 | 0 | no shadow component against the up-sun rim of the crater |
| 123 | yolo | 10 | 8 | 0 | no shadow component against the up-sun rim of the crater |
| 123 | synthetic_meta | 10 | 9 | 0 | no shadow component against the up-sun rim of the crater; shadow mask covers 65.9% of the crater area (> 60%): ROI histogram was not bimodal |

Pooled: yolo 23/28, synthetic_meta 25/30

### B4 scoring (Td 1.8, gear 2.6, density 85 px, 1.0 m/px)

| seed | detections | n | score min | median | max | mean | SAFE | CAUTION | HAZARD | UNKNOWN | overall_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | yolo | 10 | 32.28 | 73.15 | 79.44 | 69.401 | 0 | 0 | 1 | 9 | 69.4 |
| 42 | synthetic_meta | 9 | 32.54 | 71.95 | 75.6 | 67.5611111111111 | 0 | 0 | 1 | 8 | 67.56 |
| 7 | yolo | 8 | 32.32 | 71.555 | 73.19 | 62.47375 | 0 | 0 | 2 | 6 | 62.47 |
| 7 | synthetic_meta | 11 | 30.18 | 69.17 | 73.17 | 59.54727272727273 | 0 | 0 | 3 | 8 | 59.55 |
| 123 | yolo | 10 | 28.55 | 71.9 | 73.5 | 64.268 | 0 | 1 | 1 | 8 | 64.27 |
| 123 | synthetic_meta | 10 | 29.19 | 72.04499999999999 | 73.85 | 67.529 | 0 | 0 | 1 | 9 | 67.53 |

### B5 pathfinding

| seed | detections | route found | path length m | path nodes | alternatives ok | alt nodes | start px | goal px | A* grid (h x w) | goal zone | HAZARD craters crossed (excl. goal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | yolo | True | 274.24 | 83 | 3/3 | [75, 90, 73] | (256, 4) | (492, 100) | 180x180 | UNKNOWN | 0/1 |
| 42 | synthetic_meta | True | 469.58 | 150 | 3/3 | [145, 161, 150] | (256, 4) | (405, 387) | 180x180 | UNKNOWN | 0/1 |
| 7 | yolo | True | 435.38 | 137 | 3/3 | [143, 147, 151] | (256, 4) | (374, 390) | 180x180 | UNKNOWN | 0/2 |
| 7 | synthetic_meta | True | 432.42 | 136 | 3/3 | [142, 146, 150] | (256, 4) | (370, 388) | 180x180 | UNKNOWN | 0/3 |
| 123 | yolo | True | 59.74 | 18 | 3/3 | [17, 30, 25] | (256, 4) | (306, 32) | 180x180 | CAUTION | 0/1 |
| 123 | synthetic_meta | True | 50.94 | 17 | 3/3 | [23, 27, 31] | (256, 4) | (243, 51) | 180x180 | UNKNOWN | 0/1 |

### C1 solar angle sweep — seed 42, detections=synthetic_meta, azimuth 35, 1.0 m/px

| theta deg | tan(theta) | mean shadow length px | mean depth m (code: × tan) | mean depth m (alt: ÷ tan) | n craters | n measurable |
|---|---|---|---|---|---|---|
| 5 | 0.08748866352592401 | 54.971 | 4.809 | 628.318 | 9 | 1 |
| 15 | 0.2679491924311227 | 54.971 | 14.729 | 205.153 | 9 | 1 |
| 25 | 0.4663076581549986 | 54.971 | 25.633 | 117.885 | 9 | 1 |
| 35 | 0.7002075382097097 | 54.971 | 38.491 | 78.506 | 9 | 1 |
| 45 | 0.9999999999999999 | 54.971 | 54.971 | 54.971 | 9 | 1 |
| 55 | 1.4281480067421144 | 54.971 | 78.506 | 38.491 | 9 | 1 |
| 65 | 2.1445069205095586 | 54.971 | 117.885 | 25.633 | 9 | 1 |
| 75 | 3.7320508075688776 | 54.971 | 205.153 | 14.729 | 9 | 1 |
| 85 | 11.430052302761348 | 54.971 | 628.318 | 4.809 | 9 | 1 |

### C1 solar angle sweep — seed 42, detections=yolo, azimuth 35, 1.0 m/px

| theta deg | tan(theta) | mean shadow length px | mean depth m (code: × tan) | mean depth m (alt: ÷ tan) | n craters | n measurable |
|---|---|---|---|---|---|---|
| 5 | 0.08748866352592401 | 57.591 | 5.039 | 658.272 | 10 | 1 |
| 15 | 0.2679491924311227 | 57.591 | 15.432 | 214.934 | 10 | 1 |
| 25 | 0.4663076581549986 | 57.591 | 26.855 | 123.505 | 10 | 1 |
| 35 | 0.7002075382097097 | 57.591 | 40.326 | 82.249 | 10 | 1 |
| 45 | 0.9999999999999999 | 57.591 | 57.591 | 57.591 | 10 | 1 |
| 55 | 1.4281480067421144 | 57.591 | 82.249 | 40.326 | 10 | 1 |
| 65 | 2.1445069205095586 | 57.591 | 123.505 | 26.855 | 10 | 1 |
| 75 | 3.7320508075688776 | 57.591 | 214.934 | 15.432 | 10 | 1 |
| 85 | 11.430052302761348 | 57.591 | 658.272 | 5.039 | 10 | 1 |

### C2 ground truth (seed 42, synthetic_meta detections, theta 35, 1.0 m/px) — replay verified: True

| id | radius px | GT depth_scale (height units) | GT rim-to-floor relief (height units) | est depth_m | est − depth_scale | |est − depth_scale| | est − relief |
|---|---|---|---|---|---|---|---|
| CR-08 | 35 | 50.598478071510016 | 66.78636932373047 | 38.491 | -12.107478071510016 | 12.107478071510016 | -28.29536932373047 |

- replay_verified_bitwise_equal_height_map: True
- craters_skipped_not_measurable: 8
- returned_true_depth_equals_replay: True
- MAE_vs_depth_scale: 12.107478071510016
- mean_signed_error_vs_depth_scale: -12.107478071510016
- n_over: 0
- n_under: 1
- MAE_vs_rim_to_floor_relief: 28.29536932373047
- mean_signed_error_vs_relief: -28.29536932373047
- pearson_est_vs_depth_scale: nan
- pearson_est_vs_radius_px: nan
- pearson_shadow_len_vs_radius_px: nan

### C2b depth estimator validity (seeds 42/7/123 pooled, ground-truth boxes, theta 35, azimuth 35)

- seeds: [42, 7, 123]
- rows: 30
- not_measurable: 25
- pearson_est_vs_true_depth: -0.39861994556559466
- pearson_est_vs_radius_px: 0.6992320833316884
- slope_min_deg: 41.352
- slope_max_deg: 47.72
- slope_std_deg: 2.4825681541500537
- L_equals_(side-1)(cos+sin): '0/5'

### C3 depth threshold sweep — seed 42, detections=synthetic_meta (gear 2.6, density 85)

| Td m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 0.5 | 0 | 0 | 1 |
| 0.6 | 0 | 0 | 1 |
| 0.7 | 0 | 0 | 1 |
| 0.8 | 0 | 0 | 1 |
| 0.9 | 0 | 0 | 1 |
| 1.0 | 0 | 0 | 1 |
| 1.1 | 0 | 0 | 1 |
| 1.2 | 0 | 0 | 1 |
| 1.3 | 0 | 0 | 1 |
| 1.4 | 0 | 0 | 1 |
| 1.5 | 0 | 0 | 1 |
| 1.6 | 0 | 0 | 1 |
| 1.7 | 0 | 0 | 1 |
| 1.8 | 0 | 0 | 1 |
| 1.9 | 0 | 0 | 1 |
| 2.0 | 0 | 0 | 1 |
| 2.1 | 0 | 0 | 1 |
| 2.2 | 0 | 0 | 1 |
| 2.3 | 0 | 0 | 1 |
| 2.4 | 0 | 0 | 1 |
| 2.5 | 0 | 0 | 1 |
| 2.6 | 0 | 0 | 1 |
| 2.7 | 0 | 0 | 1 |
| 2.8 | 0 | 0 | 1 |
| 2.9 | 0 | 0 | 1 |
| 3.0 | 0 | 0 | 1 |
| 3.1 | 0 | 0 | 1 |
| 3.2 | 0 | 0 | 1 |
| 3.3 | 0 | 0 | 1 |
| 3.4 | 0 | 0 | 1 |
| 3.5 | 0 | 0 | 1 |
| 3.6 | 0 | 0 | 1 |
| 3.7 | 0 | 0 | 1 |
| 3.8 | 0 | 0 | 1 |
| 3.9 | 0 | 0 | 1 |
| 4.0 | 0 | 0 | 1 |
| 4.1 | 0 | 0 | 1 |
| 4.2 | 0 | 0 | 1 |
| 4.3 | 0 | 1 | 0 |
| 4.4 | 0 | 1 | 0 |
| 4.5 | 0 | 1 | 0 |
| 4.6 | 0 | 1 | 0 |
| 4.7 | 0 | 1 | 0 |
| 4.8 | 0 | 1 | 0 |
| 4.9 | 0 | 1 | 0 |
| 5.0 | 0 | 1 | 0 |

### C3 gear span sweep — seed 42, detections=synthetic_meta (Td 1.8, density 85)

| gear m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 1.0 | 0 | 0 | 1 |
| 1.1 | 0 | 0 | 1 |
| 1.2 | 0 | 0 | 1 |
| 1.3 | 0 | 0 | 1 |
| 1.4 | 0 | 0 | 1 |
| 1.5 | 0 | 0 | 1 |
| 1.6 | 0 | 0 | 1 |
| 1.7 | 0 | 0 | 1 |
| 1.8 | 0 | 0 | 1 |
| 1.9 | 0 | 0 | 1 |
| 2.0 | 0 | 0 | 1 |
| 2.1 | 0 | 0 | 1 |
| 2.2 | 0 | 0 | 1 |
| 2.3 | 0 | 0 | 1 |
| 2.4 | 0 | 0 | 1 |
| 2.5 | 0 | 0 | 1 |
| 2.6 | 0 | 0 | 1 |
| 2.7 | 0 | 0 | 1 |
| 2.8 | 0 | 0 | 1 |
| 2.9 | 0 | 0 | 1 |
| 3.0 | 0 | 0 | 1 |
| 3.1 | 0 | 0 | 1 |
| 3.2 | 0 | 0 | 1 |
| 3.3 | 0 | 0 | 1 |
| 3.4 | 0 | 0 | 1 |
| 3.5 | 0 | 0 | 1 |
| 3.6 | 0 | 0 | 1 |
| 3.7 | 0 | 0 | 1 |
| 3.8 | 0 | 0 | 1 |
| 3.9 | 0 | 0 | 1 |
| 4.0 | 0 | 0 | 1 |
| 4.1 | 0 | 0 | 1 |
| 4.2 | 0 | 0 | 1 |
| 4.3 | 0 | 0 | 1 |
| 4.4 | 0 | 0 | 1 |
| 4.5 | 0 | 0 | 1 |
| 4.6 | 0 | 0 | 1 |
| 4.7 | 0 | 0 | 1 |
| 4.8 | 0 | 0 | 1 |
| 4.9 | 0 | 0 | 1 |
| 5.0 | 0 | 0 | 1 |

Full 46×41 grid distinct (SAFE, CAUTION, HAZARD) outcomes: [(0, 0, 1), (0, 1, 0)]; corners: {'td=0.5,gear=1.0': (0, 0, 1), 'td=0.5,gear=5.0': (0, 0, 1), 'td=5.0,gear=1.0': (0, 0, 1), 'td=5.0,gear=5.0': (0, 1, 0)}

### C3 depth threshold sweep — seed 42, detections=yolo (gear 2.6, density 85)

| Td m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 0.5 | 0 | 0 | 1 |
| 0.6 | 0 | 0 | 1 |
| 0.7 | 0 | 0 | 1 |
| 0.8 | 0 | 0 | 1 |
| 0.9 | 0 | 0 | 1 |
| 1.0 | 0 | 0 | 1 |
| 1.1 | 0 | 0 | 1 |
| 1.2 | 0 | 0 | 1 |
| 1.3 | 0 | 0 | 1 |
| 1.4 | 0 | 0 | 1 |
| 1.5 | 0 | 0 | 1 |
| 1.6 | 0 | 0 | 1 |
| 1.7 | 0 | 0 | 1 |
| 1.8 | 0 | 0 | 1 |
| 1.9 | 0 | 0 | 1 |
| 2.0 | 0 | 0 | 1 |
| 2.1 | 0 | 0 | 1 |
| 2.2 | 0 | 0 | 1 |
| 2.3 | 0 | 0 | 1 |
| 2.4 | 0 | 0 | 1 |
| 2.5 | 0 | 0 | 1 |
| 2.6 | 0 | 0 | 1 |
| 2.7 | 0 | 0 | 1 |
| 2.8 | 0 | 0 | 1 |
| 2.9 | 0 | 0 | 1 |
| 3.0 | 0 | 0 | 1 |
| 3.1 | 0 | 0 | 1 |
| 3.2 | 0 | 0 | 1 |
| 3.3 | 0 | 0 | 1 |
| 3.4 | 0 | 0 | 1 |
| 3.5 | 0 | 0 | 1 |
| 3.6 | 0 | 0 | 1 |
| 3.7 | 0 | 0 | 1 |
| 3.8 | 0 | 0 | 1 |
| 3.9 | 0 | 0 | 1 |
| 4.0 | 0 | 0 | 1 |
| 4.1 | 0 | 0 | 1 |
| 4.2 | 0 | 0 | 1 |
| 4.3 | 0 | 0 | 1 |
| 4.4 | 0 | 0 | 1 |
| 4.5 | 0 | 1 | 0 |
| 4.6 | 0 | 1 | 0 |
| 4.7 | 0 | 1 | 0 |
| 4.8 | 0 | 1 | 0 |
| 4.9 | 0 | 1 | 0 |
| 5.0 | 0 | 1 | 0 |

### C3 gear span sweep — seed 42, detections=yolo (Td 1.8, density 85)

| gear m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 1.0 | 0 | 0 | 1 |
| 1.1 | 0 | 0 | 1 |
| 1.2 | 0 | 0 | 1 |
| 1.3 | 0 | 0 | 1 |
| 1.4 | 0 | 0 | 1 |
| 1.5 | 0 | 0 | 1 |
| 1.6 | 0 | 0 | 1 |
| 1.7 | 0 | 0 | 1 |
| 1.8 | 0 | 0 | 1 |
| 1.9 | 0 | 0 | 1 |
| 2.0 | 0 | 0 | 1 |
| 2.1 | 0 | 0 | 1 |
| 2.2 | 0 | 0 | 1 |
| 2.3 | 0 | 0 | 1 |
| 2.4 | 0 | 0 | 1 |
| 2.5 | 0 | 0 | 1 |
| 2.6 | 0 | 0 | 1 |
| 2.7 | 0 | 0 | 1 |
| 2.8 | 0 | 0 | 1 |
| 2.9 | 0 | 0 | 1 |
| 3.0 | 0 | 0 | 1 |
| 3.1 | 0 | 0 | 1 |
| 3.2 | 0 | 0 | 1 |
| 3.3 | 0 | 0 | 1 |
| 3.4 | 0 | 0 | 1 |
| 3.5 | 0 | 0 | 1 |
| 3.6 | 0 | 0 | 1 |
| 3.7 | 0 | 0 | 1 |
| 3.8 | 0 | 0 | 1 |
| 3.9 | 0 | 0 | 1 |
| 4.0 | 0 | 0 | 1 |
| 4.1 | 0 | 0 | 1 |
| 4.2 | 0 | 0 | 1 |
| 4.3 | 0 | 0 | 1 |
| 4.4 | 0 | 0 | 1 |
| 4.5 | 0 | 0 | 1 |
| 4.6 | 0 | 0 | 1 |
| 4.7 | 0 | 0 | 1 |
| 4.8 | 0 | 0 | 1 |
| 4.9 | 0 | 0 | 1 |
| 5.0 | 0 | 0 | 1 |

Full 46×41 grid distinct (SAFE, CAUTION, HAZARD) outcomes: [(0, 0, 1), (0, 1, 0)]; corners: {'td=0.5,gear=1.0': (0, 0, 1), 'td=0.5,gear=5.0': (0, 0, 1), 'td=5.0,gear=1.0': (0, 0, 1), 'td=5.0,gear=5.0': (0, 1, 0)}

### C4 shipped estimator on ray-cast shadow scenes (seeds 42/7/123, elevation 20, azimuth 35)

- sun_elevation_deg: 20.0
- ground_truth: {'craters_detected': 30, 'measured': 19, 'not_measurable': 11, 'coverage': 0.6333333333333333, 'rejection_reasons': {'MEASURED': 19, 'shadow mask covers 63.9% of the crater area (> 60%)': 2, 'shadow mask covers 63.8% of the crater area (> 60%)': 1, 'shadow mask covers 61.4% of the crater area (> 60%)': 1, 'shadow mask covers 76.7% of the crater area (> 60%)': 1, 'shadow mask covers 81.7% of the crater area (> 60%)': 1, 'shadow mask covers 68.7% of the crater area (> 60%)': 1, 'shadow mask covers 72.2% of the crater area (> 60%)': 1, 'shadow mask covers 67.3% of the crater area (> 60%)': 1, 'shadow mask covers 72.6% of the crater area (> 60%)': 1, 'shadow mask covers 60.9% of the crater area (> 60%)': 1}, 'pearson_est_vs_true_depth': 0.831326618316996, 'radius_only_r_on_measured_subset': 0.8291224217811244, 'shadow_minus_radius_only': 0.002204196535871561, 'partial_r_est_vs_depth_given_radius': 0.28108858022093575, 'radius_only_r_on_all_detected': 0.8636430715431034, 'pearson_est_vs_radius_px': 0.935883721633158, 'slope_min_deg': 19.167, 'slope_max_deg': 28.091, 'slope_std_deg': 2.421941150459294}
- yolo: {'craters_detected': 26, 'measured': 23, 'not_measurable': 3, 'coverage': 0.8846153846153846, 'rejection_reasons': {'MEASURED': 23, 'shadow mask covers 62.2% of the crater area (> 60%)': 1, 'shadow mask covers 74.1% of the crater area (> 60%)': 1, 'shadow mask covers 71.2% of the crater area (> 60%)': 1}, 'pearson_est_vs_true_depth': 0.8346772504506572, 'radius_only_r_on_measured_subset': 0.8730120392220289, 'shadow_minus_radius_only': -0.03833478877137164, 'partial_r_est_vs_depth_given_radius': 0.11117415494370325, 'radius_only_r_on_all_detected': 0.8764990870379938, 'pearson_est_vs_radius_px': 0.9338808312128057, 'slope_min_deg': 18.496, 'slope_max_deg': 32.841, 'slope_std_deg': 3.002132295149623}

### C5 non-crater terrain score: constant 82 vs measured roughness (YOLO boxes)

| seed | mode | score map min | mean | max | std | path length m | path nodes | goal px |
|---|---|---|---|---|---|---|---|---|
| 42 | constant_82 | 32.279998779296875 | 77.04219055175781 | 81.99793243408203 | 6.5487260818481445 | 274.24 | 83 | (492, 100) |
| 42 | roughness | 0.0 | 76.51841735839844 | 99.99482727050781 | 27.896108627319336 | 273.52 | 83 | (492, 100) |
| 7 | constant_82 | 32.31999969482422 | 73.98768615722656 | 81.99990844726562 | 10.067140579223633 | 435.38 | 137 | (374, 390) |
| 7 | roughness | 0.0 | 75.88999938964844 | 99.99400329589844 | 26.94521713256836 | 455.35 | 138 | (374, 390) |
| 123 | constant_82 | 28.549999237060547 | 76.57450103759766 | 82.0 | 7.733401298522949 | 59.74 | 18 | (306, 32) |
| 123 | roughness | 0.0 | 77.44236755371094 | 99.99996948242188 | 26.74580192565918 | 63.03 | 20 | (306, 32) |

Zone counts are unchanged by this: they come from per-crater scoring, not the terrain map. seed 42: {'safe': 0, 'caution': 0, 'hazard': 1, 'unknown': 9}; seed 7: {'safe': 0, 'caution': 0, 'hazard': 2, 'unknown': 6}; seed 123: {'safe': 0, 'caution': 1, 'hazard': 1, 'unknown': 8}
