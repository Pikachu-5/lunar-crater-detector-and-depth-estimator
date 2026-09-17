### B1 timing (seconds, seed 42, 5 runs after 1 excluded warm-up)

| detection path | stage | mean | stdev (sample) | raw runs |
|---|---|---|---|---|
| yolo | preprocess | 0.0011646200000541285 | 0.00010154243734201113 | 0.0011976999958278611, 0.00125119999574963, 0.00119270000141114, 0.000988400002825074, 0.0011931000044569373 |
| yolo | detection | 0.10759858000092208 | 0.014217936965677046 | 0.1117938000097638, 0.0974726999993436, 0.11386019999918062, 0.12544979999074712, 0.08941640000557527 |
| yolo | depth | 0.003481920002377592 | 0.0012289349064759695 | 0.002902899999753572, 0.0029484999977285042, 0.005679300011252053, 0.0029858000052627176, 0.002893099997891113 |
| yolo | terrain_depth_map | 0.0009271399962017312 | 0.00018088622105867443 | 0.0008760999917285517, 0.0008316999883390963, 0.001248700005817227, 0.0008544000011170283, 0.000824799994006753 |
| yolo | terrain_figures | 0.022437699997681193 | 0.005679135667863785 | 0.02038180000090506, 0.01972169999498874, 0.03257809999922756, 0.019462900003418326, 0.02004399998986628 |
| yolo | terrain_total | 0.023364839993882926 | 0.005859046724835175 | 0.02125789999263361, 0.020553399983327836, 0.03382680000504479, 0.020317300004535355, 0.020868799983873032 |
| yolo | scoring | 0.10870105999929365 | 0.01676391484907683 | 0.10136770000099204, 0.1007744999951683, 0.13865439999790397, 0.1024397000001045, 0.10026900000229944 |
| yolo | astar | 0.274636340001598 | 0.03575091386416351 | 0.2541851999994833, 0.2543177000043215, 0.286125800004811, 0.3324085000058403, 0.24614449999353383 |
| synthetic_meta | preprocess | 0.0013647399988258257 | 0.00025111828646382504 | 0.0011909000022569671, 0.0016780000005383044, 0.0015977000002749264, 0.0011904999992111698, 0.001166599991847761 |
| synthetic_meta | detection | 0.00038867999392095954 | 6.396410965156928e-05 | 0.00033579999580979347, 0.0003366999881109223, 0.0004039999912492931, 0.0003758999955607578, 0.000490999998874031 |
| synthetic_meta | depth | 0.002634840001701377 | 0.00034367161498356845 | 0.0023106000007828698, 0.0025639000086812302, 0.0025602999958209693, 0.00251869999920018, 0.003220700004021637 |
| synthetic_meta | terrain_depth_map | 0.0007807800022419543 | 0.0001058737755295987 | 0.0007976000051712617, 0.0007206000009318814, 0.0007852999988244846, 0.0006585000082850456, 0.0009418999979970977 |
| synthetic_meta | terrain_figures | 0.019760980000137353 | 0.005496799233294489 | 0.015793200000189245, 0.016143100001499988, 0.018453900003805757, 0.029246099991723895, 0.019168600003467873 |
| synthetic_meta | terrain_total | 0.020541760002379304 | 0.005447100853293529 | 0.016590800005360506, 0.01686370000243187, 0.01923920000263024, 0.02990460000000894, 0.02011050000146497 |
| synthetic_meta | scoring | 0.08685359999653883 | 0.007259291911751448 | 0.08108079999510664, 0.08510979999846313, 0.08443659999466036, 0.09954189999552909, 0.08409889999893494 |
| synthetic_meta | astar | 0.9417038000014145 | 0.03571516971678679 | 0.9167384000029415, 0.942651700010174, 0.9746620999940205, 0.9781718000012916, 0.8962949999986449 |

Warm-up run (excluded; includes YOLO weight load on first call): {"yolo": {"wall_total": 1.8659466999961296, "preprocess": 0.0013087999977869913, "detection": 1.311019499989925, "depth": 0.02459479999379255, "terrain_depth_map": 0.0007557999924756587, "terrain_figures": 0.13986939999449532, "terrain_total": 0.14062519998697098, "scoring": 0.0884037000068929, "astar": 0.29992000000493135}, "synthetic_meta": {"wall_total": 1.0086291000043275, "preprocess": 0.0011661000025924295, "detection": 0.00034779999987222254, "depth": 0.0026205999893136322, "terrain_depth_map": 0.0006752000044798478, "terrain_figures": 0.01621329999761656, "terrain_total": 0.016888500002096407, "scoring": 0.08885759999975562, "astar": 0.8985859000094933}}

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
| CR-01 | not measurable | not measurable | not measurable | None | 34.0 | 135.0 | 377 | no shadow component against the up-sun rim of the crater |
| CR-02 | not measurable | not measurable | not measurable | None | 54.0 | 158.0 | 947 | no shadow component against the up-sun rim of the crater |
| CR-03 | not measurable | not measurable | not measurable | None | 58.0 | 173.0 | 1457 | no shadow component against the up-sun rim of the crater |
| CR-04 | not measurable | not measurable | not measurable | None | 66.0 | 179.0 | 1908 | no shadow component against the up-sun rim of the crater |
| CR-05 | not measurable | not measurable | not measurable | None | 112.0 | 137.0 | 4160 | no shadow component against the up-sun rim of the crater |
| CR-06 | not measurable | not measurable | not measurable | None | 116.0 | 177.0 | 5720 | no shadow component against the up-sun rim of the crater |
| CR-07 | not measurable | not measurable | not measurable | None | 112.0 | 131.0 | 4297 | no shadow component against the up-sun rim of the crater |
| CR-08 | not measurable | not measurable | not measurable | None | 70.0 | 100.0 | 2095 | shadow component touches the ROI boundary, so its length is a lower bound, not a measurement |
| CR-09 | not measurable | not measurable | not measurable | None | 34.0 | 175.0 | 426 | no shadow component against the up-sun rim of the crater |

rows=9, skipped (no row)=0, not measurable=9

#### yolo

| id | shadow length px | depth m | slope deg | confidence | diameter px | Otsu T | shadow px | not-measurable reason |
|---|---|---|---|---|---|---|---|---|
| CR-01 | not measurable | not measurable | not measurable | 0.8692474365234375 | 119.0 | 179.0 | 7078 | shadow mask covers 62.7% of the crater area (> 60%): ROI histogram was not bimodal |
| CR-02 | not measurable | not measurable | not measurable | 0.8522818684577942 | 73.0 | 180.0 | 2239 | no shadow component against the up-sun rim of the crater |
| CR-03 | not measurable | not measurable | not measurable | 0.8309051394462585 | 61.0 | 175.0 | 1631 | no shadow component against the up-sun rim of the crater |
| CR-04 | not measurable | not measurable | not measurable | 0.7702561616897583 | 57.0 | 162.0 | 1180 | no shadow component against the up-sun rim of the crater |
| CR-05 | not measurable | not measurable | not measurable | 0.7589907646179199 | 68.5 | 100.0 | 2180 | shadow mask covers 60.2% of the crater area (> 60%): ROI histogram was not bimodal |
| CR-06 | not measurable | not measurable | not measurable | 0.7225830554962158 | 103.5 | 142.0 | 4867 | no shadow component against the up-sun rim of the crater |
| CR-07 | not measurable | not measurable | not measurable | 0.7092379927635193 | 37.0 | 179.0 | 564 | no shadow component against the up-sun rim of the crater |
| CR-08 | not measurable | not measurable | not measurable | 0.6920771598815918 | 127.0 | 137.0 | 5575 | no shadow component against the up-sun rim of the crater |
| CR-09 | not measurable | not measurable | not measurable | 0.6427695751190186 | 35.0 | 138.0 | 455 | no shadow component against the up-sun rim of the crater |
| CR-10 | not measurable | not measurable | not measurable | 0.5229388475418091 | 17.0 | 204.0 | 89 | no shadow component against the up-sun rim of the crater |

rows=10, skipped (no row)=0, not measurable=10

#### Not-measurable count across seeds

| seed | detections | rows | not measurable | skipped | reasons |
|---|---|---|---|---|---|
| 42 | yolo | 10 | 10 | 0 | no shadow component against the up-sun rim of the crater; shadow mask covers 60.2% of the crater area (> 60%): ROI histogram was not bimodal; shadow mask covers 62.7% of the crater area (> 60%): ROI histogram was not bimodal |
| 42 | synthetic_meta | 9 | 9 | 0 | no shadow component against the up-sun rim of the crater; shadow component touches the ROI boundary, so its length is a lower bound, not a measurement |
| 7 | yolo | 8 | 6 | 0 | no shadow component against the up-sun rim of the crater |
| 7 | synthetic_meta | 11 | 9 | 0 | no shadow component against the up-sun rim of the crater; shadow component touches the ROI boundary, so its length is a lower bound, not a measurement |
| 123 | yolo | 10 | 9 | 0 | no shadow component against the up-sun rim of the crater; shadow component touches the ROI boundary, so its length is a lower bound, not a measurement |
| 123 | synthetic_meta | 10 | 10 | 0 | no shadow component against the up-sun rim of the crater; shadow component touches the ROI boundary, so its length is a lower bound, not a measurement; shadow mask covers 60.6% of the crater area (> 60%): ROI histogram was not bimodal |

Pooled: yolo 25/28, synthetic_meta 28/30

### B4 scoring (Td 1.8, gear 2.6, density 85 px, 1.0 m/px)

| seed | detections | n | score min | median | max | mean | SAFE | CAUTION | HAZARD | UNKNOWN | overall_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | yolo | 10 | 67.74 | 73.15 | 79.44 | 73.08 | 0 | 0 | 0 | 10 | 73.08 |
| 42 | synthetic_meta | 9 | 63.95 | 71.95 | 75.6 | 71.61333333333333 | 0 | 0 | 0 | 9 | 71.61 |
| 7 | yolo | 8 | 32.32 | 71.555 | 73.19 | 62.47375 | 0 | 0 | 2 | 6 | 62.47 |
| 7 | synthetic_meta | 11 | 31.69 | 69.17 | 73.17 | 62.722727272727276 | 0 | 0 | 2 | 9 | 62.72 |
| 123 | yolo | 10 | 41.64 | 71.9 | 73.5 | 67.908 | 0 | 1 | 0 | 9 | 67.91 |
| 123 | synthetic_meta | 10 | 65.17 | 72.04499999999999 | 73.85 | 71.127 | 0 | 0 | 0 | 10 | 71.13 |

### B5 pathfinding

| seed | detections | route found | path length m | path nodes | alternatives ok | alt nodes | start px | goal px | A* grid (h x w) | goal zone | HAZARD craters crossed (excl. goal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | yolo | True | 274.24 | 83 | 3/3 | [75, 90, 73] | (256, 4) | (492, 100) | 180x180 | UNKNOWN | 0/0 |
| 42 | synthetic_meta | True | 445.86 | 136 | 3/3 | [142, 146, 150] | (256, 4) | (405, 387) | 180x180 | UNKNOWN | 0/0 |
| 7 | yolo | True | 435.38 | 137 | 3/3 | [143, 147, 151] | (256, 4) | (374, 390) | 180x180 | UNKNOWN | 0/2 |
| 7 | synthetic_meta | True | 432.42 | 136 | 3/3 | [142, 146, 150] | (256, 4) | (370, 388) | 180x180 | UNKNOWN | 0/2 |
| 123 | yolo | True | 59.74 | 18 | 3/3 | [17, 30, 25] | (256, 4) | (306, 32) | 180x180 | CAUTION | 0/0 |
| 123 | synthetic_meta | True | 50.94 | 17 | 3/3 | [23, 27, 31] | (256, 4) | (243, 51) | 180x180 | UNKNOWN | 0/0 |

### C1 solar angle sweep — seed 42, detections=synthetic_meta, azimuth 35, 1.0 m/px

| theta deg | tan(theta) | mean shadow length px | mean depth m (code: × tan) | mean depth m (alt: ÷ tan) | n craters | n measurable |
|---|---|---|---|---|---|---|
| 5 | 0.08748866352592401 | None | None | None | 9 | 0 |
| 15 | 0.2679491924311227 | None | None | None | 9 | 0 |
| 25 | 0.4663076581549986 | None | None | None | 9 | 0 |
| 35 | 0.7002075382097097 | None | None | None | 9 | 0 |
| 45 | 0.9999999999999999 | None | None | None | 9 | 0 |
| 55 | 1.4281480067421144 | None | None | None | 9 | 0 |
| 65 | 2.1445069205095586 | None | None | None | 9 | 0 |
| 75 | 3.7320508075688776 | None | None | None | 9 | 0 |
| 85 | 11.430052302761348 | None | None | None | 9 | 0 |

### C1 solar angle sweep — seed 42, detections=yolo, azimuth 35, 1.0 m/px

| theta deg | tan(theta) | mean shadow length px | mean depth m (code: × tan) | mean depth m (alt: ÷ tan) | n craters | n measurable |
|---|---|---|---|---|---|---|
| 5 | 0.08748866352592401 | None | None | None | 10 | 0 |
| 15 | 0.2679491924311227 | None | None | None | 10 | 0 |
| 25 | 0.4663076581549986 | None | None | None | 10 | 0 |
| 35 | 0.7002075382097097 | None | None | None | 10 | 0 |
| 45 | 0.9999999999999999 | None | None | None | 10 | 0 |
| 55 | 1.4281480067421144 | None | None | None | 10 | 0 |
| 65 | 2.1445069205095586 | None | None | None | 10 | 0 |
| 75 | 3.7320508075688776 | None | None | None | 10 | 0 |
| 85 | 11.430052302761348 | None | None | None | 10 | 0 |

### C2 ground truth (seed 42, synthetic_meta detections, theta 35, 1.0 m/px) — replay verified: True


All 9 craters were reported NOT MEASURABLE, so no per-crater errors exist.

### C2b depth estimator validity (seeds 42/7/123 pooled, ground-truth boxes, theta 35, azimuth 35)

- seeds: [42, 7, 123]
- rows: 30
- not_measurable: 28
- pearson_est_vs_true_depth: -0.9999999999999999
- pearson_est_vs_radius_px: 0.9999999999999999
- slope_min_deg: 42.546
- slope_max_deg: 45.05
- slope_std_deg: 1.251999999999999
- L_equals_(side-1)(cos+sin): '0/2'

### C3 depth threshold sweep — seed 42, detections=synthetic_meta (gear 2.6, density 85)

| Td m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 0.5 | 0 | 0 | 0 |
| 0.6 | 0 | 0 | 0 |
| 0.7 | 0 | 0 | 0 |
| 0.8 | 0 | 0 | 0 |
| 0.9 | 0 | 0 | 0 |
| 1.0 | 0 | 0 | 0 |
| 1.1 | 0 | 0 | 0 |
| 1.2 | 0 | 0 | 0 |
| 1.3 | 0 | 0 | 0 |
| 1.4 | 0 | 0 | 0 |
| 1.5 | 0 | 0 | 0 |
| 1.6 | 0 | 0 | 0 |
| 1.7 | 0 | 0 | 0 |
| 1.8 | 0 | 0 | 0 |
| 1.9 | 0 | 0 | 0 |
| 2.0 | 0 | 0 | 0 |
| 2.1 | 0 | 0 | 0 |
| 2.2 | 0 | 0 | 0 |
| 2.3 | 0 | 0 | 0 |
| 2.4 | 0 | 0 | 0 |
| 2.5 | 0 | 0 | 0 |
| 2.6 | 0 | 0 | 0 |
| 2.7 | 0 | 0 | 0 |
| 2.8 | 0 | 0 | 0 |
| 2.9 | 0 | 0 | 0 |
| 3.0 | 0 | 0 | 0 |
| 3.1 | 0 | 0 | 0 |
| 3.2 | 0 | 0 | 0 |
| 3.3 | 0 | 0 | 0 |
| 3.4 | 0 | 0 | 0 |
| 3.5 | 0 | 0 | 0 |
| 3.6 | 0 | 0 | 0 |
| 3.7 | 0 | 0 | 0 |
| 3.8 | 0 | 0 | 0 |
| 3.9 | 0 | 0 | 0 |
| 4.0 | 0 | 0 | 0 |
| 4.1 | 0 | 0 | 0 |
| 4.2 | 0 | 0 | 0 |
| 4.3 | 0 | 0 | 0 |
| 4.4 | 0 | 0 | 0 |
| 4.5 | 0 | 0 | 0 |
| 4.6 | 0 | 0 | 0 |
| 4.7 | 0 | 0 | 0 |
| 4.8 | 0 | 0 | 0 |
| 4.9 | 0 | 0 | 0 |
| 5.0 | 0 | 0 | 0 |

### C3 gear span sweep — seed 42, detections=synthetic_meta (Td 1.8, density 85)

| gear m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 1.0 | 0 | 0 | 0 |
| 1.1 | 0 | 0 | 0 |
| 1.2 | 0 | 0 | 0 |
| 1.3 | 0 | 0 | 0 |
| 1.4 | 0 | 0 | 0 |
| 1.5 | 0 | 0 | 0 |
| 1.6 | 0 | 0 | 0 |
| 1.7 | 0 | 0 | 0 |
| 1.8 | 0 | 0 | 0 |
| 1.9 | 0 | 0 | 0 |
| 2.0 | 0 | 0 | 0 |
| 2.1 | 0 | 0 | 0 |
| 2.2 | 0 | 0 | 0 |
| 2.3 | 0 | 0 | 0 |
| 2.4 | 0 | 0 | 0 |
| 2.5 | 0 | 0 | 0 |
| 2.6 | 0 | 0 | 0 |
| 2.7 | 0 | 0 | 0 |
| 2.8 | 0 | 0 | 0 |
| 2.9 | 0 | 0 | 0 |
| 3.0 | 0 | 0 | 0 |
| 3.1 | 0 | 0 | 0 |
| 3.2 | 0 | 0 | 0 |
| 3.3 | 0 | 0 | 0 |
| 3.4 | 0 | 0 | 0 |
| 3.5 | 0 | 0 | 0 |
| 3.6 | 0 | 0 | 0 |
| 3.7 | 0 | 0 | 0 |
| 3.8 | 0 | 0 | 0 |
| 3.9 | 0 | 0 | 0 |
| 4.0 | 0 | 0 | 0 |
| 4.1 | 0 | 0 | 0 |
| 4.2 | 0 | 0 | 0 |
| 4.3 | 0 | 0 | 0 |
| 4.4 | 0 | 0 | 0 |
| 4.5 | 0 | 0 | 0 |
| 4.6 | 0 | 0 | 0 |
| 4.7 | 0 | 0 | 0 |
| 4.8 | 0 | 0 | 0 |
| 4.9 | 0 | 0 | 0 |
| 5.0 | 0 | 0 | 0 |

Full 46×41 grid distinct (SAFE, CAUTION, HAZARD) outcomes: [(0, 0, 0)]; corners: {'td=0.5,gear=1.0': (0, 0, 0), 'td=0.5,gear=5.0': (0, 0, 0), 'td=5.0,gear=1.0': (0, 0, 0), 'td=5.0,gear=5.0': (0, 0, 0)}

### C3 depth threshold sweep — seed 42, detections=yolo (gear 2.6, density 85)

| Td m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 0.5 | 0 | 0 | 0 |
| 0.6 | 0 | 0 | 0 |
| 0.7 | 0 | 0 | 0 |
| 0.8 | 0 | 0 | 0 |
| 0.9 | 0 | 0 | 0 |
| 1.0 | 0 | 0 | 0 |
| 1.1 | 0 | 0 | 0 |
| 1.2 | 0 | 0 | 0 |
| 1.3 | 0 | 0 | 0 |
| 1.4 | 0 | 0 | 0 |
| 1.5 | 0 | 0 | 0 |
| 1.6 | 0 | 0 | 0 |
| 1.7 | 0 | 0 | 0 |
| 1.8 | 0 | 0 | 0 |
| 1.9 | 0 | 0 | 0 |
| 2.0 | 0 | 0 | 0 |
| 2.1 | 0 | 0 | 0 |
| 2.2 | 0 | 0 | 0 |
| 2.3 | 0 | 0 | 0 |
| 2.4 | 0 | 0 | 0 |
| 2.5 | 0 | 0 | 0 |
| 2.6 | 0 | 0 | 0 |
| 2.7 | 0 | 0 | 0 |
| 2.8 | 0 | 0 | 0 |
| 2.9 | 0 | 0 | 0 |
| 3.0 | 0 | 0 | 0 |
| 3.1 | 0 | 0 | 0 |
| 3.2 | 0 | 0 | 0 |
| 3.3 | 0 | 0 | 0 |
| 3.4 | 0 | 0 | 0 |
| 3.5 | 0 | 0 | 0 |
| 3.6 | 0 | 0 | 0 |
| 3.7 | 0 | 0 | 0 |
| 3.8 | 0 | 0 | 0 |
| 3.9 | 0 | 0 | 0 |
| 4.0 | 0 | 0 | 0 |
| 4.1 | 0 | 0 | 0 |
| 4.2 | 0 | 0 | 0 |
| 4.3 | 0 | 0 | 0 |
| 4.4 | 0 | 0 | 0 |
| 4.5 | 0 | 0 | 0 |
| 4.6 | 0 | 0 | 0 |
| 4.7 | 0 | 0 | 0 |
| 4.8 | 0 | 0 | 0 |
| 4.9 | 0 | 0 | 0 |
| 5.0 | 0 | 0 | 0 |

### C3 gear span sweep — seed 42, detections=yolo (Td 1.8, density 85)

| gear m | SAFE | CAUTION | HAZARD |
|---|---|---|---|
| 1.0 | 0 | 0 | 0 |
| 1.1 | 0 | 0 | 0 |
| 1.2 | 0 | 0 | 0 |
| 1.3 | 0 | 0 | 0 |
| 1.4 | 0 | 0 | 0 |
| 1.5 | 0 | 0 | 0 |
| 1.6 | 0 | 0 | 0 |
| 1.7 | 0 | 0 | 0 |
| 1.8 | 0 | 0 | 0 |
| 1.9 | 0 | 0 | 0 |
| 2.0 | 0 | 0 | 0 |
| 2.1 | 0 | 0 | 0 |
| 2.2 | 0 | 0 | 0 |
| 2.3 | 0 | 0 | 0 |
| 2.4 | 0 | 0 | 0 |
| 2.5 | 0 | 0 | 0 |
| 2.6 | 0 | 0 | 0 |
| 2.7 | 0 | 0 | 0 |
| 2.8 | 0 | 0 | 0 |
| 2.9 | 0 | 0 | 0 |
| 3.0 | 0 | 0 | 0 |
| 3.1 | 0 | 0 | 0 |
| 3.2 | 0 | 0 | 0 |
| 3.3 | 0 | 0 | 0 |
| 3.4 | 0 | 0 | 0 |
| 3.5 | 0 | 0 | 0 |
| 3.6 | 0 | 0 | 0 |
| 3.7 | 0 | 0 | 0 |
| 3.8 | 0 | 0 | 0 |
| 3.9 | 0 | 0 | 0 |
| 4.0 | 0 | 0 | 0 |
| 4.1 | 0 | 0 | 0 |
| 4.2 | 0 | 0 | 0 |
| 4.3 | 0 | 0 | 0 |
| 4.4 | 0 | 0 | 0 |
| 4.5 | 0 | 0 | 0 |
| 4.6 | 0 | 0 | 0 |
| 4.7 | 0 | 0 | 0 |
| 4.8 | 0 | 0 | 0 |
| 4.9 | 0 | 0 | 0 |
| 5.0 | 0 | 0 | 0 |

Full 46×41 grid distinct (SAFE, CAUTION, HAZARD) outcomes: [(0, 0, 0)]; corners: {'td=0.5,gear=1.0': (0, 0, 0), 'td=0.5,gear=5.0': (0, 0, 0), 'td=5.0,gear=1.0': (0, 0, 0), 'td=5.0,gear=5.0': (0, 0, 0)}

### C4 shipped estimator on ray-cast shadow scenes (seeds 42/7/123, elevation 20, azimuth 35)

- sun_elevation_deg: 20.0
- ground_truth: {'craters': 30, 'measured': 2, 'not_measurable': 28, 'rejection_reasons': {'shadow component touches the ROI boundary': 16, 'MEASURED': 2, 'shadow mask covers 65.2% of the crater area (> 60%)': 1, 'shadow mask covers 64.2% of the crater area (> 60%)': 1, 'shadow mask covers 63.5% of the crater area (> 60%)': 1, 'shadow mask covers 77.2% of the crater area (> 60%)': 1, 'shadow mask covers 82.4% of the crater area (> 60%)': 1, 'shadow mask covers 71.7% of the crater area (> 60%)': 1, 'shadow mask covers 73.6% of the crater area (> 60%)': 1, 'shadow mask covers 65.4% of the crater area (> 60%)': 1, 'shadow mask covers 68.3% of the crater area (> 60%)': 1, 'shadow mask covers 76.2% of the crater area (> 60%)': 1, 'shadow mask covers 63.2% of the crater area (> 60%)': 1, 'shadow mask covers 62.0% of the crater area (> 60%)': 1}, 'pearson_est_vs_true_depth': None, 'pearson_est_vs_radius_px': None, 'slope_min_deg': 20.457, 'slope_max_deg': 20.957, 'slope_std_deg': 0.25}
- yolo: {'craters': 26, 'measured': 18, 'not_measurable': 8, 'rejection_reasons': {'MEASURED': 18, 'shadow component touches the ROI boundary': 5, 'shadow mask covers 62.2% of the crater area (> 60%)': 1, 'shadow mask covers 74.2% of the crater area (> 60%)': 1, 'shadow mask covers 71.6% of the crater area (> 60%)': 1}, 'pearson_est_vs_true_depth': 0.8995673087987358, 'pearson_est_vs_radius_px': 0.9694402063001051, 'slope_min_deg': 18.496, 'slope_max_deg': 26.648, 'slope_std_deg': 2.1958330437838023}

### C5 non-crater terrain score: constant 82 vs measured roughness (YOLO boxes)

| seed | mode | score map min | mean | max | std | path length m | path nodes | goal px |
|---|---|---|---|---|---|---|---|---|
| 42 | constant_82 | 67.73999786376953 | 77.93296813964844 | 81.99793243408203 | 3.6558799743652344 | 274.24 | 83 | (492, 100) |
| 42 | roughness | 0.0 | 76.8014907836914 | 99.99482727050781 | 27.787796020507812 | 273.52 | 83 | (492, 100) |
| 7 | constant_82 | 32.31999969482422 | 73.98768615722656 | 81.99990844726562 | 10.067140579223633 | 435.38 | 137 | (374, 390) |
| 7 | roughness | 0.0 | 75.88999938964844 | 99.99400329589844 | 26.94521713256836 | 455.35 | 138 | (374, 390) |
| 123 | constant_82 | 41.63999938964844 | 77.80872344970703 | 82.0 | 4.4072418212890625 | 59.74 | 18 | (306, 32) |
| 123 | roughness | 0.0 | 78.11991882324219 | 99.99996948242188 | 26.460041046142578 | 63.03 | 20 | (306, 32) |

Zone counts are unchanged by this: they come from per-crater scoring, not the terrain map. seed 42: {'safe': 0, 'caution': 0, 'hazard': 0, 'unknown': 10}; seed 7: {'safe': 0, 'caution': 0, 'hazard': 2, 'unknown': 6}; seed 123: {'safe': 0, 'caution': 1, 'hazard': 0, 'unknown': 9}
