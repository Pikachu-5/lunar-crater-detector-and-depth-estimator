### B1 timing (seconds, seed 42, 5 runs after 1 excluded warm-up)

| detection path | stage | mean | stdev (sample) | raw runs |
|---|---|---|---|---|
| yolo | preprocess | 0.0009716800006572158 | 4.7740306264908325e-05 | 0.000942600003327243, 0.0009580000041751191, 0.000949099994613789, 0.0010564999975031242, 0.0009522000036668032 |
| yolo | detection | 0.08709143999731168 | 0.004345322889886249 | 0.0942287999932887, 0.08703599999716971, 0.08527650000178255, 0.08253210000111721, 0.08638379999320023 |
| yolo | depth | 0.0026834199961740524 | 0.0002114500578289212 | 0.0023406999971484765, 0.002798399989842437, 0.00266169999667909, 0.002717099996516481, 0.002899200000683777 |
| yolo | terrain_depth_map | 0.0007522799976868555 | 6.988645983915266e-05 | 0.0006470999942393973, 0.00082970000221394, 0.0007436000014422461, 0.0007409999961964786, 0.0007999999943422154 |
| yolo | terrain_figures | 0.01716230000311043 | 0.0008132393213869758 | 0.017469400001573376, 0.01748829999996815, 0.016078600005130284, 0.018152100004954264, 0.016623100003926083 |
| yolo | terrain_total | 0.017914580000797285 | 0.0008055847903331573 | 0.018116499995812774, 0.01831800000218209, 0.01682220000657253, 0.018893100001150742, 0.0174230999982683 |
| yolo | scoring | 0.0854455799999414 | 0.0018833283725491366 | 0.08603580000635702, 0.08367249999719206, 0.08328229999460746, 0.08757550000154879, 0.08666180000000168 |
| yolo | astar | 0.20474400000239257 | 0.0067771450732085495 | 0.20044129999587312, 0.20161290001124144, 0.20471030000771862, 0.21646270000201184, 0.20049279999511782 |
| synthetic_meta | preprocess | 0.0010333400015952066 | 9.610864958457489e-05 | 0.00116379999963101, 0.0010140999947907403, 0.000978900003246963, 0.001091500002075918, 0.0009184000082314014 |
| synthetic_meta | detection | 0.0003010399959748611 | 5.280807623119398e-05 | 0.00027779999072663486, 0.00026869999419432133, 0.0002769999991869554, 0.0002869000018108636, 0.0003947999939555302 |
| synthetic_meta | depth | 0.0019550000026356427 | 0.00016685165713095277 | 0.0018356000073254108, 0.0022237000084714964, 0.0018253000016557053, 0.0018823999998858199, 0.002007999995839782 |
| synthetic_meta | terrain_depth_map | 0.0005337600014172494 | 1.76451950572573e-05 | 0.0005164000031072646, 0.0005402000097092241, 0.0005292999994708225, 0.0005219000013312325, 0.0005609999934677035 |
| synthetic_meta | terrain_figures | 0.015544880004017613 | 0.0011157843740895678 | 0.015957900002831593, 0.01701609999872744, 0.014668000003439374, 0.014214000009815209, 0.015868400005274452 |
| synthetic_meta | terrain_total | 0.016078640005434863 | 0.001122805569286919 | 0.016474300005938858, 0.017556300008436665, 0.015197300002910197, 0.014735900011146441, 0.016429399998742156 |
| synthetic_meta | scoring | 0.07685072000895161 | 0.0010508693615965772 | 0.07813050001277588, 0.0766113999998197, 0.07569000001240056, 0.07773780000570696, 0.07608390001405496 |
| synthetic_meta | astar | 0.7582493000023532 | 0.02037038876332052 | 0.7688043000089237, 0.7548544000019319, 0.7702457999985199, 0.7732624000054784, 0.7240795999969123 |

Warm-up run (excluded; includes YOLO weight load on first call): {"yolo": {"wall_total": 1.8133538999973098, "preprocess": 0.002054099997621961, "detection": 1.2864168000087375, "depth": 0.04092990000208374, "terrain_depth_map": 0.0007960999937495217, "terrain_figures": 0.1475973999913549, "terrain_total": 0.14839349998510443, "scoring": 0.07778919998963829, "astar": 0.2577158999920357}, "synthetic_meta": {"wall_total": 0.8593376000062563, "preprocess": 0.0009107999940169975, "detection": 0.0002789999998640269, "depth": 0.001930400001583621, "terrain_depth_map": 0.0005573000089498237, "terrain_figures": 0.01382549999107141, "terrain_total": 0.014382800000021234, "scoring": 0.07766729999275412, "astar": 0.7640184999909252}}

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
