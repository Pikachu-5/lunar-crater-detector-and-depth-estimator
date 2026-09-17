### B1 timing (seconds, seed 42, 5 runs after 1 excluded warm-up)

| detection path | stage | mean | stdev (sample) | raw runs |
|---|---|---|---|---|
| yolo | preprocess | 0.0009687200014013797 | 2.245210608260523e-05 | 0.001004100005957298, 0.0009447000047657639, 0.0009585999941918999, 0.0009616000024834648, 0.000974599999608472 |
| yolo | detection | 0.08162688000011258 | 0.0038685935366697577 | 0.08552210000925697, 0.07809889999043662, 0.07821160000457894, 0.08597359999839682, 0.08032819999789353 |
| yolo | depth | 0.0026785400026710705 | 0.00030154660004401873 | 0.00292990000161808, 0.002729600004386157, 0.0026745000068331137, 0.002885299996705726, 0.002173400003812276 |
| yolo | terrain_depth_map | 0.0007204399997135624 | 9.066582745199825e-05 | 0.000829399999929592, 0.0006901999877300113, 0.0007513000018661842, 0.0007466999959433451, 0.0005846000130986795 |
| yolo | terrain_figures | 0.01661887999798637 | 0.0006619508345307328 | 0.01692519999051001, 0.015462199997273274, 0.016816100003779866, 0.017130899999756366, 0.01675999999861233 |
| yolo | terrain_total | 0.017339319997699932 | 0.0006933668426109242 | 0.0177545999904396, 0.016152399985003285, 0.01756740000564605, 0.01787759999569971, 0.01734460001171101 |
| yolo | scoring | 0.0855145999987144 | 0.001756794928748953 | 0.08616869999968912, 0.08261070000298787, 0.08605660000466742, 0.08729929999390151, 0.08543769999232609 |
| yolo | astar | 0.20510260000010022 | 0.007293495761946749 | 0.21509230000083335, 0.2034646999964025, 0.19590630001039244, 0.2019628999987617, 0.20908679999411106 |
| synthetic_meta | preprocess | 0.0009505400055786594 | 2.8315505233365384e-05 | 0.000995800000964664, 0.0009385000012116507, 0.0009298000077251345, 0.000928400011616759, 0.0009602000063750893 |
| synthetic_meta | detection | 0.0002736800001002848 | 7.461361562084439e-06 | 0.0002837999927578494, 0.00027129999944008887, 0.00026330001128371805, 0.00027630000840872526, 0.0002736999886110425 |
| synthetic_meta | depth | 0.002422319998731837 | 0.0009399243991452273 | 0.004081099992617965, 0.0018807000014930964, 0.0019988999993074685, 0.0018896999972639605, 0.002261200002976693 |
| synthetic_meta | terrain_depth_map | 0.0005445599992526696 | 6.948721582220983e-05 | 0.0006674000032944605, 0.0005276999872876331, 0.0004994000046281144, 0.000509100005729124, 0.0005191999953240156 |
| synthetic_meta | terrain_figures | 0.014577600001939573 | 0.0025235022519220973 | 0.019030800001928583, 0.013367399995331652, 0.014163600004394539, 0.013205900002503768, 0.01312030000553932 |
| synthetic_meta | terrain_total | 0.015122160001192242 | 0.002590126745375616 | 0.019698200005223043, 0.013895099982619286, 0.014663000009022653, 0.013715000008232892, 0.013639500000863336 |
| synthetic_meta | scoring | 0.07352237999730278 | 0.001745917713713346 | 0.07063930000003893, 0.07358169999497477, 0.07518960000015795, 0.07456369999272283, 0.07363759999861941 |
| synthetic_meta | astar | 0.7296060199936619 | 0.013259224066636417 | 0.7330973999924026, 0.7146996999945259, 0.7190473999944516, 0.7482064999931026, 0.7329790999938268 |

Warm-up run (excluded; includes YOLO weight load on first call): {"yolo": {"wall_total": 1.7099817000125768, "preprocess": 0.0009340999968117103, "detection": 1.2193538999999873, "depth": 0.027333300007740036, "terrain_depth_map": 0.000720500000170432, "terrain_figures": 0.12246840000443626, "terrain_total": 0.1231889000046067, "scoring": 0.08846680000715423, "astar": 0.25063469998713117}, "synthetic_meta": {"wall_total": 0.8318427000049269, "preprocess": 0.0009597000025678426, "detection": 0.00028589999419637024, "depth": 0.0033226999948965386, "terrain_depth_map": 0.0007728000055067241, "terrain_figures": 0.016692100005457178, "terrain_total": 0.017464900010963902, "scoring": 0.0737370999995619, "astar": 0.735871999990195}}

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
