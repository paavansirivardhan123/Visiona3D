# Bird's-eye map, planner and guidance

Mapping and planning layer for Visiona-Walk (project "Vishnu 3D"): it turns one
monocular frame's depth, floor segmentation and calibration outputs into a
ground-frame occupancy map, a costmap, a recommended path and a guidance
decision. Every automated result in this repo is **SYNTHETIC ONLY** (analytic
scenes, fake backends); no real-model measurement has been made, and every
numeric default below is a starting point marked `TO TUNE ON REAL DATA`, not a
validated value.

## Safety principles (verbatim)

These four principles are enforced in the code and repeated in the docstring of
every module under `mapping/` and `planning/`:

1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.

## Definitions

- **Ground frame G** — X right, Y forward (row grows with forward distance),
  Z up, origin below the camera on the floor plane. Produced by
  `geometry.lift.lift_frame` / `geometry.frames.camera_to_ground`.
- **Grid** — `mapping/grid.py`. `col = floor((X - x_min)/res)`,
  `row = floor((Y - y_min)/res)` (a `+1e-9` epsilon keeps exact boundaries in
  their cell). Defaults: `res` 0.10 m, X in [-3, 3], Y in [-0.5, 7.0].
- **Cell states** — `CELL_UNKNOWN 0`, `CELL_FREE 1` (certified walkable),
  `CELL_OCCUPIED 2`, `CELL_UNCERTAIN 3` (floor-labelled but not certified or
  vetoed). These are distinct from the mask class indices in `core/config.py`.
- **Obstacle height band** — points with height Z in [0.15, 2.0] m above the
  floor plane that are not certified floor. This catches overhangs a naive
  floor-only map would miss.
- **BEV map** — `mapping/bev.py`. `build_bev(pointmap_g, labels,
  certified_final, vetoed, pessimistic_pointmap, cfg)` votes per cell:
  OCCUPIED from non-floor labels (wall, obstacle), vetoed floor labels, and any
  non-certified point inside the height band; FREE from certified floor points
  with |Z| <= tau; a cell with both votes is OCCUPIED; FREE needs at least
  `min_points_per_cell` votes; floor-labelled cells without enough certified
  votes are UNCERTAIN; no votes means UNKNOWN. The pessimistic layer adds
  OCCUPIED votes at conformal lower-bound positions (Safety Principle 2).
- **Pessimistic (lower-bound) layer** — every obstacle pixel's depth is shrunk
  to its split-conformal lower bound (`lower = pred * (1 - q)`) before being
  lifted to G, and those points cast OCCUPIED votes. The score
  `overestimation` is the intended one for obstacle distance, but both
  supported scores produce the same lower bound. With the stub `q = 0.20` used
  in tests, a wall at 4.0 m is also placed at 3.2 m.
- **Certified floor** — floor pixels whose segmentation probability passes the
  conformal floor threshold (`calibration.conformal_floor.certified_floor`)
  and which survive the geometric veto (`calibration.geometric_veto.
  veto_floor_mask`: the point must lie within `tau0_m` of the fitted floor
  plane, with a slope allowance `tau1`).
- **Rolling map** — `mapping/rolling.py`. `RollingMap.update(bev, odometry,
  dt)` fuses maps over time with exponential decay (half-life 1.5 s),
  integer-cell translation and nearest-neighbour rotation from an optional
  `Odometry2D(dx, dy, dyaw)` (identity when None), and a conservative state
  rule: occupied score above threshold wins, then free, then uncertain, else
  unknown. Temporal fusion is validated only on synthetic motion; the source of
  odometry is not part of this work.
- **Costmap** — `planning/costmap.py`. A numpy-only two-pass chamfer distance
  transform (1, sqrt(2) metric) gives the clearance field; cells are blocked
  when they are not FREE (Safety Principle 1) or when their clearance is within
  `user_half_width + safety_margin` of a blocked cell (inflation).
- **A\*** — `planning/astar.py`. 8-connected with an admissible octile
  heuristic, no diagonal corner cutting through blocked cells, step cost
  `step * (1 + k * ((c_max - clearance)/c_max)^2)` for clearance < c_max.
  Returns status `OK`, `NO_PATH` or `START_BLOCKED`; never raises.
- **Targets** — `planning/targets.py`. Explore mode: farthest reachable FREE
  cell in a forward cone of ±60 degrees at least 1.5 m away, ties broken by
  centreline |X|. Anchor mode: nearest reachable FREE cell to a lifted object
  anchor within a radius. Multi-path: 30-degree sectors, farthest reachable
  cell per sector, up to 3 candidate paths ranked by length then clearance.
- **Smoothing** — `planning/smooth.py`. Weighted moving-average filter that
  keeps endpoints pinned and re-validates every segment against the blocked
  mask; falls back to the raw path if smoothing would collide.
- **Guidance** — `planning/guidance.py`. `Guidance(decision in {GUIDE,
  CAUTION, STOP_UNCLEAR, STOP_BLOCKED}, heading_deg (positive right),
  lookahead_point_g, distance_to_turn_m, left_clearance_m, right_clearance_m,
  nearest_obstacle {label, distance_nominal_m, bearing_deg, and optional
  distance_lower_m / distance_upper_m}, reasons, abstain_features, timings)`.
  It calls `calibration.abstain.AbstainRule` on corridor features computed from
  the map; STOP_UNCLEAR on abstain; STOP_BLOCKED when no path; CAUTION when
  minimum clearance is below the caution threshold.
- **Nearest-obstacle telemetry** — `mapping/obstacles.py`. Distances are
  measured from the user origin over the pixels the BEV classified as
  obstacles. Conformal bounds are the minima of the lower/upper point maps over
  those pixels; for every obstacle pixel `lower_i <= d_i <= upper_i`, so
  `min lower` and `min upper` bound the true nearest distance. Missing bounds
  are omitted, never replaced by the nominal value.
- **Demo and ablation scripts** — `scripts/plan_demo.py` (one image or
  `--fake`; prints the Guidance and writes `bev_path.png`,
  `guidance_summary.json`, and a `run_summary.json` with intrinsics, fusion,
  certification and backend-status provenance) and
  `scripts/run_planner_eval.py` (SYNTHETIC ONLY ablation, described below).

### Real-image demo path

Real-image mode constructs SegFormer-B0, Depth Anything V2 (indoor or outdoor)
and YOLO11n from local weights, and optionally wraps SegFormer with MobileSAM
using a tuning report from `scripts/tune_fusion.py`. It exits `2` with fetch or
calibration instructions when weights, calibrated intrinsics, or MobileSAM are
missing, and refuses a synthetic tuning report in a real run. The image
resolution must match the intrinsics exactly; there is no field-of-view guess.

Certification is probability-only. `--floor-threshold` (default 0.90, TO TUNE
ON REAL DATA) feeds the pipeline's conformal floor check and geometric veto, and
`certified_final` is the only input that can vote FREE. If certification is
unavailable, the demo fails closed to an all-False mask, prints a warning, and
guidance reports STOP_UNCLEAR instead of treating fused labels as free space.

The demo does not fit a depth interval by itself. Pass
`--conformal-depth reports/calibration/depth_interval.json` to activate the
pessimistic lower-bound obstacle layer. The artifact is written by
`scripts/run_calibration_real.py` and loaded through
`calibration/depth_interval_config.py`, which validates schema, provenance and
arithmetic, and refuses a fake-mode artifact in a real run. Without it,
obstacles use nominal distances only, the demo warns, and `run_summary.json`
records `pessimistic_layer: false`; with it, the artifact provenance is
recorded under `depth_interval` and the layer is `true`. Validation does not
establish coverage: every quantile remains TO TUNE ON REAL DATA.

The same artifact feeds nearest-obstacle telemetry. With it,
`run_summary.json` records `nearest_obstacle` with the nominal distance and the
conformal lower/upper bounds, and `AbstainRule` abstains when the relative
interval width exceeds its threshold (default 0.50, TO TUNE ON REAL DATA).
Without the artifact, no bound is fabricated: the interval check is skipped and
`abstain_features.relative_interval_width` is `null`. An inverted interval is
treated as non-finite and abstains. The demo prints the bounds on the
`Nearest Obstacle` telemetry line.

```bash
python scripts/plan_demo.py --fake
python scripts/run_calibration_real.py --dataset data/visiona \
    --intrinsics reports/camera_intrinsics.json --out reports/calibration
python scripts/plan_demo.py frame.png --intrinsics reports/camera_intrinsics.json \
    --weights-dir weights --depth-variant indoor \
    --conformal-depth reports/calibration/depth_interval.json \
    --fusion-config reports/fusion_tuning/fusion_tuning.json --out-dir reports/demo_plan
```

For an offline end-to-end check, pass `--fake` together with a `fake`-mode
artifact (or omit `--conformal-depth`).

## Every numeric default is TO TUNE ON REAL DATA

None of the values below has been validated on real data. Each is also labelled
in the source file where it is defined.

| Default | Value | Defined in |
|---|---|---|
| `res` (cell size) | 0.10 m | `mapping/grid.py` |
| `x_min` / `x_max` | -3.0 / 3.0 m | `mapping/grid.py` |
| `y_min` / `y_max` | -0.5 / 7.0 m | `mapping/grid.py` |
| `z_band_min` / `z_band_max` (height band) | 0.15 / 2.0 m | `mapping/grid.py` |
| `min_points_per_cell` | 5 | `mapping/grid.py` |
| `tau_floor` (max \|Z\| for floor votes) | 0.10 m | `mapping/grid.py` |
| `vote_threshold` | 0.5 | `mapping/grid.py` |
| rolling `half_life_s` | 1.5 s | `mapping/rolling.py` |
| rolling `occ_score_threshold` | 1.0 | `mapping/rolling.py` |
| rolling `free_score_threshold` | 5.0 | `mapping/rolling.py` |
| `user_half_width_m` | 0.35 m | `planning/costmap.py` |
| `safety_margin_m` | 0.10 m | `planning/costmap.py` |
| `k_clearance` (cost weight) | 2.0 | `planning/costmap.py` |
| `c_max_m` (clearance cutoff) | 1.5 m | `planning/costmap.py` |
| explore `cone_deg` | 60 degrees | `planning/targets.py` |
| explore `min_dist_m` | 1.5 m | `planning/targets.py` |
| anchor `radius_m` | 1.5 m | `planning/targets.py` |
| smoothing `iterations` | 3 | `planning/smooth.py` |
| smoothing `weight_smooth` | 0.5 | `planning/smooth.py` |
| `caution_clearance_m` | 0.20 m | `planning/guidance.py` |
| `lookahead_distance_m` | 1.5 m | `planning/guidance.py` |
| `turn_threshold_deg` | 15 degrees | `planning/guidance.py` |
| demo `mount_height_m` / `pitch_deg` | 1.4 m / 15 degrees | `scripts/plan_demo.py` |
| demo `floor_threshold` | 0.90 | `scripts/plan_demo.py` |
| ablation `user_half_width_m` (truth check) | 0.35 m | `scripts/run_planner_eval.py` |

The calibration thresholds that feed the map (conformal alpha, floor
probability threshold, veto `tau0_m`/`tau1`, abstention thresholds) are listed
and marked in `docs/` calibration notes and `calibration/*.py`; they are also
starting points only.

## Measured results (SYNTHETIC ONLY)

All numbers come from analytic scenes and fake backends. Commands:
`python scripts/run_planner_eval.py --scenes 200 --seed 42` and
`python -m pytest tests -q`. Generated reports are local (`reports/` is
gitignored); regenerate them instead of relying on stored artifacts.

- **M8 conformal margin** (stub `q = 0.20`, wall at 4.0 m): the pessimistic
  layer places the wall at 3.2 m (rows 40 -> 32 at res 0.10 m). Planning from
  (0, 1.8) with default inflation, the nominal path's furthest waypoint is
  Y = 3.55 m (0.45 m true clearance to the wall) and the pessimistic path's is
  Y = 2.75 m (1.25 m) - a gain of exactly 0.80 m, the conformal margin.
- **M11 glass patch (measured, not assumed)**: a 2 m wide glass pane at 4.0 m
  whose pixels see through to 8.0 m and are mislabelled as floor. Measured:
  9,850 glass pixels; 0 of the 630 glass-footprint cells are certified FREE
  (609 UNKNOWN, 21 OCCUPIED); the recommended path stops at Y = 3.55 m, i.e.
  the Safety-Principle-1 inflation around UNKNOWN cells keeps a 0.45 m stand-off
  from an invisible pane. The system does not detect glass as such - it simply
  never certifies it.
- **M11 wall-as-floor corruption (30%)**: the geometric veto keeps the measured
  unsafe rate at 0.0 (no path waypoint comes within the user half-width of the
  true wall); without certification the naive variant's unsafe rate on the same
  scenes is 14.0% (see the ablation table below).
- **M12 ablation** (200 seeded scenes, wall 3.5-6.5 m, 0-2 boxes incl. floating
  overhangs, 3% depth noise, 30% wall-as-floor corruption on odd scenes;
  runtime 39.1 s < 60 s in the latest run - machine-load dependent):

  | Variant | Unsafe rate (all) | Unsafe (corrupted scenes) | Availability | Mean min clearance | Latency p50/p95 |
  |---|---|---|---|---|---|
  | A naive | 10.5% [6.5%, 15.0%] | 14.0% [7.0%, 21.0%] | 79.5% | 0.10 m | 42.1 / 64.2 ms |
  | B + conformal floor | 0.0% | 0.0% | 65.5% | 0.39 m | 29.9 / 43.7 ms |
  | C + geometric veto | 0.0% | 0.0% | 14.5% | 0.49 m | 18.3 / 33.0 ms |
  | D + conformal margin (full) | 0.0% | 0.0% | 8.5% | 0.50 m | 32.0 / 44.1 ms |

  Safety buys availability: the safer variants refuse to guide far more often.
  Availability is measured as the fraction of scenes with decision GUIDE or
  CAUTION; the low C/D numbers are dominated by abstention on the corrupted
  scenes, not by planning failures.
- **M13 speed and determinism**: median A* latency on a 70 x 75 grid is
  1.7 ms (limit 50 ms; the ablation's per-scene p50 including map build is
  18.3-42.1 ms across variants in the latest run); identical inputs and seeds
  give identical results (`test_m13_repeat_runs_are_identical`,
  `test_t10_determinism`). The new mapping/planning test files add 19.5 s to the
  suite (limit 60 s); the full suite took 136.1 s standalone and 124.9 s inside
  the combined runner - both under the 150 s target, but load-sensitive.

## Limitations

- **No dynamic obstacles.** Everything is static per frame; the rolling map's
  decay removes old evidence but nothing detects motion.
- **No stairs or curbs.** A single floor plane is assumed (RANSAC fit or the
  mount geometry). Stairs, curbs and ramps violate the plane assumption and are
  out of scope; the height band [0.15, 2.0] m does not model a step-down edge
  as an obstacle by itself.
- **Glass.** Glass is not detected. It is handled only by failing to certify:
  see-through depth lands beyond the pane and the pane's own cells stay
  UNKNOWN, so the planner keeps a 0.45 m (inflation-radius) stand-off (measured
  above). A glass pane closer than the inflation radius to the user, or glass
  the segmenter certifies as floor with a consistent plane fit, is not covered
  by any measurement.
- **Odometry source.** The rolling map accepts `Odometry2D(dx, dy, dyaw)` but
  the source of odometry (visual, IMU, other) is not part of this work; the
  rotation/translation warp is validated only on synthetic motion, with
  nearest-neighbour cell rotation.
- **Single monocular frame.** The map covers the camera's field of view only;
  occluded space stays UNKNOWN and is never traversed (this is a safety
  feature, but it also limits how far the planner can recommend).
- **The planner only recommends** (Safety Principle 4): the Guidance output is
  a suggestion with reasons, never a claim that the path is safe.

## Glossary

- **BEV** — bird's-eye view occupancy grid in ground frame G.
- **Certified floor** — floor pixels that passed both the conformal floor
  threshold and the geometric veto.
- **Chamfer distance transform** — approximate Euclidean distance to the
  nearest blocked cell computed with (1, sqrt(2)) grid steps.
- **Conformal lower bound** — depth shrunk to the split-conformal quantile so
  the true distance is above it with the calibrated probability.
- **Corridor features** — certified-floor fraction, interval width, plane
  inlier ratio, scale status and veto fraction fed to the abstention rule.
- **Inflation** — marking cells blocked when their clearance is within
  user-half-width plus safety-margin of an obstacle.
- **Nearest-obstacle telemetry** — the nominal distance and optional conformal
  lower/upper bounds to the closest BEV-classified obstacle pixel, fed to
  guidance and abstention.
- **Octile heuristic** — admissible grid heuristic: cardinal steps cost 1,
  diagonal steps cost sqrt(2).
- **Pessimistic layer** — the second, lower-bound obstacle vote layer in the
  BEV map (Safety Principle 2).
- **Veto (geometric)** — rejecting certified floor points whose 3D position is
  inconsistent with the fitted floor plane.
