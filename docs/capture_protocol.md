# Visiona3D Capture Protocol

This document describes the real-world data collection protocol for Visiona3D.

## Hardware Setup

### Camera Mount
- **Height**: Measure and record camera height above ground (`h_known`) to ±1 cm accuracy
- **Pitch**: Measure and record camera pitch angle (`pitch_deg`) to ±0.5° accuracy
  - Positive = tilted down toward ground
  - Use digital inclinometer on camera body
- **Roll/Yaw**: Record if non-zero (typically 0° for forward-facing)
- **Mount stability**: Verify no vibration or shift during recording

### Intrinsics Calibration
- **Per-device**: Record checkerboard calibration for each camera device
- **Procedure**:
  1. Print checkerboard (e.g., 9x6, 30mm squares)
  2. Capture 20+ images at varying distances/angles
  3. Compute fx, fy, cx, cy using OpenCV `calibrateCamera`
  4. Record full K matrix and distortion coefficients
  5. Save as JSON with device ID and date
- **Re-calibrate**: After any lens adjustment, mount change, or monthly

### Rangefinder Ground Truth
- **Device**: 1D or 2D laser rangefinder (e.g., TFmini, RPLIDAR)
- **Mounting**: Co-located with camera (≤5 cm offset), same orientation
- **Synchronization**: Hardware trigger or timestamp alignment (≤5 ms)
- **Calibration**: Verify rangefinder accuracy against tape measure at 1m, 2m, 4m, 6m

## Data Collection Procedure

### Route Definition
A **route** is a continuous walk from start to end in one environment/condition.
- Duration: 30-120 seconds of walking
- Distance: 20-100 metres
- Single environment tag (indoor/outdoor)
- Single lighting tag (day/dusk/dark)
- Single hazard tag (glass/mirror/wet/none)
- Single crowd tag (none/low/high)

### Recording
1. **Start**: Begin recording at route start point
2. **Walk**: Natural walking pace, carrying camera at mount height
3. **Ground truth points**: At 5-10 locations along route:
   - Stop and hold pose steady for 2 seconds
   - Rangefinder measures distance to target (floor point at 1-6m ahead, wall, obstacle)
   - Record: frame timestamp, pixel (u,v), rangefinder distance, target description
4. **End**: Stop recording at route end

### Diversity Targets (per dataset)
| Factor | Target |
|--------|--------|
| Routes indoor | ≥ 10 |
| Routes outdoor | ≥ 10 |
| Routes day | ≥ 10 |
| Routes dusk | ≥ 5 |
| Routes dark | ≥ 5 |
| Routes with glass | ≥ 5 |
| Routes with mirror | ≥ 3 |
| Routes wet | ≥ 3 |
| Routes crowd low | ≥ 8 |
| Routes crowd high | ≥ 5 |
| Height range | 1.0 - 2.0 m |
| Pitch range | 0° - 30° |

## Synchronization & Timestamps

- **Camera**: Record frame timestamps (monotonic, nanoseconds since route start)
- **Rangefinder**: Record measurement timestamps (same clock domain)
- **IMU** (if available): Record at ≥100 Hz for pose refinement
- **GPS** (outdoor): Record for route geolocation

## Data Format

### Per Frame
- RGB image (compressed, e.g., JPEG 85%)
- Depth map placeholder (to be filled by model)
- Segmentation mask placeholder (to be filled by annotator)
- Intrinsics K matrix
- Mount config (h_known, pitch_deg, roll_deg, yaw_deg)

### Per Route Manifest
- Route ID
- Tags (env, light, hazard, crowd)
- Mount config (averaged over route)
- List of frames with timestamps and ground-truth points
- Sync info (camera-rangefinder offset)

### Ground-Truth Points
For each annotated point:
- Frame ID
- Pixel (u, v) in camera image
- Rangefinder Euclidean range (metres)
- Converted z-depth using `range_to_depth(range, u, v, K)`
- Target description (floor@3m, wall, box_front, etc.)
- Method: `rangefinder`

## Why Splits Are By Route

**Route-level splits prevent data leakage** because:
1. **Temporal correlation**: Consecutive frames are highly correlated (same scene, similar pose)
2. **Environment correlation**: All frames in a route share lighting, layout, surface properties
3. **Mount consistency**: Camera height/pitch are constant within a route
4. **Realistic evaluation**: Test routes represent unseen environments, not just unseen frames

Random frame splitting would inflate performance by testing on near-duplicates of training frames.

## Quality Control

- **Daily**: Verify mount height/pitch hasn't shifted
- **Per route**: Check rangefinder sync (visual spot-check)
- **Weekly**: Re-run checkerboard calibration
- **Post-collection**: Run annotation verification on 10% of frames

## Ethical & Privacy

- **Faces**: Blur faces in RGB before storage (or exclude RGB from dataset)
- **Private property**: Obtain permission for indoor locations
- **Consent**: Inform participants; allow opt-out
- **Data retention**: Delete raw video after frame extraction + annotation