# Visiona3D Coordinate Frames and Mathematical Notation

- Camera frame C (OpenCV convention): x right, y down, z forward. Units: metres. Depth d of a pixel is its z coordinate in C, not the Euclidean range.
- Intrinsics K = [[fx,0,cx],[0,fy,cy],[0,0,1]]. Pixel (u,v): u right, v down.
- Floor plane in C: { p : n.p = h }, where n is the unit normal pointing DOWN toward the floor and h > 0 is the perpendicular distance from the camera centre to the floor (camera height). For pitch theta (degrees, positive = camera tilted down, roll = 0): n = (0, cos(theta), sin(theta)).
- Ground frame G (gravity-aligned, origin on the floor directly below the camera): X right, Y forward (horizontal), Z up. For a point p in C: e_up = -n; e_fwd = normalize(z_c - (z_c.e_up) e_up) with z_c = (0,0,1); e_right = e_fwd x e_up; then X = e_right.p, Y = e_fwd.p, Z = h - n.p (height above floor).
- Symbols: d = depth, d_hat = predicted depth, h_known = true mounted camera height, h_hat = camera height estimated from the plane fit, s = h_known / h_hat = metric scale correction.

---

## Worked Numeric Examples

### Example 1: Pinhole Projection and Backprojection Roundtrip

A point at depth $z = 2.5\text{ m}$ backprojected from pixel $(u, v) = (200, 100)$ round-trips exactly to pixel $(200, 100)$.

```python
>>> import numpy as np
>>> from core.config import Intrinsics
>>> from geometry.camera import backproject, project
>>> K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
>>> depth = np.full((480, 640), 2.5, dtype=np.float64)
>>> points_c = backproject(depth, K)
>>> pixels = project(points_c, K)
>>> bool(np.allclose(pixels[100, 200], [200.0, 100.0], atol=1e-6))
True

```

### Example 2: Transformation from Camera Frame C to Ground Frame G

For a camera at height $h = 1.4\text{ m}$ with zero pitch ($\theta = 0^\circ$), the ground normal pointing down is $n = (0, 1, 0)$. A floor point at distance $z = 3.0\text{ m}$ with $y = 1.4\text{ m}$ in $C$ maps to $X = 0\text{ m}, Y = 3.0\text{ m}, Z = 0.0\text{ m}$ in $G$.

```python
>>> import numpy as np
>>> from geometry.frames import camera_to_ground
>>> n = np.array([0.0, 1.0, 0.0])
>>> h = 1.4
>>> p = np.array([0.0, 1.4, 3.0])
>>> p_g = camera_to_ground(p, n, h)
>>> bool(np.allclose(p_g, [0.0, 3.0, 0.0], atol=1e-6))
True

```

### Example 3: Metric Scale Correction

A camera mounted at known height $h_{\text{known}} = 1.4\text{ m}$ observes an unscaled depth map where plane fitting yields estimated height $\hat{h} = 1.75\text{ m}$. The raw scale correction is $s = h_{\text{known}} / \hat{h} = 0.8$. Applying $s$ to an unscaled wall depth $\hat{d} = 5.0\text{ m}$ recovers the true metric distance $d = 4.0\text{ m}$.

```python
>>> import numpy as np
>>> from core.config import MountConfig
>>> mount = MountConfig(h_known=1.4, pitch_deg=0.0)
>>> h_hat = 1.75
>>> s_raw = mount.h_known / h_hat
>>> bool(np.isclose(s_raw, 0.8))
True
>>> d_hat = 5.0
>>> d_metric = d_hat * s_raw
>>> bool(np.isclose(d_metric, 4.0))
True

```

---

## Dataset and Evaluation Conventions

### Ground-Truth Distance
Ground-truth `distance_m` is the **z-depth in camera frame C** of the annotated point, *not* Euclidean range.

### Range-to-Depth Conversion
A laser rangefinder measures Euclidean range along the pixel ray: the point is `range * r/|r|` with `r = K^-1 [u,v,1]`; depth = `range * r_z/|r|`.

Functions provided in `geometry.camera`:
- `range_to_depth(range_m, u, v, K)` — range → depth
- `depth_to_range(depth, u, v, K)` — depth → range

**Worked Example 4: Range-to-Depth at Principal Point**
At the principal point `(cx, cy)`, the ray is `(0, 0, 1)`, so `|r| = 1` and `r_z/|r| = 1`. Range equals depth.

```python
>>> from core.config import Intrinsics
>>> from geometry.camera import range_to_depth, depth_to_range
>>> K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
>>> float(range_to_depth(5.0, 320.0, 240.0, K))
5.0
>>> float(depth_to_range(5.0, 320.0, 240.0, K))
5.0

```

**Worked Example 5: Range-to-Depth Off-Axis**
At `u=420` (100px right of center), `v=240` (on center row): `dx = 0.2`, `dy = 0`, `dz = 1`, `|r| = sqrt(1.04) ≈ 1.0198`. Depth = `range / sqrt(1.04)`.

```python
>>> import numpy as np
>>> K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
>>> d = range_to_depth(5.0, 420.0, 240.0, K)
>>> bool(np.isclose(d, 5.0 / np.sqrt(1.04)))
True
>>> # Round-trip
>>> r = depth_to_range(d, 420.0, 240.0, K)
>>> bool(np.isclose(r, 5.0))
True

```

**Worked Example 6: Array Round-Trip**
```python
>>> import numpy as np
>>> ranges = np.array([2.0, 3.5, 5.0, 7.0])
>>> us = np.full(4, 320.0)
>>> vs = np.full(4, 240.0)
>>> depths = range_to_depth(ranges, us, vs, K)
>>> ranges_back = depth_to_range(depths, us, vs, K)
>>> bool(np.allclose(ranges, ranges_back, rtol=1e-9))
True

```

### Mask Class Indices (Defined in `core.config`)
| Index | Name | Semantics |
|-------|------|-----------|
| 0 | unlabeled | Ignored everywhere |
| 1 | floor | Floor / ground plane |
| 2 | wall | Vertical wall |
| 3 | obstacle | Obstacle (object above ground) |
| 4 | other | Other (ceiling, vegetation, etc.) |
| 255 | ignore | Ignore region (glass, mirror, ambiguous) |

### Range Bins (for per-range metrics)
Left-inclusive, right-exclusive:
- Bin 0: `[0, 2)` m
- Bin 1: `[2, 4)` m
- Bin 2: `[4, 6)` m
- Bin 3: `[6, ∞)` m

### Tag Vocabularies (Defined in `core.config`)
| Tag | Values |
|-----|--------|
| env | indoor, outdoor |
| light | day, dusk, dark |
| hazard | glass, mirror, wet, none |
| crowd | none, low, high |

## Perception conventions

These conventions apply to every new perception backend, fake, pipeline output
and evaluation record.

- **Images** are RGB `uint8` with shape `(H, W, 3)`.
- **Depth** is metric z-depth in camera frame C, `float32` `(H, W)`, in metres,
  with `NaN` where a pixel is invalid. Depth is *not* Euclidean range.
- **Labels** use the existing mask indices from `core.config`:
  0 unlabeled, 1 floor, 2 wall, 3 obstacle, 4 other, 255 ignore.
- **Probabilities** are `float32` `(H, W, 4)` in channel order
  `[floor, wall, obstacle, other]`, summing to 1 along the channel axis.
- **Sky** is delivered as a separate boolean `(H, W)` mask. Sky pixels are
  excluded from depth sanitising and from all label metrics. Sky is never a
  segmentation probability channel.
- **Resize / letterbox:** when an image is resized or letterboxed to a model
  input, intrinsics scale with it: `fx' = sx * fx`, `cx' = sx * cx + pad_x`,
  `fy' = sy * fy`, `cy' = sy * cy + pad_y`, where `pad_x`/`pad_y` are the
  left/top pad offsets. Model outputs are mapped back to the ORIGINAL image
  resolution before use. The original K remains the K of record.
- **Safety rule:** an unknown or unmapped class label can never map to floor.
  Unknown names map to other (4); sky maps to other in the label map and is
  carried separately by the sky mask.
