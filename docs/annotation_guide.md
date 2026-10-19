# Visiona3D Annotation Guide

This guide describes how to label data for the Visiona3D single-RGB-camera navigation aid dataset.

## Mask Class Indices

| Index | Class | Description |
|-------|-------|-------------|
| 0 | unlabeled | Ignored everywhere - no annotation available |
| 1 | floor | Walkable ground plane (floor, ground, pavement) |
| 2 | wall | Vertical walls, barriers, building facades |
| 3 | obstacle | Objects above ground that impede navigation (furniture, poles, people, vehicles) |
| 4 | other | Ceiling, vegetation, sky, non-walkable flat surfaces |
| 255 | ignore | Regions that are ambiguous or unreliable |

## Explicit Labeling Rules

### Floor (Class 1)
- **Include**: Solid ground, pavement, carpet, tile, wood flooring, grass, dirt paths
- **Exclude**: Reflective floors that mirror surroundings (label as `ignore`/`glass` instead)
- **Stairs**: Label each tread as `floor`; risers as `wall` or `obstacle`
- **Curbs**: Top surface = `floor`; vertical face = `wall`
- **Ramps**: Treat as `floor` (walkable inclined plane)

### Wall (Class 2)
- **Include**: Vertical building walls, fences, railings, doors (closed), large vertical barriers
- **Exclude**: Windows/glass panes (label as `ignore`/`glass`), mirrors
- **Doors**: Closed = `wall`; Open doorway = `floor` (if passable) or `wall` (frame)

### Obstacle (Class 3)
- **Include**: People, chairs, tables, poles, bollards, parked vehicles, boxes, luggage, plants/pots
- **Partial visibility**: Label visible portions; occluded parts remain as-is
- **Height**: Any object rising >5 cm above floor that blocks path

### Other (Class 4)
- **Include**: Ceiling, sky, tree canopy, overhead signs, non-walkable horizontal surfaces (countertops, shelves)
- **Exclude**: Floor-like surfaces that are walkable

### Ignore / Unlabeled (Class 0 / 255)
- **Class 0 (unlabeled)**: No annotation available for this pixel
- **Class 255 (ignore)**: Explicitly marked as unreliable:
  - **Glass**: Transparent windows, glass doors, glass walls
  - **Mirrors**: Reflective surfaces that show reflected scene
  - **Wet floors**: Specular reflections, puddles that mirror surroundings
  - **Shadows**: Deep cast shadows where surface is ambiguous
  - **Reflective floors**: Polished marble, wet tile that acts like mirror
  - **Glare/Saturation**: Over-exposed regions
  - **Motion blur**: Severe blur making class indeterminate

## Special Cases

### Glass & Mirrors
- **Always label as 255 (ignore)** regardless of what's behind them
- Do not attempt to label "through" glass
- Tag frame with `hazard: glass` or `hazard: mirror`

### Reflective Floors / Puddles
- Label as `255 (ignore)` with `hazard: wet` tag
- Do not label as floor even if walkable - depth is unreliable

### Shadows
- **Soft shadows**: Label underlying surface normally
- **Hard/deep shadows** where texture is lost: Label as `255 (ignore)`

### Doors
- **Closed**: `wall` (2)
- **Open > 0.8m clearance**: `floor` (1) for doorway opening, `wall` for frame
- **Partially open**: Label visible parts accordingly

### Stairs & Curbs
- **Stair treads**: `floor` (1)
- **Risers**: `wall` (2) if vertical, `obstacle` (3) if sloped
- **Curb top**: `floor` (1); **Curb face**: `wall` (2)

### People
- **Standing/walking**: `obstacle` (3)
- **Sitting**: `obstacle` (3) for person; chair also `obstacle` (3)
- **Crowd**: Tag frame with `crowd: low` or `crowd: high`

## Frame Sampling Per Route

- **Minimum**: 20 frames per route
- **Recommended**: 50-100 frames per route
- **Diversity**: Sample across:
  - Different lighting (if route spans time of day)
  - Different crowd densities
  - Turns, straight corridors, open areas, doorways
  - Varying distances to walls/obstacles

## Ground-Truth Distance Points

For each frame, annotate **5-10 distance points** at known locations:
- Pixel (u, v) of the point
- True z-depth in camera frame (metres)
- Method: `rangefinder`, `lidar`, `manual`, or `synthetic`

Preferred locations:
- Floor at 1m, 2m, 3m, 4m, 5m, 6m ahead
- Wall at known distance
- Obstacle front face
- Box corners (for synthetic)

## Tagging

Each route gets consistent tags:
- `env`: `indoor` or `outdoor`
- `light`: `day`, `dusk`, `dark`
- `hazard`: `glass`, `mirror`, `wet`, `none`
- `crowd`: `none`, `low`, `high`

Per-frame tags can override route tags if conditions change.