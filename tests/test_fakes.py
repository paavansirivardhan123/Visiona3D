"""P6: fake backends reproduce ground truth and corrupt exactly as requested."""

from __future__ import annotations

import numpy as np

from core.config import MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER, Intrinsics
from perception.fakes import FakeDepthBackend, FakeDetector, FakeSegBackend


H, W = 120, 160
K = Intrinsics(fx=140.0, fy=140.0, cx=80.0, cy=60.0, width=W, height=H)
SCENE = dict(
    K=K,
    H=H,
    W=W,
    h=1.4,
    pitch_deg=15.0,
    wall_distance=4.0,
    boxes=[(-0.2, 0.2, 2.4, 2.7, 0.4, 0.8)],
    seed=5,
)
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


def test_p6_zero_corruption_reproduces_ground_truth_exactly():
    seg_backend = FakeSegBackend(**SCENE)
    depth_backend = FakeDepthBackend(**SCENE)
    detector = FakeDetector(**SCENE)

    seg = seg_backend.predict(IMAGE)
    depth = depth_backend.predict(IMAGE)
    dets = detector.predict(IMAGE)

    assert np.array_equal(depth.depth_m, seg_backend.depth_true.astype(np.float32))
    assert np.array_equal(seg.sky_mask, seg_backend.depth_true <= 0.0)
    non_sky = ~seg.sky_mask
    assert np.all(seg.labels[non_sky & seg_backend.floor_mask] == MASK_FLOOR)
    assert np.all(seg.labels[non_sky & seg_backend.wall_mask] == MASK_WALL)
    for box_mask in seg_backend.box_masks:
        assert np.all(seg.labels[non_sky & box_mask] == MASK_OBSTACLE)
    assert np.all(seg.labels[non_sky & ~(seg_backend.floor_mask | seg_backend.wall_mask | seg_backend.box_masks[0])] == MASK_OTHER)

    one_hot_index = seg.labels.astype(np.int64) - 1
    assert np.all(seg.probs[np.arange(H)[:, None], np.arange(W)[None, :], one_hot_index] == 1.0)
    assert len(dets) == 1


def test_p6_scale_factor_is_exactly_applied():
    backend = FakeDepthBackend(**SCENE, k=1.7)
    output = backend.predict(IMAGE)
    expected = (1.7 * backend.depth_true).astype(np.float32)
    assert np.array_equal(output.depth_m, expected)


def test_p6_wall_as_floor_flips_exactly_requested_pixels():
    fraction = 0.25
    backend = FakeSegBackend(**SCENE, wall_as_floor_fraction=fraction)
    output = backend.predict(IMAGE)

    wall_pixels = int(np.sum(backend.wall_mask))
    requested = int(round(fraction * wall_pixels))
    flipped = int(np.sum((output.labels == MASK_FLOOR) & backend.wall_mask))
    still_wall = int(np.sum((output.labels == MASK_WALL) & backend.wall_mask))
    assert requested > 0
    assert flipped == requested
    assert still_wall == wall_pixels - requested


def test_p6_floor_as_wall_flips_exactly_requested_pixels():
    backend = FakeSegBackend(**SCENE, floor_as_wall_fraction=0.30)
    output = backend.predict(IMAGE)
    floor_pixels = int(np.sum(backend.floor_mask))
    requested = int(round(0.30 * floor_pixels))
    flipped = int(np.sum((output.labels == MASK_WALL) & backend.floor_mask))
    assert flipped == requested


def test_p6_same_seed_gives_identical_output():
    seg_a = FakeSegBackend(**SCENE, wall_as_floor_fraction=0.2, label_noise_rate=0.05)
    seg_b = FakeSegBackend(**SCENE, wall_as_floor_fraction=0.2, label_noise_rate=0.05)
    depth_a = FakeDepthBackend(**SCENE, k=1.3, rel_noise=0.02, outlier_fraction=0.05)
    depth_b = FakeDepthBackend(**SCENE, k=1.3, rel_noise=0.02, outlier_fraction=0.05)

    out_a = seg_a.predict(IMAGE)
    out_b = seg_b.predict(IMAGE)
    assert np.array_equal(out_a.labels, out_b.labels)
    assert np.array_equal(out_a.probs, out_b.probs)
    assert np.array_equal(out_a.sky_mask, out_b.sky_mask)

    dep_a = depth_a.predict(IMAGE)
    dep_b = depth_b.predict(IMAGE)
    assert np.array_equal(dep_a.depth_m, dep_b.depth_m)

    # Repeated predictions on one instance are deterministic too.
    assert np.array_equal(depth_a.predict(IMAGE).depth_m, dep_a.depth_m)
    assert np.array_equal(seg_a.predict(IMAGE).labels, out_a.labels)
