"""O1-O6: nearest-obstacle telemetry with optional conformal bounds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mapping.obstacles import nearest_obstacle_telemetry


def _points(*rows: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def test_o1_nominal_lower_and_upper_minima():
    nominal = _points((0.0, 4.0, 0.0), (0.5, 4.2, 0.0), (0.0, 1.0, 0.0))
    lower = _points((0.0, 3.2, 0.0), (0.5, 3.0, 0.0), (0.0, 0.8, 0.0))
    upper = _points((0.0, 4.8, 0.0), (0.5, 5.0, 0.0), (0.0, 1.2, 0.0))
    mask = np.array([True, True, False])

    telemetry = nearest_obstacle_telemetry(
        nominal,
        mask,
        lower_pointmap_g=lower,
        upper_pointmap_g=upper,
    )
    assert telemetry["label"] == "obstacle"
    assert telemetry["distance_nominal_m"] == pytest.approx(4.0)
    assert telemetry["bearing_deg"] == pytest.approx(0.0)
    # Minima over obstacle pixels bound the true nearest distance.
    assert telemetry["distance_lower_m"] == pytest.approx(math.hypot(0.5, 3.0))
    assert telemetry["distance_upper_m"] == pytest.approx(4.8)
    assert telemetry["distance_lower_m"] < telemetry["distance_nominal_m"]
    assert telemetry["distance_nominal_m"] < telemetry["distance_upper_m"]


def test_o2_ignores_nan_and_out_of_mask_points():
    nominal = _points(
        (0.0, 2.0, 0.0),
        (np.nan, 1.0, 0.0),
        (0.0, np.nan, 0.0),
        (0.0, 1.5, 0.0),
    )
    mask = np.array([False, True, True, True])
    telemetry = nearest_obstacle_telemetry(nominal, mask)
    assert telemetry["distance_nominal_m"] == pytest.approx(1.5)


def test_o3_no_obstacle_returns_none_and_infinity():
    nominal = _points((0.0, 2.0, 0.0))
    telemetry = nearest_obstacle_telemetry(
        nominal,
        np.array([False]),
        lower_pointmap_g=nominal,
        upper_pointmap_g=nominal,
    )
    assert telemetry["label"] == "none"
    assert math.isinf(telemetry["distance_nominal_m"])
    assert "distance_lower_m" not in telemetry
    assert "distance_upper_m" not in telemetry


def test_o4_missing_or_unbounded_bounds_are_not_fabricated():
    nominal = _points((0.0, 4.0, 0.0))
    mask = np.array([True])

    no_bounds = nearest_obstacle_telemetry(nominal, mask)
    assert "distance_lower_m" not in no_bounds
    assert "distance_upper_m" not in no_bounds

    lower = _points((0.0, 3.2, 0.0))
    upper = _points((np.nan, np.nan, np.nan))
    one_sided = nearest_obstacle_telemetry(
        nominal, mask, lower_pointmap_g=lower, upper_pointmap_g=upper
    )
    assert one_sided["distance_lower_m"] == pytest.approx(3.2)
    assert "distance_upper_m" not in one_sided


def test_o5_bearing_is_positive_right():
    nominal = _points((1.0, 1.0, 0.0))
    telemetry = nearest_obstacle_telemetry(nominal, np.array([True]))
    assert telemetry["bearing_deg"] == pytest.approx(45.0)


def test_o6_shape_mismatches_raise():
    nominal = _points((0.0, 4.0, 0.0), (0.0, 5.0, 0.0))
    with pytest.raises(ValueError, match="obstacle_mask must match"):
        nearest_obstacle_telemetry(nominal, np.array([True]))
    with pytest.raises(ValueError, match="same number of points"):
        nearest_obstacle_telemetry(
            nominal,
            np.array([True, False]),
            lower_pointmap_g=_points((0.0, 3.0, 0.0)),
        )
