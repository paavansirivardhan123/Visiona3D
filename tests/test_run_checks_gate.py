"""P13 gate: the new perception tests, not the whole suite, must stay under 30 s."""

from __future__ import annotations

from geometry.scripts import run_checks

P13_NODE = "tests/test_pipeline.py::test_p13_repeat_runs_are_identical"
P13_NODES = {P13_NODE: "PASSED"}
P7_STUB = {"passed": True, "measured": "stub", "details": {}}


def _p13_row(monkeypatch, suite_seconds: float, perception_seconds: float) -> dict:
    monkeypatch.setattr(run_checks, "measure_p7_details", lambda: P7_STUB)
    rows = run_checks.build_perception_results(
        P13_NODES, suite_seconds=suite_seconds, perception_seconds=perception_seconds
    )
    return next(row for row in rows if row["id"] == "P13")


def test_p13_ignores_slow_full_suite_when_perception_tests_are_fast(monkeypatch):
    row = _p13_row(monkeypatch, suite_seconds=110.0, perception_seconds=29.9)
    assert row["passed"] is True


def test_p13_fails_when_perception_tests_reach_thirty_seconds(monkeypatch):
    row = _p13_row(monkeypatch, suite_seconds=1.0, perception_seconds=30.0)
    assert row["passed"] is False
