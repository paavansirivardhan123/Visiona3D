"""Offline tests for the public starter menu."""

from __future__ import annotations

import builtins

import starter


def _feed(monkeypatch, answers):
    replies = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(replies))


def test_enter_starts_webcam_with_estimated_distances(monkeypatch):
    calls = []
    monkeypatch.setattr(starter, "_run", lambda args: calls.append(args) or 0)
    _feed(monkeypatch, ["", "0", "0"])

    assert starter.menu() == 0
    assert calls == [["--source", "0", "--hfov", "70"]]


def test_video_choice_needs_only_a_file_path(monkeypatch, tmp_path):
    video = tmp_path / "route.mp4"
    video.write_bytes(b"not a real video")

    calls = []
    monkeypatch.setattr(starter, "_run", lambda args: calls.append(args) or 0)
    _feed(monkeypatch, ["2", f'"{video}"', "0", "0"])

    assert starter.menu() == 0
    assert calls == [["--source", str(video), "--hfov", "70"]]


def test_advanced_webcam_uses_calibration_file(monkeypatch, tmp_path):
    intrinsics = tmp_path / "camera_intrinsics.json"
    intrinsics.write_text("{}", encoding="utf-8")

    calls = []
    monkeypatch.setattr(starter, "_run", lambda args: calls.append(args) or 0)
    _feed(monkeypatch, ["3", "1", str(intrinsics), "0", "0", "0"])

    assert starter.menu() == 0
    assert calls == [["--source", "0", "--intrinsics", str(intrinsics)]]
