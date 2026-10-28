"""Offline tests for the public starter questions."""

from __future__ import annotations

import builtins
from pathlib import Path

import starter


def _feed(monkeypatch, answers):
    replies = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(replies))


def test_defaults_start_webcam_and_save_the_map(monkeypatch):
    calls = []
    monkeypatch.setattr(starter, "_run", lambda args: calls.append(args) or 0)
    _feed(monkeypatch, ["", "", "", "", "", "", "", ""])

    assert starter.menu() == 0
    assert calls == [
        [
            "--source",
            "0",
            "--profile",
            "fast",
            "--map-png",
            str(Path(starter.DEFAULT_MAP)),
            "--hfov",
            starter.DEFAULT_HFOV,
        ]
    ]


def test_video_answers_build_the_full_command(monkeypatch, tmp_path):
    video = tmp_path / "route.mp4"
    video.write_bytes(b"not a real video")
    intrinsics = tmp_path / "camera_intrinsics.json"
    intrinsics.write_text("{}", encoding="utf-8")

    calls = []
    monkeypatch.setattr(starter, "_run", lambda args: calls.append(args) or 0)
    _feed(
        monkeypatch,
        ["2", f'"{video}"', "60", "2", "no", "no", "yes", str(tmp_path / "clips") + "\\", "?", str(intrinsics)],
    )

    assert starter.menu() == 0
    assert calls == [
        [
            "--source",
            str(video),
            "--max-frames",
            "60",
            "--profile",
            "balanced",
            "--no-speech",
            "--no-voice",
            "--out-video",
            str(tmp_path / "clips" / "demo.mp4"),
            "--headless",
            "--intrinsics",
            str(intrinsics),
        ]
    ]


def test_yes_no_accepts_words_and_digits(monkeypatch):
    _feed(monkeypatch, ["maybe", "0", "yes"])
    assert starter._ask_yes_no(4, "Speech?", True) is False
    assert starter._ask_yes_no(4, "Speech?", False) is True


def test_frames_rejects_text_and_zero(monkeypatch):
    _feed(monkeypatch, ["ten", "0", "60"])
    assert starter._ask_frames(2) == 60
    _feed(monkeypatch, [""])
    assert starter._ask_frames(2) is None


def test_output_into_a_folder_gets_the_default_name(monkeypatch, tmp_path):
    folder = tmp_path / "maps"
    folder.mkdir()
    _feed(monkeypatch, [str(folder)])
    path = starter._ask_output(5, "Where?", "x", starter.DEFAULT_MAP, ".png")
    assert path == folder / "session_map.png"


def test_output_without_suffix_gets_one(monkeypatch, tmp_path):
    _feed(monkeypatch, [str(tmp_path / "walk")])
    path = starter._ask_output(5, "Where?", "x", starter.DEFAULT_MAP, ".png")
    assert path == tmp_path / "walk.png"


def test_calibration_question_prints_help(monkeypatch, capsys):
    _feed(monkeypatch, ["?", ""])
    assert starter._ask_calibration(7) is None
    assert "checkerboard" in capsys.readouterr().out.lower()


def test_main_forwards_arguments_without_questions(monkeypatch):
    calls = []
    monkeypatch.setattr(starter, "_run", lambda args: calls.append(args) or 0)
    _feed(monkeypatch, [])

    assert starter.main(["--source", "route.mp4", "--hfov", "70"]) == 0
    assert calls == [["--source", "route.mp4", "--hfov", "70"]]
