"""Offline tests for real-time video playback and HUD reasons.

No models, camera, video file or window is used here.
"""

from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np

from live.overlay import decision_reason
from live.runner import LiveRunner


def _guidance(decision: str, reasons: list[str]) -> SimpleNamespace:
    return SimpleNamespace(decision=decision, reasons=reasons)


class TestDecisionReason:
    def test_stop_returns_first_abstention_reason(self):
        guidance = _guidance(
            "STOP_UNCLEAR",
            ["certified floor fraction in corridor 0.000 < 0.300"],
        )
        assert decision_reason(guidance).startswith("certified floor fraction")

    def test_generic_pass_message_is_not_shown(self):
        guidance = _guidance("STOP_UNCLEAR", ["all abstention checks passed"])
        assert decision_reason(guidance) == ""

    def test_guide_has_no_reason(self):
        guidance = _guidance("GUIDE", ["Path certified and clear; guidance active"])
        assert decision_reason(guidance) == ""

    def test_missing_guidance_is_safe(self):
        assert decision_reason(None) == ""


class _FakeSource:
    def __init__(self, frames, stop_on_rewind=None):
        self.frames = list(frames)
        self.index = 0
        self.rewinds = 0
        self.stop_on_rewind = stop_on_rewind

    def read(self):
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def rewind(self):
        self.rewinds += 1
        self.index = 0
        if self.stop_on_rewind is not None:
            self.stop_on_rewind.set()


def _bare_runner(frames, fps: float, loop: bool = False) -> LiveRunner:
    runner = LiveRunner.__new__(LiveRunner)
    runner.config = SimpleNamespace(loop=loop, start_frame=0)
    runner.info = SimpleNamespace(fps=fps)
    runner.source = _FakeSource(frames)
    runner._stop = threading.Event()
    runner._cond = threading.Condition()
    runner._frame = None
    runner._capture_done = False
    runner._capture_index = 0
    runner._resize = lambda frame: frame
    return runner


class TestFilePlayback:
    def test_reader_keeps_only_the_newest_frame_and_marks_done(self):
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
        runner = _bare_runner(frames, fps=1000.0)

        runner._capture_file_loop()

        assert runner._capture_done is True
        assert runner._frame is not None
        assert runner._frame[0] == 3
        assert runner._frame[2] is frames[-1]
        assert runner._capture_index == 3

    def test_loop_rewinds_and_stops_when_asked(self):
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
        runner = _bare_runner(frames, fps=1000.0, loop=True)
        runner.source.stop_on_rewind = runner._stop

        runner._capture_file_loop()

        assert runner.source.rewinds == 1
        assert runner._capture_done is True
