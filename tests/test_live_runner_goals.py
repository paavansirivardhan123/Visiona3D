"""Offline tests for goal/voice wiring in LiveRunner (no models or devices)."""

from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest

from live.anchors import ObjectAnchor
from live.goals import GoalSpec
from live.runner import LiveConfig, LiveRunner


class _FakeSpeech:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, dict]] = []
        self.ducked = False

    def speak(self, text, **kwargs):
        self.spoken.append((text, kwargs))

    def duck(self):
        self.ducked = True

    def unduck(self):
        self.ducked = False


def _bare_runner() -> LiveRunner:
    runner = LiveRunner.__new__(LiveRunner)
    runner.config = SimpleNamespace(arrival_distance_m=1.5, arrival_stable_frames=3)
    runner._goal_lock = threading.Lock()
    runner._goal_tracker = None
    runner._goal_agent = None
    runner._last_index = 0
    runner._pending_goal_events = []
    runner._speech_enabled = True
    runner._voice_listening = False
    runner._voice_hint_given = False
    runner._last_voice_error_s = {}
    runner._errors = []
    runner.speech = _FakeSpeech()
    return runner


def _obs(class_name="chair", distance_m=1.2, track_id=1, lateral_m=0.0):
    return ObjectAnchor(
        track_id=track_id,
        class_name=class_name,
        conf=0.9,
        box_xyxy=(0.0, 0.0, 10.0, 10.0),
        distance_m=distance_m,
        position_g=(lateral_m, distance_m, 0.0),
        status="OK",
    )


class TestVoiceTranscriptWiring:
    def test_sit_transcript_sets_expanded_goal_and_speaks(self):
        runner = _bare_runner()
        runner._on_voice_transcript("I want to sit")
        assert runner._goal_tracker is not None
        assert runner._goal_tracker.spec.candidates == (
            "chair",
            "bench",
            "couch",
            "bed",
        )
        assert len(runner._pending_goal_events) == 1
        kind, text = runner._pending_goal_events[0][1:]
        assert kind == "goal_set"
        assert "chair" in text
        assert runner.speech.spoken
        assert "Goal set" in runner.speech.spoken[-1][0]

    def test_cancel_transcript_clears_goal(self):
        runner = _bare_runner()
        runner._on_voice_transcript("I want to sit")
        runner._on_voice_transcript("Cancel the goal")
        assert runner._goal_tracker is None
        kinds = [event[1] for event in runner._pending_goal_events]
        assert kinds == ["goal_set", "goal_cancelled"]
        assert "cancelled" in runner.speech.spoken[-1][0].lower()

    def test_unknown_transcript_becomes_generic_goal(self):
        runner = _bare_runner()
        runner._on_voice_transcript("find my red umbrella")
        assert runner._goal_tracker is not None
        assert runner._goal_tracker.spec.candidates == ("red umbrella", "umbrella")

    def test_blank_transcript_is_ignored(self):
        runner = _bare_runner()
        runner._on_voice_transcript("   ")
        assert runner._goal_tracker is None
        assert runner._pending_goal_events == []
        assert runner.speech.spoken == []


class TestVoiceDucking:
    def test_listening_ducks_and_release_unducks(self):
        runner = _bare_runner()
        runner._on_voice_listening(True)
        assert runner.speech.ducked is True
        runner._on_voice_listening(False)
        assert runner.speech.ducked is False

    def test_muted_speech_stays_ducked(self):
        runner = _bare_runner()
        runner._speech_enabled = False
        runner._on_voice_listening(True)
        assert runner.speech.ducked is False
        runner._on_voice_listening(False)
        assert runner.speech.ducked is False


class _FakeAgent:
    def __init__(self, spec=None, error=None):
        self.available = True
        self.spec = spec
        self.last_error = error
        self.calls: list[str] = []

    def interpret(self, text):
        self.calls.append(text)
        return self.spec


class TestGeminiGoalWiring:
    def test_agent_spec_is_used_and_event_is_labelled(self):
        runner = _bare_runner()
        runner._goal_agent = _FakeAgent(GoalSpec("sit", ("bench",), "seat"))
        runner._on_voice_transcript("I could really use a seat")
        assert runner._goal_tracker is not None
        assert runner._goal_tracker.spec.candidates == ("bench",)
        kind, text = runner._pending_goal_events[-1][1:]
        assert kind == "goal_set"
        assert "[gemini]" in text
        assert "Goal set" in runner.speech.spoken[-1][0]

    def test_agent_failure_falls_back_to_local_parser(self):
        runner = _bare_runner()
        runner._goal_agent = _FakeAgent(None, "timeout")
        runner._on_voice_transcript("I want to sit")
        assert runner._goal_tracker is not None
        assert runner._goal_tracker.spec.candidates == (
            "chair",
            "bench",
            "couch",
            "bed",
        )
        kind, text = runner._pending_goal_events[-1][1:]
        assert kind == "goal_set"
        assert "[local]" in text
        assert any("goal agent: timeout" in error for error in runner._errors)

    def test_agent_cancel_intent_clears_the_goal(self):
        runner = _bare_runner()
        runner._on_voice_transcript("I want to sit")
        runner._goal_agent = _FakeAgent(GoalSpec("cancel", (), ""))
        runner._on_voice_transcript("I am done sitting")
        assert runner._goal_tracker is None
        assert runner._pending_goal_events[-1][1] == "goal_cancelled"

    def test_local_parser_used_when_agent_is_disabled(self):
        runner = _bare_runner()
        runner._goal_agent = None
        runner._on_voice_transcript("I want to sit")
        assert runner._goal_tracker is not None
        assert runner._goal_tracker.spec.candidates == (
            "chair",
            "bench",
            "couch",
            "bed",
        )
        kind, text = runner._pending_goal_events[-1][1:]
        assert kind == "goal_set"
        assert "[local]" in text


class TestGoalStatusAndConfig:
    def test_goal_line_reports_candidate_and_search(self):
        runner = _bare_runner()
        runner._on_voice_transcript("I want to sit")
        tracker = runner._goal_tracker
        line = runner._goal_line(tracker, _obs("couch", 2.4))
        assert "couch 2.4 m" in line
        line = runner._goal_line(tracker, None)
        assert "searching" in line
        assert "chair" in line

    def test_goal_line_reports_arrival(self):
        runner = _bare_runner()
        runner._on_voice_transcript("I want to sit")
        tracker = runner._goal_tracker
        for _ in range(tracker.stable_frames):
            text = tracker.update(_obs("chair", 1.0))
        assert text is not None
        assert runner._goal_line(tracker, _obs("chair", 1.0)) == "goal sit: reached"

    def test_config_rejects_two_goal_sources(self):
        config = LiveConfig(hfov_deg=70.0, goal_text="sit", goal_class="chair")
        with pytest.raises(ValueError):
            config.validate()

    def test_config_rejects_bad_arrival_settings(self):
        with pytest.raises(ValueError):
            LiveConfig(hfov_deg=70.0, arrival_distance_m=0.0).validate()
        with pytest.raises(ValueError):
            LiveConfig(hfov_deg=70.0, arrival_stable_frames=0).validate()

    def test_config_accepts_natural_language_goal(self):
        LiveConfig(hfov_deg=70.0, goal_text="I want to sit").validate()
