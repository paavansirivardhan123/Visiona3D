"""Offline tests for deterministic goals and one-shot arrival."""

from __future__ import annotations

import math

import pytest

from live.anchors import ObjectAnchor
from live.goals import (
    UNKNOWN_DISTANCE_M,
    GoalTracker,
    arrival_message,
    build_goal_tracker,
    canonical_class,
    is_cancel_goal,
    parse_goal_text,
    select_goal_candidate,
)


def _obs(
    class_name: str,
    distance_m: float,
    track_id: int = 1,
    status: str = "OK",
    lateral_m: float = 0.0,
) -> ObjectAnchor:
    return ObjectAnchor(
        track_id=track_id,
        class_name=class_name,
        conf=0.9,
        box_xyxy=(0.0, 0.0, 10.0, 10.0),
        distance_m=distance_m,
        position_g=(float(lateral_m), float(distance_m), 0.0),
        status=status,
    )


class TestGoalParsing:
    def test_sit_expands_to_chair_bench_couch_bed(self):
        spec = parse_goal_text("I want to sit")
        assert spec.intent == "sit"
        assert spec.candidates == ("chair", "bench", "couch", "bed")

    def test_sit_down_and_take_a_seat_use_same_expansion(self):
        for text in ("I want to sit down", "please take a seat", "I am looking for a seat"):
            assert parse_goal_text(text).candidates == (
                "chair",
                "bench",
                "couch",
                "bed",
            )

    def test_water_expands_to_containers_and_sources(self):
        spec = parse_goal_text("I need some water")
        assert spec.intent == "water"
        assert "bottle" in spec.candidates
        assert "cup" in spec.candidates
        assert "sink" in spec.candidates
        assert "refrigerator" in spec.candidates

    def test_restaurant_does_not_trigger_rest(self):
        spec = parse_goal_text("find a restaurant")
        assert spec.intent == "restaurant"
        assert "chair" not in spec.candidates

    def test_unknown_goal_keeps_phrase_and_head_noun(self):
        spec = parse_goal_text("find my red umbrella")
        assert spec.candidates == ("red umbrella", "umbrella")

    def test_direct_class_name_is_kept(self):
        spec = parse_goal_text("chair")
        assert spec.candidates == ("chair",)

    def test_sofa_is_canonicalised_to_detector_name_couch(self):
        assert canonical_class("sofas") == "couch"
        assert canonical_class("couch") == "couch"

    def test_plural_candidates_match_singular_detections(self):
        assert canonical_class("benches") == "bench"
        assert canonical_class("wine glasses") == "wine glass"

    def test_empty_goal_is_rejected(self):
        with pytest.raises(ValueError):
            parse_goal_text("   ")

    def test_cancel_phrases_are_recognised(self):
        assert is_cancel_goal("please cancel the goal")
        assert is_cancel_goal("Stop the goal now")
        assert not is_cancel_goal("I want to sit")

    def test_unknown_distance_sentinel_matches_legacy(self):
        assert UNKNOWN_DISTANCE_M == 999.0


class TestCandidateSelection:
    def test_nearest_candidate_wins_across_classes(self):
        observations = [
            _obs("chair", 3.0, track_id=1),
            _obs("bench", 1.8, track_id=2),
            _obs("person", 0.5, track_id=3),
        ]
        chosen = select_goal_candidate(observations, ("chair", "bench", "couch"))
        assert chosen is not None
        assert chosen.class_name == "bench"

    def test_synonym_candidate_matches_detector_class(self):
        chosen = select_goal_candidate([_obs("couch", 2.0)], ("sofa",))
        assert chosen is not None
        assert chosen.class_name == "couch"

    def test_non_ok_anchor_is_never_selected(self):
        observations = [
            _obs("chair", 0.8, track_id=1, status="TINY_MASK"),
            _obs("chair", 2.4, track_id=2, status="OK"),
        ]
        chosen = select_goal_candidate(observations, ("chair",))
        assert chosen is not None
        assert chosen.track_id == 2

    def test_no_usable_candidate_returns_none(self):
        assert select_goal_candidate([_obs("chair", 0.0, status="EMPTY_MASK")], ("chair",)) is None
        assert select_goal_candidate([], ("chair",)) is None

    def test_class_priority_breaks_distance_ties(self):
        observations = [
            _obs("bench", 2.0, track_id=1),
            _obs("chair", 2.0, track_id=2),
        ]
        chosen = select_goal_candidate(observations, ("chair", "bench"))
        assert chosen is not None
        assert chosen.class_name == "chair"


class TestGoalArrival:
    def test_arrival_needs_stable_frames_and_fires_once(self):
        tracker = GoalTracker(
            parse_goal_text("I want to sit"),
            arrival_distance_m=1.5,
            stable_frames=3,
        )
        near = _obs("couch", 1.4, track_id=7)
        assert tracker.update(near) is None
        assert tracker.update(near) is None
        message = tracker.update(near)
        assert message is not None
        assert message.startswith("You have reached your goal destination.")
        assert "couch" in message
        assert tracker.update(near) is None
        assert tracker.active is False

    def test_arrival_does_not_fire_above_threshold(self):
        tracker = GoalTracker(
            parse_goal_text("I want to sit"),
            arrival_distance_m=1.5,
            stable_frames=2,
        )
        far = _obs("chair", 1.6)
        assert tracker.update(far) is None
        assert tracker.update(far) is None
        assert tracker.active is True

    def test_distance_above_threshold_resets_streak(self):
        tracker = GoalTracker(
            parse_goal_text("I want to sit"),
            arrival_distance_m=1.5,
            stable_frames=3,
        )
        near = _obs("chair", 1.2, track_id=4)
        far = _obs("chair", 2.0, track_id=4)
        assert tracker.update(near) is None
        assert tracker.update(near) is None
        assert tracker.update(far) is None
        assert tracker.update(near) is None
        assert tracker.update(near) is None
        assert tracker.update(near) is not None

    def test_track_change_resets_streak(self):
        tracker = GoalTracker(
            parse_goal_text("I want to sit"),
            arrival_distance_m=1.5,
            stable_frames=3,
        )
        first = _obs("chair", 1.2, track_id=1)
        second = _obs("bench", 1.2, track_id=2)
        assert tracker.update(first) is None
        assert tracker.update(first) is None
        assert tracker.update(second) is None
        assert tracker.update(second) is None
        assert tracker.update(second) is not None

    def test_missing_or_non_ok_candidate_resets_streak(self):
        tracker = GoalTracker(
            parse_goal_text("I want to sit"),
            arrival_distance_m=1.5,
            stable_frames=2,
        )
        near = _obs("chair", 1.01)
        assert tracker.update(near) is None
        assert tracker.update(None) is None
        assert tracker.update(near) is None
        assert tracker.update(near) is not None

    def test_arrival_direction_is_spoken(self):
        message = arrival_message(_obs("chair", 0.8, lateral_m=1.5))
        assert "to your right" in message
        message = arrival_message(_obs("chair", 0.8, lateral_m=-1.5))
        assert "to your left" in message
        message = arrival_message(_obs("chair", 0.8, lateral_m=0.0))
        assert "straight ahead" in message

    def test_goal_tracker_selects_from_expanded_candidates(self):
        tracker = GoalTracker(parse_goal_text("I want to sit"))
        chosen = tracker.select(
            [
                _obs("bottle", 0.8, track_id=1),
                _obs("couch", 2.5, track_id=2),
            ]
        )
        assert chosen is not None
        assert chosen.class_name == "couch"

    def test_build_tracker_from_class_or_text(self):
        from_text = build_goal_tracker(goal_text="I want to sit")
        from_class = build_goal_tracker(goal_class="chair")
        assert from_text is not None
        assert from_text.spec.candidates == ("chair", "bench", "couch", "bed")
        assert from_class is not None
        assert from_class.spec.candidates == ("chair",)
        assert build_goal_tracker() is None
        with pytest.raises(ValueError):
            build_goal_tracker(goal_text="sit", goal_class="chair")


def test_arrival_message_is_finite_for_nan_bearing():
    observation = _obs("chair", 1.0)
    object.__setattr__(observation, "position_g", (math.nan, 1.0, 0.0))
    message = arrival_message(observation)
    assert "straight ahead" in message
