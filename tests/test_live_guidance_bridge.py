"""Speech gating for guidance and nearby objects (no TTS engine)."""

from __future__ import annotations

from types import SimpleNamespace

from live.guidance_bridge import GuidanceAnnouncer


def _guidance(decision: str, heading: float = 0.0, nearest: dict | None = None):
    return SimpleNamespace(
        decision=decision,
        heading_deg=heading,
        nearest_obstacle={} if nearest is None else nearest,
    )


def _obs(track_id: int, distance: float, class_name: str = "person"):
    return SimpleNamespace(
        track_id=track_id, distance_m=distance, class_name=class_name, status="OK"
    )


def test_stop_is_emergency_and_rate_limited():
    announcer = GuidanceAnnouncer(stop_repeat_s=2.0)
    first = announcer.update(_guidance("STOP_UNCLEAR"), now_s=10.0)
    assert first is not None
    assert first.emergency and first.priority
    assert "unclear" in first.text.lower()

    assert announcer.update(_guidance("STOP_UNCLEAR"), now_s=11.0) is None
    again = announcer.update(_guidance("STOP_UNCLEAR"), now_s=12.1)
    assert again is not None and again.emergency


def test_caution_uses_distance_and_repeats_only_after_cooldown():
    announcer = GuidanceAnnouncer(caution_repeat_s=4.0)
    guidance = _guidance(
        "CAUTION",
        nearest={"distance_nominal_m": 1.4, "bearing_deg": -25.0},
    )
    first = announcer.update(guidance, now_s=0.0)
    assert first is not None
    assert "1.4" in first.text
    assert "left" in first.text
    assert announcer.update(guidance, now_s=1.0) is None
    assert announcer.update(guidance, now_s=4.1) is not None


def test_guide_announces_heading_changes():
    announcer = GuidanceAnnouncer(guide_repeat_s=8.0, heading_change_deg=15.0)
    first = announcer.update(_guidance("GUIDE", heading=-22.0), now_s=0.0)
    assert first is not None
    assert "left" in first.text and "20" in first.text

    assert announcer.update(_guidance("GUIDE", heading=-20.0), now_s=1.0) is None
    changed = announcer.update(_guidance("GUIDE", heading=3.0), now_s=2.0)
    assert changed is not None
    assert "Clear ahead" in changed.text


def test_disabled_announcer_is_silent():
    announcer = GuidanceAnnouncer(enabled=False)
    assert announcer.update(_guidance("STOP_BLOCKED"), now_s=0.0) is None
    assert announcer.object_alert([_obs(1, 1.0)], now_s=0.0) is None


def test_object_alert_is_nearest_and_per_track():
    announcer = GuidanceAnnouncer()
    objects = [_obs(1, 1.8, "chair"), _obs(2, 0.9, "person")]
    first = announcer.object_alert(objects, now_s=0.0)
    assert first is not None
    assert first.text.startswith("Person")
    assert "0.9" in first.text
    assert announcer.object_alert(objects, now_s=1.0) is None
    assert announcer.object_alert(objects, now_s=5.1) is not None


def test_object_alert_ignores_bad_anchors_and_far_objects():
    announcer = GuidanceAnnouncer()
    bad = SimpleNamespace(track_id=1, distance_m=0.5, class_name="person", status="TINY_MASK")
    far = _obs(2, 5.0)
    assert announcer.object_alert([bad, far], now_s=0.0) is None
