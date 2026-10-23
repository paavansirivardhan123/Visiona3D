"""Guidance and object speech gating for the live system.

The planner produces structured recommendations; the legacy ``SpeechEngine``
speaks strings. This module is the bridge, with explicit priorities so a STOP
is never queued behind a navigation suggestion and repeated text does not
chatter.

All intervals are TO TUNE ON REAL DATA.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

DEFAULT_STOP_REPEAT_S = 2.0  # TO TUNE ON REAL DATA
DEFAULT_CAUTION_REPEAT_S = 4.0  # TO TUNE ON REAL DATA
DEFAULT_GUIDE_REPEAT_S = 8.0  # TO TUNE ON REAL DATA
DEFAULT_HEADING_CHANGE_DEG = 15.0  # TO TUNE ON REAL DATA
DEFAULT_CAUTION_DISTANCE_CHANGE_M = 0.5  # TO TUNE ON REAL DATA
DEFAULT_OBJECT_MAX_DISTANCE_M = 2.0  # TO TUNE ON REAL DATA
DEFAULT_OBJECT_REPEAT_S = 5.0  # TO TUNE ON REAL DATA

STOP_DECISIONS = ("STOP_UNCLEAR", "STOP_BLOCKED")


@dataclass(frozen=True)
class Announcement:
    """One gated utterance ready for ``SpeechEngine.speak``."""

    text: str
    kind: str  # "stop" | "caution" | "guide" | "object"
    priority: bool = False
    emergency: bool = False


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _turn_phrase(heading_deg: float) -> str:
    magnitude = abs(float(heading_deg))
    if magnitude < 10.0:
        return "Clear ahead. Continue."
    degrees = int(round(magnitude / 5.0) * 5)
    side = "right" if heading_deg > 0.0 else "left"
    return f"Bear {side} {degrees} degrees."


class GuidanceAnnouncer:
    """Rate-limit guidance and nearby-object announcements."""

    def __init__(
        self,
        enabled: bool = True,
        stop_repeat_s: float = DEFAULT_STOP_REPEAT_S,
        caution_repeat_s: float = DEFAULT_CAUTION_REPEAT_S,
        guide_repeat_s: float = DEFAULT_GUIDE_REPEAT_S,
        heading_change_deg: float = DEFAULT_HEADING_CHANGE_DEG,
        caution_distance_change_m: float = DEFAULT_CAUTION_DISTANCE_CHANGE_M,
    ) -> None:
        self.enabled = bool(enabled)
        self.stop_repeat_s = float(stop_repeat_s)
        self.caution_repeat_s = float(caution_repeat_s)
        self.guide_repeat_s = float(guide_repeat_s)
        self.heading_change_deg = float(heading_change_deg)
        self.caution_distance_change_m = float(caution_distance_change_m)
        self._last_kind: str | None = None
        self._last_time: dict[str, float] = {}
        self._last_heading: float | None = None
        self._last_caution_m: float | None = None
        self._last_object_time: dict[int, float] = {}

    def update(self, guidance: Any, now_s: float) -> Announcement | None:
        """Return the next guidance utterance, or None if suppressed."""
        if not self.enabled or guidance is None:
            return None
        decision = str(getattr(guidance, "decision", ""))
        if decision not in ("GUIDE", "CAUTION", *STOP_DECISIONS):
            return None

        heading = _finite(getattr(guidance, "heading_deg", 0.0)) or 0.0
        nearest = getattr(guidance, "nearest_obstacle", None) or {}
        distance = _finite(nearest.get("distance_nominal_m"))
        bearing = _finite(nearest.get("bearing_deg"))

        if decision in STOP_DECISIONS:
            kind = "stop"
            if decision == "STOP_BLOCKED":
                text = "Stop. The path is blocked."
            else:
                text = "Stop. The path is unclear."
            return self._gate(
                kind=kind,
                text=text,
                now_s=now_s,
                repeat_s=self.stop_repeat_s,
                priority=True,
                emergency=True,
                force_change=self._last_kind != kind,
            )

        if decision == "CAUTION":
            kind = "caution"
            if distance is None:
                text = "Caution. Obstacle nearby."
            else:
                side = ""
                if bearing is not None and abs(bearing) >= 20.0:
                    side = " to your right" if bearing > 0.0 else " to your left"
                text = f"Caution. Obstacle {distance:.1f} metres ahead{side}."
            changed = (
                self._last_caution_m is None
                or distance is None
                or abs(distance - self._last_caution_m) >= self.caution_distance_change_m
            )
            announcement = self._gate(
                kind=kind,
                text=text,
                now_s=now_s,
                repeat_s=self.caution_repeat_s,
                priority=True,
                emergency=False,
                force_change=self._last_kind != kind or changed,
            )
            if announcement is not None:
                self._last_caution_m = distance if distance is not None else self._last_caution_m
            return announcement

        kind = "guide"
        text = _turn_phrase(heading)
        heading_changed = (
            self._last_heading is None
            or abs(heading - self._last_heading) >= self.heading_change_deg
        )
        announcement = self._gate(
            kind=kind,
            text=text,
            now_s=now_s,
            repeat_s=self.guide_repeat_s,
            priority=False,
            emergency=False,
            force_change=self._last_kind != kind or heading_changed,
        )
        if announcement is not None:
            self._last_heading = heading
        return announcement

    def object_alert(
        self,
        observations: list,
        now_s: float,
        max_distance_m: float = DEFAULT_OBJECT_MAX_DISTANCE_M,
        repeat_s: float = DEFAULT_OBJECT_REPEAT_S,
    ) -> Announcement | None:
        """Announce the nearest close object once per track and cooldown."""
        if not self.enabled or not observations:
            return None
        candidates = [
            obs
            for obs in observations
            if str(getattr(obs, "status", "")) == "OK"
            and (distance := _finite(getattr(obs, "distance_m", None))) is not None
            and distance <= float(max_distance_m)
        ]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda obs: float(obs.distance_m))
        track_id = int(getattr(nearest, "track_id", -1))
        last = self._last_object_time.get(track_id)
        if last is not None and now_s - last < repeat_s:
            return None
        self._last_object_time[track_id] = float(now_s)
        name = str(getattr(nearest, "class_name", "object"))
        text = f"{name.capitalize()}, {float(nearest.distance_m):.1f} metres ahead."
        return Announcement(text=text, kind="object", priority=False, emergency=False)

    def _gate(
        self,
        *,
        kind: str,
        text: str,
        now_s: float,
        repeat_s: float,
        priority: bool,
        emergency: bool,
        force_change: bool,
    ) -> Announcement | None:
        last = self._last_time.get(kind)
        if not force_change and last is not None and now_s - last < repeat_s:
            return None
        self._last_time[kind] = float(now_s)
        self._last_kind = kind
        return Announcement(text=text, kind=kind, priority=priority, emergency=emergency)

    def reset(self) -> None:
        self._last_kind = None
        self._last_time.clear()
        self._last_heading = None
        self._last_caution_m = None
        self._last_object_time.clear()


__all__ = [
    "Announcement",
    "DEFAULT_CAUTION_REPEAT_S",
    "DEFAULT_GUIDE_REPEAT_S",
    "DEFAULT_OBJECT_MAX_DISTANCE_M",
    "DEFAULT_STOP_REPEAT_S",
    "GuidanceAnnouncer",
    "STOP_DECISIONS",
]
