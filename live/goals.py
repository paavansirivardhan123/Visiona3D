"""Deterministic user goals for the live system.

A spoken goal such as "I want to sit" does not name one detector class. It
expands to a candidate set (chair, bench, couch, bed), and the planner routes
to whichever candidate is nearest in the current frame. Candidates that are not
detected are simply absent; their distance is conceptually the legacy "999 m"
unknown sentinel and they can never trigger arrival.

The interpreter is local and deterministic: it needs no cloud LLM, no network
and no API key. A goal only chooses where to plan; it never certifies free
space, and every navigation decision still comes from the safety stack.

All numeric defaults are TO TUNE ON REAL DATA.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

from live.anchors import ObjectAnchor

UNKNOWN_DISTANCE_M = 999.0  # legacy unknown-distance sentinel
DEFAULT_GOAL_ARRIVAL_M = 1.5  # TO TUNE ON REAL DATA
DEFAULT_ARRIVAL_STABLE_FRAMES = 3  # TO TUNE ON REAL DATA
DEFAULT_ARRIVAL_CONE_DEG = 10.0  # TO TUNE ON REAL DATA; "ahead" half-angle

GOAL_MAP: dict[str, tuple[str, ...]] = {
    # Class names follow the detector vocabulary (YOLO calls a sofa "couch").
    "water": ("bottle", "cup", "wine glass", "sink", "refrigerator"),
    "sit": ("chair", "bench", "couch", "bed"),
    "rest": ("chair", "bench", "couch", "bed", "dining table"),
    "food": (
        "apple",
        "banana",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "pizza",
        "donut",
        "cake",
        "refrigerator",
        "dining table",
    ),
    "exit": ("door", "stairs"),
    "wallet": ("handbag", "backpack"),
    "money": ("handbag", "backpack"),
}

GOAL_ALIASES: dict[str, tuple[str, ...]] = {
    "water": ("water", "drink", "thirsty", "thirst", "hydration"),
    "sit": ("sit", "sitting", "seat", "seated", "sit down", "take a seat", "have a seat"),
    "rest": ("rest", "resting", "relax", "lie down", "nap", "sleep", "sleepy", "tired"),
    "food": ("food", "eat", "eating", "hungry", "meal", "snack"),
    "exit": ("exit", "leave", "way out", "outside", "get out", "go out", "home", "go home"),
    "wallet": ("wallet", "purse", "handbag", "bag", "backpack"),
    "money": ("money", "cash", "pay", "payment"),
}

_CANCEL_PHRASES = (
    "cancel goal",
    "cancel the goal",
    "stop goal",
    "stop the goal",
    "clear goal",
    "clear the goal",
    "forget the goal",
    "never mind",
    "no goal",
)

# Detector names are not the words users say. Canonicalise to detector names.
_CLASS_SYNONYMS = {
    "sofa": "couch",
    "settee": "couch",
    "fridge": "refrigerator",
    "table": "dining table",
    "tv": "tv",
    "television": "tv",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
}

_IRREGULAR_SINGULAR = {
    "glasses": "glass",
    "benches": "bench",
    "couches": "couch",
    "chairs": "chair",
    "beds": "bed",
    "bottles": "bottle",
    "cups": "cup",
    "stairs": "stair",
    "keys": "key",
    "doors": "door",
    "tables": "table",
    "handbags": "handbag",
    "backpacks": "backpack",
}

_LEAD_PHRASE = re.compile(
    r"^(?:(?:please|hey|ok|okay)\s+)*"
    r"(?:"
    r"i\s+(?:really\s+)?(?:want|need|would\s+like|wanna|am\s+looking\s+for)\s+(?:to\s+)?"
    r"|can\s+you\s+(?:please\s+)?(?:find|take\s+me\s+to|show\s+me)\s+"
    r"|find(?:\s+me)?\s+"
    r"|where\s+(?:is|are)\s+(?:the\s+)?"
    r"|take\s+me\s+to\s+|lead\s+me\s+to\s+|navigate\s+to\s+|bring\s+me\s+to\s+"
    r")"
)
_LEADING_DETERMINER = re.compile(r"^(?:the|a|an|my|our|your|some)\s+")
_TRAILING_POLITE = re.compile(r"\s+(?:please|now|thanks|thank you)$")


@dataclass(frozen=True)
class GoalSpec:
    """One active goal: the user's words plus the detector classes to seek."""

    intent: str
    candidates: tuple[str, ...]
    text: str

    @property
    def candidates_text(self) -> str:
        return ", ".join(self.candidates)


def _normalize(text: str) -> str:
    value = str(text or "").lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _singular_last_word(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    last = words[-1]
    if last in _IRREGULAR_SINGULAR:
        words[-1] = _IRREGULAR_SINGULAR[last]
    elif last.endswith("ies") and len(last) > 3:
        words[-1] = last[:-3] + "y"
    elif last.endswith("s") and not last.endswith("ss"):
        words[-1] = last[:-1]
    return " ".join(words)


def canonical_class(name: str) -> str:
    """Normalise a detector class name for matching across synonyms."""
    text = _singular_last_word(_normalize(name))
    return _CLASS_SYNONYMS.get(text, text)


def _phrase_present(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _clean_generic(normalized: str) -> str:
    text = _LEAD_PHRASE.sub("", normalized).strip()
    text = _LEADING_DETERMINER.sub("", text).strip()
    text = _TRAILING_POLITE.sub("", text).strip()
    return text or normalized


def _add_candidate(candidates: list[str], seen: set[str], name: str) -> None:
    clean = _normalize(name)
    if not clean:
        return
    key = canonical_class(clean)
    if key in seen:
        return
    seen.add(key)
    candidates.append(clean)


def parse_goal_text(text: str) -> GoalSpec:
    """Expand a natural-language goal into detector classes.

    Args:
        text: A spoken or typed goal, for example ``"I want to sit"``.

    Returns:
        A GoalSpec whose ``candidates`` are ordered by intent priority. For an
        unknown goal the cleaned phrase and its head noun are candidates, so
        "find my red umbrella" can still match a detected ``umbrella``.

    Raises:
        ValueError: If the text contains no letters or digits.
    """
    raw = str(text or "").strip()
    normalized = _normalize(raw)
    if not normalized:
        raise ValueError("goal text is empty")

    matched: list[str] = []
    candidates: list[str] = []
    seen: set[str] = set()
    for intent, aliases in GOAL_ALIASES.items():
        if not any(_phrase_present(normalized, alias) for alias in aliases):
            continue
        matched.append(intent)
        for name in GOAL_MAP[intent]:
            _add_candidate(candidates, seen, name)

    if matched:
        return GoalSpec(intent="+".join(matched), candidates=tuple(candidates), text=raw)

    cleaned = _clean_generic(normalized)
    _add_candidate(candidates, seen, cleaned)
    if " " in cleaned:
        _add_candidate(candidates, seen, cleaned.split()[-1])
    return GoalSpec(intent=cleaned, candidates=tuple(candidates), text=raw)


def is_cancel_goal(text: str) -> bool:
    """True when the transcript explicitly cancels the active goal."""
    normalized = _normalize(text)
    return any(_phrase_present(normalized, phrase) for phrase in _CANCEL_PHRASES)


def _usable(observation: ObjectAnchor) -> bool:
    status = str(getattr(observation, "status", ""))
    distance = float(getattr(observation, "distance_m", float("nan")))
    position = getattr(observation, "position_g", ())
    if status != "OK" or not math.isfinite(distance) or distance <= 0.0:
        return False
    try:
        values = [float(v) for v in position]
    except (TypeError, ValueError):
        return False
    return len(values) == 3 and all(math.isfinite(value) for value in values)


def select_goal_candidate(
    observations: Sequence[ObjectAnchor],
    candidates: Sequence[str],
) -> ObjectAnchor | None:
    """Return the nearest usable observation among the goal's classes.

    Non-OK anchors behave like the unknown 999 m sentinel: they are never
    selected, and arrival cannot fire from them.
    """
    order: dict[str, int] = {}
    for index, name in enumerate(candidates):
        order.setdefault(canonical_class(name), index)
    if not order:
        return None

    ranked: list[tuple[float, int, int, ObjectAnchor]] = []
    for observation in observations:
        key = canonical_class(str(getattr(observation, "class_name", "")))
        if key not in order or not _usable(observation):
            continue
        ranked.append(
            (
                float(observation.distance_m),
                order[key],
                int(getattr(observation, "track_id", -1)),
                observation,
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[:3])
    return ranked[0][3]


def _direction_phrase(bearing_deg: float, cone_deg: float) -> str:
    if not math.isfinite(bearing_deg) or abs(bearing_deg) <= cone_deg:
        return "straight ahead"
    return "to your right" if bearing_deg > 0.0 else "to your left"


def arrival_message(
    observation: ObjectAnchor,
    cone_deg: float = DEFAULT_ARRIVAL_CONE_DEG,
) -> str:
    """One spoken sentence for a reached goal."""
    label = _normalize(str(getattr(observation, "class_name", "object"))) or "object"
    distance = max(0.0, float(observation.distance_m))
    try:
        bearing = float(observation.bearing_deg)
    except (AttributeError, TypeError, ValueError):
        bearing = 0.0
    direction = _direction_phrase(bearing, cone_deg)
    return (
        "You have reached your goal destination. "
        f"The {label} is {distance:.1f} meters {direction}."
    )


class GoalTracker:
    """Active goal plus the hysteresis needed for a one-shot arrival sentence."""

    def __init__(
        self,
        spec: GoalSpec,
        arrival_distance_m: float = DEFAULT_GOAL_ARRIVAL_M,
        stable_frames: int = DEFAULT_ARRIVAL_STABLE_FRAMES,
        cone_deg: float = DEFAULT_ARRIVAL_CONE_DEG,
    ) -> None:
        if float(arrival_distance_m) <= 0.0:
            raise ValueError("arrival_distance_m must be positive")
        if int(stable_frames) < 1:
            raise ValueError("stable_frames must be at least 1")
        self.spec = spec
        self.arrival_distance_m = float(arrival_distance_m)  # TO TUNE ON REAL DATA
        self.stable_frames = int(stable_frames)  # TO TUNE ON REAL DATA
        self.cone_deg = float(cone_deg)  # TO TUNE ON REAL DATA
        self._streak_track: int | None = None
        self._streak = 0
        self._arrived = False
        self.arrival_text: str | None = None

    @property
    def active(self) -> bool:
        return not self._arrived

    def select(self, observations: Sequence[ObjectAnchor]) -> ObjectAnchor | None:
        return select_goal_candidate(observations, self.spec.candidates)

    def update(self, observation: ObjectAnchor | None) -> str | None:
        """Feed one frame's nearest goal candidate; return arrival text once."""
        if self._arrived:
            return None
        if observation is None or not _usable(observation):
            self._streak_track = None
            self._streak = 0
            return None

        if float(observation.distance_m) > self.arrival_distance_m:
            self._streak_track = None
            self._streak = 0
            return None

        track_id = int(getattr(observation, "track_id", -1))
        if track_id == self._streak_track:
            self._streak += 1
        else:
            self._streak_track = track_id
            self._streak = 1
        if self._streak < self.stable_frames:
            return None

        self._arrived = True
        self.arrival_text = arrival_message(observation, self.cone_deg)
        return self.arrival_text


def build_goal_tracker(
    goal_text: str | None = None,
    goal_class: str | None = None,
    arrival_distance_m: float = DEFAULT_GOAL_ARRIVAL_M,
    stable_frames: int = DEFAULT_ARRIVAL_STABLE_FRAMES,
) -> GoalTracker | None:
    """Build a tracker from ``--goal`` text or a legacy ``--goal-class``."""
    text = "" if goal_text is None else str(goal_text).strip()
    single = "" if goal_class is None else str(goal_class).strip()
    if text and single:
        raise ValueError("pass goal text or goal class, not both")
    if text:
        spec = parse_goal_text(text)
    elif single:
        spec = GoalSpec(intent=single, candidates=(single,), text=single)
    else:
        return None
    return GoalTracker(
        spec,
        arrival_distance_m=arrival_distance_m,
        stable_frames=stable_frames,
    )


__all__ = [
    "DEFAULT_ARRIVAL_CONE_DEG",
    "DEFAULT_ARRIVAL_STABLE_FRAMES",
    "DEFAULT_GOAL_ARRIVAL_M",
    "GOAL_ALIASES",
    "GOAL_MAP",
    "GoalSpec",
    "GoalTracker",
    "UNKNOWN_DISTANCE_M",
    "arrival_message",
    "build_goal_tracker",
    "canonical_class",
    "is_cancel_goal",
    "parse_goal_text",
    "select_goal_candidate",
]
