"""P2: class-map construction from model names, safety and override behaviour."""

from __future__ import annotations

import pytest

from core.config import MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER
from perception.class_map import DEFAULT_RULES, build_class_map

ID2LABEL = {
    0: "wall",
    1: "floor",
    2: "road",
    3: "person",
    4: "sky",
    5: "windowpane",
    6: "banana",
}

# Hand-derived from the documented default rules:
# wall->wall, floor->floor, road->floor, person->obstacle,
# sky->other (sky mask is separate), windowpane->obstacle, banana->other.
EXPECTED = {
    0: MASK_WALL,
    1: MASK_FLOOR,
    2: MASK_FLOOR,
    3: MASK_OBSTACLE,
    4: MASK_OTHER,
    5: MASK_OBSTACLE,
    6: MASK_OTHER,
}


def test_p2_default_rules_map_exact():
    assert build_class_map(ID2LABEL) == EXPECTED


def test_p2_unknown_names_never_map_to_floor():
    unknown = {0: "banana", 1: "spaceship", 2: "unknown thing", 3: "sky"}
    mapping = build_class_map(unknown)
    assert mapping == {0: MASK_OTHER, 1: MASK_OTHER, 2: MASK_OTHER, 3: MASK_OTHER}
    assert all(value != MASK_FLOOR for value in mapping.values())


def test_p2_every_id_maps_exactly_once():
    mapping = build_class_map(ID2LABEL)
    assert set(mapping) == set(ID2LABEL)
    assert len(mapping) == len(ID2LABEL)
    assert all(value in (MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER) for value in mapping.values())


def test_p2_changing_a_rule_changes_the_map():
    rules = {MASK_OBSTACLE: ("banana",), MASK_FLOOR: ("floor",)}
    mapping = build_class_map({0: "banana", 1: "floor", 2: "road"}, rules)
    assert mapping[0] == MASK_OBSTACLE
    assert mapping[1] == MASK_FLOOR
    assert mapping[2] == MASK_OTHER  # road is not in the replacement rules


def test_p2_multiname_label_and_conflicting_rules():
    mapping = build_class_map({0: "windowpane; window"}, DEFAULT_RULES)
    assert mapping[0] == MASK_OBSTACLE

    with pytest.raises(ValueError, match="claimed by both"):
        build_class_map({0: "floor"}, {MASK_FLOOR: ("floor",), MASK_OBSTACLE: ("floor",)})
