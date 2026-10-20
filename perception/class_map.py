"""Model-name-to-mask-class mapping for segmentation backends.

The map is built from a model's ``id2label`` NAMES, never from hand-typed
class indices. ADE20K (SegFormer) and other datasets order classes differently,
so looking up names is the only safe way to transfer labels.

Default rules (overridable via ``rules=``):

- floor: floor, road, sidewalk, path, rug, carpet, grass, earth, field, sand
- wall: wall, building, house, skyscraper, fence
- obstacle: person, car, chair, table, pole, tree, stairs, step, door,
  windowpane and similar named objects (see DEFAULT_RULES)
- sky: mapped to other here; backends report it separately through ``sky_mask``
- every other / unknown name: other

Safety rule: an unknown or unmapped class name can never map to floor.
A name claimed by more than one rule is an error, not a silent priority.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence
import numpy as np

from core.config import MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER


DEFAULT_RULES: dict[int, tuple[str, ...]] = {
    MASK_FLOOR: (
        "floor",
        "road",
        "sidewalk",
        "path",
        "rug",
        "carpet",
        "grass",
        "earth",
        "field",
        "sand",
    ),
    MASK_WALL: (
        "wall",
        "building",
        "house",
        "skyscraper",
        "fence",
    ),
    MASK_OBSTACLE: (
        "person",
        "pedestrian",
        "car",
        "truck",
        "bus",
        "van",
        "bicycle",
        "motorcycle",
        "minibike",
        "chair",
        "armchair",
        "seat",
        "stool",
        "bench",
        "ottoman",
        "table",
        "coffee table",
        "dining table",
        "desk",
        "counter",
        "countertop",
        "cabinet",
        "shelf",
        "bookcase",
        "wardrobe",
        "chest of drawers",
        "pole",
        "column",
        "pillar",
        "bannister",
        "railing",
        "tree",
        "palm",
        "plant",
        "stairs",
        "stairway",
        "step",
        "escalator",
        "door",
        "screen door",
        "windowpane",
        "window",
        "sofa",
        "cushion",
        "pillow",
        "box",
        "barrel",
        "basket",
        "bag",
        "bottle",
        "vase",
        "lamp",
        "streetlight",
        "traffic light",
        "sign",
        "signboard",
        "poster",
        "bulletin board",
        "awning",
        "canopy",
        "tower",
        "fireplace",
        "stove",
        "oven",
        "refrigerator",
        "microwave",
        "sink",
        "toilet",
        "bathtub",
        "shower",
        "radiator",
        "television receiver",
        "monitor",
        "screen",
        "computer",
        "machine",
        "fan",
        "clock",
        "sculpture",
        "fountain",
        "rock",
        "apparel",
        "towel",
        "blanket",
        "book",
        "tray",
        "plate",
        "pot",
        "food",
        "ashcan",
        "bar",
    ),
}

_SKY_NAMES = ("sky",)
_MULTI_NAME_SPLIT = re.compile(r"[;,]")


def normalize_class_name(name: str) -> str:
    """Normalise a dataset class name for matching."""
    text = str(name).strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def _names_in(name: str) -> list[str]:
    """Split multi-name labels such as ``windowpane; window``."""
    return [normalize_class_name(part) for part in _MULTI_NAME_SPLIT.split(str(name)) if part.strip()]


def _resolve_rules(rules: Mapping[int, Sequence[str]] | None) -> dict[int, tuple[str, ...]]:
    source = DEFAULT_RULES if rules is None else rules
    resolved: dict[int, tuple[str, ...]] = {}
    for mask_id, names in source.items():
        mask_id = int(mask_id)
        if mask_id not in (MASK_FLOOR, MASK_WALL, MASK_OBSTACLE):
            raise ValueError(
                f"rules may only target floor (1), wall (2) or obstacle (3); got {mask_id}"
            )
        resolved[mask_id] = tuple(normalize_class_name(n) for n in names)

    owner: dict[str, int] = {}
    for mask_id, names in resolved.items():
        for name in names:
            previous = owner.get(name)
            if previous is not None and previous != mask_id:
                raise ValueError(
                    f"class name {name!r} is claimed by both mask {previous} and mask {mask_id}"
                )
            owner[name] = mask_id
    return resolved


def build_class_map(
    id2label: Mapping[int, str] | Sequence[str],
    rules: Mapping[int, Sequence[str]] | None = None,
) -> dict[int, int]:
    """Build ``{source_class_id: mask_class_id}`` from model class names.

    Args:
        id2label: Mapping (or sequence indexed by id) of model class id to name.
        rules: Optional replacement for DEFAULT_RULES. Keys must be 1 (floor),
            2 (wall) or 3 (obstacle); names not matched map to other (4).

    Returns:
        Mapping covering every source id exactly once. Values are mask class
        indices from core.config (1=floor, 2=wall, 3=obstacle, 4=other).

    Raises:
        ValueError: If a name is claimed by multiple rules, or rules target an
            invalid mask class.
    """
    resolved = _resolve_rules(rules)

    name_to_mask: dict[str, int] = {}
    for mask_id, names in resolved.items():
        for name in names:
            name_to_mask[name] = mask_id

    items: list[tuple[int, str]]
    if isinstance(id2label, Mapping):
        items = [(int(k), str(v)) for k, v in id2label.items()]
    else:
        items = [(idx, str(name)) for idx, name in enumerate(id2label)]

    mapping: dict[int, int] = {}
    for class_id, name in items:
        targets: set[int] = set()
        for candidate in _names_in(name):
            if candidate in name_to_mask:
                targets.add(name_to_mask[candidate])
            elif candidate in _SKY_NAMES:
                targets.add(MASK_OTHER)
            else:
                # Unknown names are deliberately other, never floor.
                targets.add(MASK_OTHER)
        if len(targets) > 1:
            raise ValueError(
                f"class id {class_id} ({name!r}) matches multiple mask classes {sorted(targets)}"
            )
        mapping[class_id] = targets.pop()
    return mapping


def class_map_table(id2label: Mapping[int, str] | Sequence[str], rules: Mapping[int, Sequence[str]] | None = None) -> list[tuple[int, str, int]]:
    """Return ``[(source_id, name, mask_id), ...]`` for documentation/debugging."""
    mapping = build_class_map(id2label, rules)
    if isinstance(id2label, Mapping):
        names = {int(k): str(v) for k, v in id2label.items()}
    else:
        names = {idx: str(name) for idx, name in enumerate(id2label)}
    return [(class_id, names[class_id], mapping[class_id]) for class_id in sorted(mapping)]


__all__ = [
    "DEFAULT_RULES",
    "build_class_map",
    "normalize_class_name",
    "class_map_table",
]
