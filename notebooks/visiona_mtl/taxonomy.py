"""Single source of truth for the classes, label encoding and source-label mappings.

Semantic PNG encoding used everywhere in this project:

    0..7  environment classes (floor, walkable_path, road, wall, glass, staircase, door, sky)
    254   OTHER_LABELED: the pixel IS labeled, and it is none of the 8 classes (grass, building, a chair, ...).
          It is a NEGATIVE for every class the image supervises.
    255   IGNORE: unknown / unlabeled. It contributes nothing to any loss.

Why 254 exists: with only ids 0..7 and 255, every labeled-but-unmapped pixel would have to become 255, and the
model would never see "this grass is NOT floor". 254 supplies that negative without adding a ninth output.
"""
from __future__ import annotations

SEM_CLASSES = ["floor", "walkable_path", "road", "wall", "glass", "staircase", "door", "sky"]
SEM_ID = {name: i for i, name in enumerate(SEM_CLASSES)}
N_SEM = len(SEM_CLASSES)
OTHER_LABELED = 254
IGNORE = 255

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
assert len(COCO80) == 80

# ---- COCO-Stuff (names verified against labels.txt, 2026-11-23; ids are 1-based, pixel value = id - 1) ----
COCOSTUFF_TO_SEM = {
    "floor-marble": "floor", "floor-other": "floor", "floor-stone": "floor", "floor-tile": "floor",
    "floor-wood": "floor", "carpet": "floor", "rug": "floor",
    "pavement": "walkable_path",
    "road": "road",
    "wall-brick": "wall", "wall-concrete": "wall", "wall-other": "wall", "wall-panel": "wall",
    "wall-stone": "wall", "wall-tile": "wall", "wall-wood": "wall",
    "window-other": "glass",                       # window-blind is deliberately NOT mapped
    "stairs": "staircase",
    "door-stuff": "door",
    "sky-other": "sky",                            # confirmed: id 157
}

# ---- ADE20K SceneParsing: 150 names in training-id order (source label id k = index + 1; 0 = unlabeled) ----
ADE20K_NAMES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet", "sidewalk",
    "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa",
    "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink",
    "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow",
    "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book",
    "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar",
    "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator",
    "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyor belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven",
    "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier",
    "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]
assert len(ADE20K_NAMES) == 150
ADE20K_TO_SEM = {
    "wall": "wall", "floor": "floor", "rug": "floor",
    "sidewalk": "walkable_path", "path": "walkable_path",
    "road": "road",
    "windowpane": "glass",                         # NOT the ADE class named 'glass' (a drinking glass)
    "stairs": "staircase", "stairway": "staircase", "step": "staircase", "escalator": "staircase",
    "door": "door", "screen door": "door",
    "sky": "sky",
}

# ---- Hugging Face segments/sidewalk-semantic: 35 classes, id order from the dataset card (0 = unlabeled) ----
HF_SIDEWALK_NAMES = [
    "unlabeled", "flat-road", "flat-sidewalk", "flat-crosswalk", "flat-cyclinglane", "flat-parkingdriveway",
    "flat-railtrack", "flat-curb", "human-person", "human-rider", "vehicle-car", "vehicle-truck", "vehicle-bus",
    "vehicle-tramtrain", "vehicle-motorcycle", "vehicle-bicycle", "vehicle-caravan", "vehicle-cartrailer",
    "construction-building", "construction-door", "construction-wall", "construction-fenceguardrail",
    "construction-bridge", "construction-tunnel", "construction-stairs", "object-pole", "object-trafficsign",
    "object-trafficlight", "nature-vegetation", "nature-terrain", "sky", "void-ground", "void-dynamic",
    "void-static", "void-unclear",
]
assert len(HF_SIDEWALK_NAMES) == 35
HF_SIDEWALK_TO_SEM = {                              # unambiguous classes only
    "flat-sidewalk": "walkable_path", "flat-road": "road", "construction-door": "door",
    "construction-wall": "wall", "construction-stairs": "staircase", "sky": "sky",
}
HF_SIDEWALK_REVIEW = {                              # judgment calls, NOT mapped: decide before use
    "flat-crosswalk": "walkable but crosses a road: walkable_path or a hazard?",
    "flat-cyclinglane": "flat, shared with bicycles: road or walkable_path?",
    "flat-parkingdriveway": "flat, vehicle area: road?",
    "flat-railtrack": "not walkable: road or other?",
    "flat-curb": "belongs to the later class `curb`",
}
HF_SIDEWALK_IGNORE = ("unlabeled", "void-ground", "void-dynamic", "void-static", "void-unclear")

# ---- per-dataset supervision: which classes an image can say both "yes" and "no" about ----
ALL_CLASSES = list(range(N_SEM))
SUPERVISED = {
    "coco_stuff": ALL_CLASSES, "ade20k": ALL_CLASSES, "hf_sidewalk": ALL_CLASSES,
    "gdd_train": [SEM_ID["glass"]], "gdd_test": [SEM_ID["glass"]],
    "stairs_kaist": [SEM_ID["staircase"]],
}
# Mapping to the four labels the existing Visiona3D safety layer uses.
APP_LABEL = {
    "floor": "floor", "walkable_path": "floor", "wall": "wall", "glass": "wall",
    "staircase": "obstacle", "door": "obstacle", "road": "obstacle", "sky": "sky",
}
