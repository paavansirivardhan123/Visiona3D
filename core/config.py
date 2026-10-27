from dataclasses import dataclass
import math
import numpy as np


class Config:
    """Centralized configuration."""

    SOURCES = {
        "FRONT": "sample-vid/3Dir/vid1/FRONT.mp4",
        "LEFT":  "sample-vid/3Dir/vid1/LEFT.mp4",
        "RIGHT": "sample-vid/3Dir/vid1/RIGHT.mp4",
        "BACK":  None,
    }

    MODEL_PATH     = "yolo11n.pt"
    IMG_SIZE       = 640
    CONF_THRESHOLD = 0.35
    FRAME_SKIP     = 3  # Process every 3rd frame for better sync (was 2)

    METERS_PER_STEP = 0.75
    MAX_DISTANCE_M  = 10.0 * METERS_PER_STEP
    CONSIDER_MAX_M  = 7.0 * METERS_PER_STEP
    HIGH_PRIORITY_M = 3.0 * METERS_PER_STEP

    FOCAL_LENGTH_PX     = 700
    AVG_OBJECT_WIDTH_CM = 30
    KNOWN_WIDTHS_CM = {
        "person": 50, "car": 180, "truck": 250, "bus": 250,
        "bicycle": 60, "motorcycle": 80, "dog": 40, "cat": 25,
        "bottle": 8, "chair": 50, "cup": 8, "laptop": 35,
        "backpack": 35, "book": 20, "dining table": 80,
    }

    from dotenv import load_dotenv
    load_dotenv()

    GROUP_THRESHOLD = 5

    OBJECT_PRIORITY = {
        "person": 10, "child": 10,
        "car": 9, "truck": 9, "bus": 9, "motorcycle": 8, "bicycle": 7,
        "dog": 6, "cat": 5,
        "chair": 3, "dining table": 3, "bottle": 2, "cup": 2,
    }
    DEFAULT_PRIORITY = 1
    
    THREAT_HIGH_THRESHOLD = 70.0
    PENALTY_APPLY         = 30.0
    PENALTY_DECAY         = 3.0

    BEEP_LEVELS = [
        (0.5,  2000, 80),
        (1.0,  1500, 100),
        (1.5,  1000, 120),
    ]

    SPEECH_RATE       = 185
    SPEECH_COOLDOWN   = 2.5
    SEMANTIC_COOLDOWN = 5.0
    MAX_MESSAGES      = 3
    
    # New cooldown configurations
    FRONT_CAMERA_COOLDOWN = 3.0      # seconds for FRONT camera non-threat objects
    GOAL_SAME_PANEL_COOLDOWN = 5.0   # seconds for goal objects in same direction with decreasing distance (was 2.0)
    GOAL_CONSTANT_COOLDOWN = 7.0     # seconds for goal objects with constant distance (was 5.0)

    WINDOW_NAME = "Visiona AI"
    DISPLAY_W   = 640
    DISPLAY_H   = 480
    HUD_H       = 90

    LOG_ENABLED = True
    LOG_DIR     = "logs"

    LLM_MODEL       = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0
    LLM_COOLDOWN    = 5.0

    DEPTH_MODEL_ID      = "depth-anything/Depth-Anything-V2-Small-hf"
    MIDAS_DEFAULT_SCALE = 1.0
    DEPTH_SMOOTH_FRAMES = 5

    DEPTH_SPIKE_THRESHOLD = 1.5
    KF_PROCESS_NOISE      = 0.01
    KF_MEASUREMENT_NOISE  = 0.1

    TRACK_HISTORY_LEN = 30
    TRACK_MAX_AGE     = 10
    TRACKER_BACKEND   = "bytetrack"

    SPEED_MIN_THRESHOLD  = 0.2
    LATERAL_THRESHOLD_PX = 20
    SPEED_SMOOTH_FRAMES  = 5

    TTC_SMOOTH_ALPHA   = 0.3
    TTC_WARN_THRESHOLD = 3.0

    FRAME_BUDGET_MS = 2000


# ---------------------------------------------------------------------------
# Geometry module configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intrinsics:
    """Camera intrinsic parameters for pinhole camera model.

    Attributes:
        fx: Focal length along x axis in pixels.
        fy: Focal length along y axis in pixels.
        cx: Principal point x coordinate in pixels.
        cy: Principal point y coordinate in pixels.
        width: Image width in pixels. Default: 1280.
        height: Image height in pixels. Default: 720.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1280
    height: int = 720

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        """Construct Intrinsics from image dimensions and horizontal FOV."""
        hfov_rad = math.radians(hfov_deg)
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    def as_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class MountConfig:
    """Camera mounting configuration relative to ground.

    Attributes:
        h_known: True mounted camera height above ground in metres. Default: 1.4 m.
        pitch_deg: Nominal camera pitch angle in degrees (positive = tilted down). Default: 15.0 deg.
    """

    h_known: float = 1.4
    pitch_deg: float = 15.0


@dataclass(frozen=True)
class RansacConfig:
    """RANSAC plane fitting hyperparameters.

    Attributes:
        max_iterations: Number of 3-point sample iterations. Default: 500.
        tau0: Base inlier distance threshold in metres. Default: 0.02 m.
        tau1: Depth-dependent inlier distance multiplier. Default: 0.01.
        max_points: Maximum points subsampled for RANSAC candidate evaluation. Default: 5000.
        min_points: Minimum points required to attempt plane fit. Default: 500.
        min_inlier_ratio: Minimum fraction of inliers required for valid fit. Default: 0.5.
        max_pitch_deg: Maximum plausible pitch angle in degrees. Default: 60.0 deg.
    """

    max_iterations: int = 500
    tau0: float = 0.02
    tau1: float = 0.01
    max_points: int = 5000
    min_points: int = 500
    min_inlier_ratio: float = 0.5
    max_pitch_deg: float = 60.0


@dataclass(frozen=True)
class ScaleConfig:
    """Scale estimator filtering hyperparameters.

    Attributes:
        alpha: Exponential moving average smoothing factor. Default: 0.2.
        max_step_change: Maximum allowed relative change in a single frame. Default: 0.15.
        outlier_threshold: Relative deviation threshold to flag an outlier. Default: 0.25.
        outlier_patience: Consecutive outlier frames to accept a persistent shift. Default: 5.
        initial_scale: Initial scale correction value. Default: 1.0.
    """

    alpha: float = 0.2
    max_step_change: float = 0.15
    outlier_threshold: float = 0.25
    outlier_patience: int = 5
    initial_scale: float = 1.0


# ---------------------------------------------------------------------------
# Dataset and Evaluation Conventions (Research Section)
# ---------------------------------------------------------------------------
#
# These constants define the ground-truth conventions, mask taxonomy, range bins,
# and tag vocabularies used by the dataset, annotation, and evaluation harness.
# They are defined here alongside Intrinsics and MountConfig to avoid config drift.
#


# Mask class indices (define once, reference everywhere)
MASK_UNLABELED = 0      # Ignored everywhere
MASK_FLOOR = 1          # Floor / ground plane
MASK_WALL = 2           # Vertical wall
MASK_OBSTACLE = 3       # Obstacle (object above ground)
MASK_OTHER = 4          # Other (ceiling, vegetation, etc.)
MASK_IGNORE = 255       # Ignore region (glass, mirror, ambiguous)

MASK_CLASS_NAMES = {
    MASK_UNLABELED: "unlabeled",
    MASK_FLOOR: "floor",
    MASK_WALL: "wall",
    MASK_OBSTACLE: "obstacle",
    MASK_OTHER: "other",
    MASK_IGNORE: "ignore",
}

MASK_EVAL_CLASSES = [MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER]


# Range bins for every per-range metric (left-inclusive, right-exclusive)
# Bins: [0, 2), [2, 4), [4, 6), [6, inf) metres
RANGE_BIN_EDGES = [0.0, 2.0, 4.0, 6.0, float("inf")]
RANGE_BIN_NAMES = ["0-2m", "2-4m", "4-6m", "6+m"]


# Tag vocabularies (define once)
ENV_TAGS = ["indoor", "outdoor"]
LIGHT_TAGS = ["day", "dusk", "dark"]
HAZARD_TAGS = ["glass", "mirror", "wet", "none"]
CROWD_TAGS = ["none", "low", "high"]

ALL_TAG_VOCABS = {
    "env": ENV_TAGS,
    "light": LIGHT_TAGS,
    "hazard": HAZARD_TAGS,
    "crowd": CROWD_TAGS,
}


def range_bin_index(depth_m: float) -> int:
    """Return the range bin index for a given depth in metres.

    Bins: [0,2), [2,4), [4,6), [6,inf) — left-inclusive, right-exclusive.
    Returns -1 for invalid (NaN, negative) depths.
    """
    if not np.isfinite(depth_m) or depth_m < 0.0:
        return -1
    for i, edge in enumerate(RANGE_BIN_EDGES[:-1]):
        if depth_m < RANGE_BIN_EDGES[i + 1]:
            return i
    return len(RANGE_BIN_EDGES) - 2  # Last bin for >= 6m


def validate_tag(tag_name: str, tag_value: str) -> bool:
    """Validate a tag value against the defined vocabulary."""
    vocab = ALL_TAG_VOCABS.get(tag_name)
    if vocab is None:
        return False
    return tag_value in vocab
