"""Visiona3D ground-plane and spatial geometry module."""

from core.config import Intrinsics, MountConfig, RansacConfig, ScaleConfig
from geometry.camera import backproject, project, ray_directions
from geometry.frames import camera_to_ground, ground_frame_basis, ground_to_camera
from geometry.ground_plane import FitStatus, PlaneFit, fit_ground_plane
from geometry.lift import Anchor, AnchorStatus, LiftResult, lift_frame, object_anchor
from geometry.scale import ScaleEstimator, ScaleResult

__all__ = [
    # config — canonical home is core.config
    "Intrinsics",
    "MountConfig",
    "RansacConfig",
    "ScaleConfig",
    # camera
    "project",
    "backproject",
    "ray_directions",
    # ground plane
    "FitStatus",
    "PlaneFit",
    "fit_ground_plane",
    # frames
    "camera_to_ground",
    "ground_to_camera",
    "ground_frame_basis",
    # scale
    "ScaleResult",
    "ScaleEstimator",
    # lift
    "LiftResult",
    "AnchorStatus",
    "Anchor",
    "lift_frame",
    "object_anchor",
]
