"""Metric scale estimation and temporal smoothing using ground plane height.

Follows Visiona3D geometry notation:
- h_known: true mounted camera height in metres.
- h_hat: camera height estimated from the plane fit.
- s = h_known / h_hat: metric scale correction factor applied to predicted depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from core.config import MountConfig, ScaleConfig
from geometry.ground_plane import FitStatus


@dataclass(frozen=True)
class ScaleResult:
    """Result of scale update.

    Attributes:
        s: Current smoothed metric scale correction factor.
        status: Status string ('OK', 'OUTLIER_IGNORED', or plane fit error status).
    """

    s: float
    status: str


class ScaleEstimator:
    """Estimates and temporally filters metric scale correction factor s.

    The first valid measurement initialises s directly from h_known / h_hat so
    that metric distances are correct from the first frame. EMA smoothing, the
    per-frame step clip and the persistence rule apply only after initialisation.
    """

    def __init__(
        self,
        mount: MountConfig,
        cfg: Optional[ScaleConfig] = None,
    ) -> None:
        self.mount = mount
        self.cfg = cfg if cfg is not None else ScaleConfig()
        self.s = float(self.cfg.initial_scale)
        self.consecutive_outliers = 0
        self._last_outlier_val: Optional[float] = None
        self._initialized = False

    def update(
        self,
        h_hat: float,
        status: Union[FitStatus, str] = FitStatus.OK,
    ) -> ScaleResult:
        """Update scale factor with new plane height estimate.

        Args:
            h_hat: Estimated camera height in metres.
            status: Status of the plane fit (FitStatus enum or string).

        Returns:
            ScaleResult containing filtered scale s and status.
        """
        status_str = status.value if isinstance(status, FitStatus) else str(status)

        # If fit failed or height is non-positive, keep previous s
        if status_str != FitStatus.OK.value or h_hat <= 0.0 or not np.isfinite(h_hat):
            return ScaleResult(s=self.s, status=status_str)

        s_raw = self.mount.h_known / h_hat
        if not np.isfinite(s_raw) or s_raw <= 0.0:
            return ScaleResult(s=self.s, status=FitStatus.IMPLAUSIBLE_GEOMETRY.value)

        # First valid measurement: initialise directly, bypassing EMA/clip/outlier logic.
        if not self._initialized:
            self.s = float(s_raw)
            self._initialized = True
            self.consecutive_outliers = 0
            self._last_outlier_val = None
            return ScaleResult(s=self.s, status="OK")

        # Check deviation from current EMA
        deviation = abs(s_raw - self.s) / self.s

        if deviation > self.cfg.outlier_threshold:
            # Check if this outlier persists in the same direction
            if self._last_outlier_val is not None:
                rel_diff = abs(s_raw - self._last_outlier_val) / self._last_outlier_val
                if rel_diff <= self.cfg.outlier_threshold:
                    self.consecutive_outliers += 1
                else:
                    self.consecutive_outliers = 1
            else:
                self.consecutive_outliers = 1

            self._last_outlier_val = s_raw

            if self.consecutive_outliers >= self.cfg.outlier_patience:
                # Persistent shift confirmed after patience frames: reset to candidate
                self.s = float(s_raw)
                self.consecutive_outliers = 0
                self._last_outlier_val = None
                return ScaleResult(s=self.s, status="OK")

            # Ignore single/transient outlier frame
            return ScaleResult(s=self.s, status="OUTLIER_IGNORED")

        # Inlier update
        self.consecutive_outliers = 0
        self._last_outlier_val = None

        # EMA smoothing
        s_ema = (1.0 - self.cfg.alpha) * self.s + self.cfg.alpha * s_raw

        # Clip single-frame change to +-max_step_change
        delta_max = self.cfg.max_step_change * self.s
        s_clipped = float(np.clip(s_ema, self.s - delta_max, self.s + delta_max))
        self.s = s_clipped

        return ScaleResult(s=self.s, status="OK")
