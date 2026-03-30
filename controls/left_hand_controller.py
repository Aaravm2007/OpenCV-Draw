"""
Left-hand scene controller.

Maps left-hand pinch gestures to global scene transforms:
  Drag midpoint   → pan scene (XY)
  Distance change → zoom
  Angle change    → rotate scene around Z (twist gesture)

All deltas are relative to the state recorded at pinch-start so the
transforms never drift.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from controls.scene_controller import SceneController
from utils.geometry import angle_diff
from config import (
    FOCAL_LENGTH, SCENE_CENTER_Z,
    SCENE_PAN_SENSITIVITY, SCENE_ZOOM_MIN, SCENE_ZOOM_MAX,
)


class LeftHandController:

    def __init__(self, scene: SceneController) -> None:
        self._scene = scene
        self._active = False

        # Captured at pinch start
        self._start_mid:   Optional[np.ndarray] = None
        self._start_dist:  float = 1.0
        self._start_angle: float = 0.0
        self._start_pan:   Optional[np.ndarray] = None
        self._start_zoom:  float = 1.0
        self._start_rot_z: float = 0.0

    # ------------------------------------------------------------------

    def update(self, left_pinch) -> None:
        """Call every frame with the current left-hand PinchState (or None)."""
        if left_pinch is None or not left_pinch.is_pinching:
            self._active = False
            return

        if not self._active:
            # Pinch just started — snapshot current scene state
            self._start_mid   = np.array(left_pinch.midpoint, dtype=float)
            self._start_dist  = max(left_pinch.distance_px, 1.0)
            self._start_angle = left_pinch.angle
            self._start_pan   = self._scene.pan.copy()
            self._start_zoom  = self._scene.zoom
            self._start_rot_z = self._scene.rotation[2]
            self._active      = True
            return

        # ── Ongoing pinch ──

        current_mid   = np.array(left_pinch.midpoint, dtype=float)
        delta_px      = current_mid - self._start_mid
        dist_ratio    = left_pinch.distance_px / self._start_dist
        delta_angle   = angle_diff(left_pinch.angle, self._start_angle)

        # Pan: screen delta → world delta at scene-centre depth
        pan_scale = SCENE_CENTER_Z / FOCAL_LENGTH * SCENE_PAN_SENSITIVITY
        self._scene.pan = self._start_pan + np.array(
            [delta_px[0] * pan_scale, -delta_px[1] * pan_scale]
        )

        # Zoom: multiplicative
        new_zoom = self._start_zoom * dist_ratio
        self._scene.zoom = float(np.clip(new_zoom, SCENE_ZOOM_MIN, SCENE_ZOOM_MAX))

        # Rotation Z: twist gesture
        self._scene.rotation[2] = self._start_rot_z + delta_angle


