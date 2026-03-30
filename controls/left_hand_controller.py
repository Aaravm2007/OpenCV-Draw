"""
Left-hand scene controller — pan only.

Maps left-hand pinch drag to global scene pan (XY).
All deltas are relative to the state recorded at pinch-start so the
transform never drifts.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from controls.scene_controller import SceneController
from config import FOCAL_LENGTH, SCENE_CENTER_Z, SCENE_PAN_SENSITIVITY


class LeftHandController:

    def __init__(self, scene: SceneController) -> None:
        self._scene = scene
        self._active = False

        # Captured at pinch start
        self._start_mid: Optional[np.ndarray] = None
        self._start_pan: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def update(self, left_pinch) -> None:
        """Call every frame with the current left-hand PinchState (or None)."""
        if left_pinch is None or not left_pinch.is_pinching:
            self._active = False
            return

        if not self._active:
            # Pinch just started — snapshot current scene pan
            self._start_mid = np.array(left_pinch.midpoint, dtype=float)
            self._start_pan = self._scene.pan.copy()
            self._active    = True
            return

        # ── Ongoing pinch — pan only ──
        current_mid = np.array(left_pinch.midpoint, dtype=float)
        delta_px    = current_mid - self._start_mid

        pan_scale = SCENE_CENTER_Z / FOCAL_LENGTH * SCENE_PAN_SENSITIVITY
        self._scene.pan = self._start_pan + np.array(
            [delta_px[0] * pan_scale, -delta_px[1] * pan_scale]
        )
