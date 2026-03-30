"""
Global scene transform.

All object world vertices are multiplied by this matrix before camera
projection, so pan/zoom/rotation affect every object uniformly.

Transform order: T(pan) · Ry · Rx · T(pivot_in) · S(zoom) · T(pivot_out)

The rotation pivots around SCENE_CENTER_Z on the Z axis so objects spin
in place rather than swinging off screen.
"""

from __future__ import annotations

import numpy as np

from utils.math3d import (
    translation, rotation_x, rotation_y, rotation_z,
    scale_matrix, identity,
)
from config import SCENE_CENTER_Z, SCENE_ZOOM_MIN, SCENE_ZOOM_MAX


class SceneController:

    def __init__(self) -> None:
        self.pan:      np.ndarray = np.zeros(2, dtype=float)   # world-space X, Y
        self.zoom:     float      = 1.0
        self.rotation: np.ndarray = np.zeros(3, dtype=float)   # rx, ry, rz (radians)

    # ------------------------------------------------------------------

    def get_matrix(self) -> np.ndarray:
        """Return the 4×4 scene-space transform matrix."""
        # Move objects so the pivot (z=SCENE_CENTER_Z) is at origin, rotate, move back
        T_in   = translation(0.0, 0.0, -SCENE_CENTER_Z)
        Rx     = rotation_x(self.rotation[0])
        Ry     = rotation_y(self.rotation[1])
        Rz     = rotation_z(self.rotation[2])
        T_out  = translation(0.0, 0.0,  SCENE_CENTER_Z)
        S      = scale_matrix(self.zoom, self.zoom, self.zoom)
        T_pan  = translation(self.pan[0], self.pan[1], 0.0)

        return T_pan @ T_out @ Rz @ Ry @ Rx @ T_in @ S

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.pan[:]    = 0.0
        self.zoom      = 1.0
        self.rotation[:] = 0.0
