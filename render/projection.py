"""
Perspective projection.

PerspectiveCamera converts world-space (Nx3) vertices to screen-space
(px, py, depth) tuples using the standard pinhole model.

Coordinate conventions
----------------------
World space : right-handed, Y-up, Z toward viewer (camera looks along +Z).
Screen space: origin top-left, Y-down.

The Y-flip is applied inside project():  py = -f*y/z + cy
Vertices with z ≤ SCENE_Z_NEAR are marked as None (clipped).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from config import FOCAL_LENGTH, SCENE_Z_NEAR


# Screen point: (pixel_x, pixel_y, world_depth)  or  None if clipped
ScreenPoint = Optional[Tuple[int, int, float]]


class PerspectiveCamera:

    def __init__(
        self,
        focal_length: float = FOCAL_LENGTH,
        cx: float = 640.0,
        cy: float = 360.0,
    ) -> None:
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy

    # ------------------------------------------------------------------

    def project(self, world_verts: np.ndarray) -> List[ScreenPoint]:
        """
        Project (N, 3) world vertices to screen.

        Returns a list of (px, py, depth) or None for clipped vertices.
        """
        result: List[ScreenPoint] = []
        f = self.focal_length

        for v in world_verts:
            x, y, z = float(v[0]), float(v[1]), float(v[2])
            if z <= SCENE_Z_NEAR:
                result.append(None)
                continue
            px = int(f * x / z + self.cx)
            py = int(-f * y / z + self.cy)   # Y flip: world-up → screen-up
            result.append((px, py, z))

        return result

    def update_centre(self, frame_w: int, frame_h: int) -> None:
        """Keep the principal point at the frame centre."""
        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0
