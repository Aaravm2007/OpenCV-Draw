"""
Object manager.

Owns all DrawableObjects in the scene.
Renders them each frame using a PerspectiveCamera.

Rendering features:
  - Depth-based edge dimming (far edges appear darker).
  - Selected objects draw in the highlight colour with thicker lines.
  - Vertices behind the camera (z ≤ SCENE_Z_NEAR) are silently clipped.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from objects.drawable_object import DrawableObject
from render.projection import PerspectiveCamera
from utils.math3d import apply_transform as _apply_transform
from config import (
    WIREFRAME_THICKNESS,
    WIREFRAME_SELECTED_COLOR,
    WIREFRAME_DEPTH_DIM,
    SCENE_Z_NEAR,
)


class ObjectManager:

    def __init__(self) -> None:
        self._objects:  List[DrawableObject]    = []
        self._selected: Optional[DrawableObject] = None

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def add(self, obj: DrawableObject) -> None:
        self._objects.append(obj)

    def remove(self, obj: DrawableObject) -> None:
        if obj in self._objects:
            self._objects.remove(obj)
        if self._selected is obj:
            self._selected = None

    def clear(self) -> None:
        self._objects.clear()
        self._selected = None

    @property
    def objects(self) -> List[DrawableObject]:
        return list(self._objects)

    @property
    def selected(self) -> Optional[DrawableObject]:
        return self._selected

    def select(self, obj: Optional[DrawableObject]) -> None:
        if self._selected is not None:
            self._selected.selected = False
        self._selected = obj
        if obj is not None:
            obj.selected = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        camera: PerspectiveCamera,
        scene_matrix=None,   # Optional[np.ndarray] — applied before projection
    ) -> None:
        """Draw all objects onto frame in-place."""
        unselected = [o for o in self._objects if not o.selected]
        selected   = [o for o in self._objects if o.selected]

        for obj in unselected + selected:
            self._render_object(frame, obj, camera, scene_matrix)

    # ------------------------------------------------------------------

    def _render_object(
        self,
        frame: np.ndarray,
        obj: DrawableObject,
        camera: PerspectiveCamera,
        scene_matrix=None,
    ) -> None:
        world_verts = obj.get_world_vertices()
        if scene_matrix is not None:
            world_verts = _apply_transform(world_verts, scene_matrix)
        screen_verts = camera.project(world_verts)

        h, w = frame.shape[:2]

        # Determine depth range for this object (for dimming)
        depths = [sv[2] for sv in screen_verts if sv is not None]
        if not depths:
            return
        z_min, z_max = min(depths), max(depths)
        z_range = max(z_max - z_min, 0.001)

        base_color    = WIREFRAME_SELECTED_COLOR if obj.selected else obj.color
        base_thick    = WIREFRAME_THICKNESS + (1 if obj.selected else 0)

        for i, j in obj.edges:
            p1 = screen_verts[i]
            p2 = screen_verts[j]
            if p1 is None or p2 is None:
                continue

            px1, py1, d1 = p1
            px2, py2, d2 = p2

            # Skip if both endpoints are completely off screen
            if not (_in_frame(px1, py1, w, h) or _in_frame(px2, py2, w, h)):
                continue

            # Depth dimming: farther edges are darker
            avg_z      = (d1 + d2) / 2.0
            t          = (avg_z - z_min) / z_range           # 0=near, 1=far
            brightness = 1.0 - WIREFRAME_DEPTH_DIM * t
            color = tuple(int(c * brightness) for c in base_color)

            cv2.line(frame, (px1, py1), (px2, py2),
                     color, base_thick, cv2.LINE_AA)

        # Draw a small dot at the projected object centre
        centre_world  = np.array([obj.transform.position])
        centre_screen = camera.project(centre_world)
        if centre_screen[0] is not None:
            cx, cy, _ = centre_screen[0]
            if _in_frame(cx, cy, w, h):
                dot_color = WIREFRAME_SELECTED_COLOR if obj.selected else base_color
                cv2.circle(frame, (cx, cy), 3, dot_color, -1)


def _in_frame(x: int, y: int, w: int, h: int) -> bool:
    margin = 200   # allow slightly off-screen endpoints (line may still cross frame)
    return -margin < x < w + margin and -margin < y < h + margin
