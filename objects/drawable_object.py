"""
DrawableObject — base class for all pseudo-3D scene objects.

Each object owns:
  - a set of vertices in local space (Nx3)
  - a list of edges as index pairs
  - a Transform (position / rotation / scale)
  - a display colour
  - a selected flag (used by Phase 6 interaction)
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from objects.transform import Transform
from utils.math3d import apply_transform


class DrawableObject:

    _next_id: int = 0

    def __init__(
        self,
        vertices: np.ndarray,            # (N, 3) local-space vertices
        edges: List[Tuple[int, int]],    # index pairs
        transform: Transform,
        color: Tuple[int, int, int],     # BGR
        object_type: str = "object",
    ) -> None:
        self.vertices    = np.array(vertices, dtype=float)
        self.edges       = list(edges)
        self.transform   = transform
        self.color       = color
        self.object_type = object_type
        self.selected    = False

        # Unique ID for debugging / selection lookup
        self.uid = DrawableObject._next_id
        DrawableObject._next_id += 1

    # ------------------------------------------------------------------

    def get_world_vertices(self) -> np.ndarray:
        """Return vertices transformed to world space by this object's matrix."""
        return apply_transform(self.vertices, self.transform.get_matrix())

    def get_center_world(self) -> np.ndarray:
        """World-space position of the object origin."""
        return self.transform.position.copy()

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DrawableObject(uid={self.uid}, type={self.object_type}, "
            f"pos={self.transform.position.round(2)})"
        )
