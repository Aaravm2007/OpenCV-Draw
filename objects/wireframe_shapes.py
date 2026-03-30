"""
Wireframe primitive factory functions.

Each function returns a DrawableObject with vertices and edges
defined in local space centred at the origin.

Shapes
------
Cube      — 8 vertices, 12 edges
Plane     — 4 vertices, 5 edges (rectangle + one diagonal for visual clarity)
Prism     — 6 vertices, 9 edges (triangular prism)
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from objects.drawable_object import DrawableObject
from objects.transform import Transform


# ---------------------------------------------------------------------------
# Cube
# ---------------------------------------------------------------------------

# Vertex layout (half-size = 0.5):
#
#    7 ---- 6
#   /|     /|
#  4 ---- 5 |
#  | 3 -- | 2
#  |/     |/
#  0 ---- 1
#
_CUBE_VERTS = np.array([
    [-0.5, -0.5, -0.5],  # 0
    [ 0.5, -0.5, -0.5],  # 1
    [ 0.5, -0.5,  0.5],  # 2
    [-0.5, -0.5,  0.5],  # 3
    [-0.5,  0.5, -0.5],  # 4
    [ 0.5,  0.5, -0.5],  # 5
    [ 0.5,  0.5,  0.5],  # 6
    [-0.5,  0.5,  0.5],  # 7
], dtype=float)

_CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical pillars
]


def create_cube(
    color: Tuple[int, int, int] = (200, 200, 200),
    pos: Tuple[float, float, float] = (0.0, 0.0, 5.0),
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scl: float = 1.0,
) -> DrawableObject:
    return DrawableObject(
        vertices=_CUBE_VERTS.copy(),
        edges=_CUBE_EDGES,
        transform=Transform.make(pos=pos, rot=rot, scl=(scl, scl, scl)),
        color=color,
        object_type="cube",
    )


# ---------------------------------------------------------------------------
# Plane (rectangle lying flat on XZ)
# ---------------------------------------------------------------------------

_PLANE_VERTS = np.array([
    [-0.5, 0.0, -0.5],  # 0
    [ 0.5, 0.0, -0.5],  # 1
    [ 0.5, 0.0,  0.5],  # 2
    [-0.5, 0.0,  0.5],  # 3
], dtype=float)

_PLANE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # border
    (0, 2),                             # diagonal
]


def create_plane(
    color: Tuple[int, int, int] = (100, 180, 100),
    pos: Tuple[float, float, float] = (0.0, -1.0, 5.0),
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scl: Tuple[float, float, float] = (4.0, 1.0, 3.0),
) -> DrawableObject:
    return DrawableObject(
        vertices=_PLANE_VERTS.copy(),
        edges=_PLANE_EDGES,
        transform=Transform.make(pos=pos, rot=rot, scl=scl),
        color=color,
        object_type="plane",
    )


# ---------------------------------------------------------------------------
# Triangular prism
# ---------------------------------------------------------------------------

# Front triangle: verts 0,1,2  (z = -0.5)
# Back  triangle: verts 3,4,5  (z =  0.5)
#   2       5
#  / \     / \
# 0 - 1   3 - 4

_PRISM_VERTS = np.array([
    [-0.5, -0.5, -0.5],  # 0
    [ 0.5, -0.5, -0.5],  # 1
    [ 0.0,  0.5, -0.5],  # 2
    [-0.5, -0.5,  0.5],  # 3
    [ 0.5, -0.5,  0.5],  # 4
    [ 0.0,  0.5,  0.5],  # 5
], dtype=float)

_PRISM_EDGES = [
    (0, 1), (1, 2), (2, 0),   # front triangle
    (3, 4), (4, 5), (5, 3),   # back triangle
    (0, 3), (1, 4), (2, 5),   # longitudinal edges
]


def create_prism(
    color: Tuple[int, int, int] = (180, 120, 60),
    pos: Tuple[float, float, float] = (0.0, 0.0, 5.0),
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scl: float = 1.0,
) -> DrawableObject:
    return DrawableObject(
        vertices=_PRISM_VERTS.copy(),
        edges=_PRISM_EDGES,
        transform=Transform.make(pos=pos, rot=rot, scl=(scl, scl, scl)),
        color=color,
        object_type="prism",
    )
