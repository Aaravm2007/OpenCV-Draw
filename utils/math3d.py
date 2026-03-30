"""
3-D linear-algebra helpers for the pseudo-3D rendering pipeline.
All functions return new arrays; they never mutate their inputs.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Rotation matrices (right-hand rule, angles in radians)
# ---------------------------------------------------------------------------

def rotation_x(angle: float) -> np.ndarray:
    """4×4 homogeneous rotation matrix around the X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1,  0,  0, 0],
        [0,  c, -s, 0],
        [0,  s,  c, 0],
        [0,  0,  0, 1],
    ], dtype=float)


def rotation_y(angle: float) -> np.ndarray:
    """4×4 homogeneous rotation matrix around the Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=float)


def rotation_z(angle: float) -> np.ndarray:
    """4×4 homogeneous rotation matrix around the Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=float)


def translation(tx: float, ty: float, tz: float) -> np.ndarray:
    """4×4 homogeneous translation matrix."""
    m = np.eye(4, dtype=float)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    """4×4 homogeneous scale matrix."""
    return np.diag([sx, sy, sz, 1.0]).astype(float)


def identity() -> np.ndarray:
    """4×4 identity matrix."""
    return np.eye(4, dtype=float)


# ---------------------------------------------------------------------------
# Perspective projection
# ---------------------------------------------------------------------------

def perspective_project(
    point_3d: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> Tuple[int, int]:
    """
    Project a 3-D point to integer 2-D screen coordinates.

    Args:
        point_3d:     (x, y, z) in camera / world space.
        focal_length: Focal length in pixels.
        cx, cy:       Principal point (typically frame centre).

    Returns:
        (px, py) integer screen coordinates.
    """
    x, y, z = float(point_3d[0]), float(point_3d[1]), float(point_3d[2])
    z = z if abs(z) > 1e-4 else 1e-4
    px = focal_length * x / z + cx
    py = focal_length * y / z + cy
    return (int(round(px)), int(round(py)))


def apply_transform(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 homogeneous transform to an (N, 3) array of vertices.

    Returns an (N, 3) array of transformed 3-D points.
    """
    n = vertices.shape[0]
    homogeneous = np.ones((n, 4), dtype=float)
    homogeneous[:, :3] = vertices
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]
