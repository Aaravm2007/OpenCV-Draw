"""
2D/3D geometric helper functions.
Pure functions — no state, no side effects.
"""

import math
from typing import Tuple
import numpy as np


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def distance_3d(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3-D points."""
    return float(np.linalg.norm(np.array(p2, dtype=float) - np.array(p1, dtype=float)))


def midpoint_2d(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[float, float]:
    """Midpoint between two 2-D points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def midpoint_3d(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Midpoint between two 3-D points."""
    return (
        (p1[0] + p2[0]) / 2.0,
        (p1[1] + p2[1]) / 2.0,
        (p1[2] + p2[2]) / 2.0,
    )


def angle_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Signed angle in radians from p1 to p2 (range: -π … π)."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns zero vector if input is zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return np.zeros_like(v)
    return v / norm


def angle_diff(a: float, b: float) -> float:
    """Shortest signed difference between two angles in radians (range -π…π).

    Returns the minimal rotation from angle b to angle a.
    Both inputs may be any value; the result is normalised to (-π, π].
    """
    d = a - b
    while d >  math.pi: d -= 2.0 * math.pi
    while d < -math.pi: d += 2.0 * math.pi
    return d
