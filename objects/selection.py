"""
Object selection via pinch.

find_closest_object() projects every object's world-space centre through the
current scene matrix + camera and returns the one whose screen position is
within OBJ_SELECT_THRESHOLD_PX of the given screen point, or None.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from utils.geometry import distance_2d
from utils.math3d import apply_transform
from config import OBJ_SELECT_THRESHOLD_PX


def find_closest_object(
    objects: list,                        # List[DrawableObject]
    screen_point: Tuple[float, float],
    camera,                               # PerspectiveCamera
    scene_matrix: Optional[np.ndarray],
):
    """
    Return the DrawableObject whose projected centre is closest to screen_point
    and within OBJ_SELECT_THRESHOLD_PX, or None if nothing is close enough.
    """
    best_obj  = None
    best_dist = float(OBJ_SELECT_THRESHOLD_PX)

    for obj in objects:
        centre_w = obj.transform.position.reshape(1, 3)   # (1, 3)

        if scene_matrix is not None:
            centre_w = apply_transform(centre_w, scene_matrix)

        projected = camera.project(centre_w)
        if projected[0] is None:
            continue

        px, py, _ = projected[0]
        dist = distance_2d(screen_point, (px, py))

        if dist < best_dist:
            best_dist = dist
            best_obj  = obj

    return best_obj
