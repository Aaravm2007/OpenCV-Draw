"""
Thumb–index pinch detection.

Normalises the pinch distance against hand size so the threshold works
across different hand sizes and camera distances.  Hysteresis (separate
ON / OFF thresholds) prevents rapid flickering at the boundary.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math

from utils.geometry import distance_2d
from utils.constants import WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP
from config import PINCH_ON_THRESHOLD, PINCH_OFF_THRESHOLD


@dataclass
class PinchState:
    is_pinching: bool
    distance_px: float           # raw pixel distance between tips
    normalized_distance: float   # relative to hand size (0 = closed, ~1 = open)
    midpoint: Tuple[float, float]
    angle: float                 # radians, atan2 from thumb tip → index tip


def _hand_size(landmarks: List[Tuple[float, float, float]]) -> float:
    wrist   = landmarks[WRIST][:2]
    mid_mcp = landmarks[MIDDLE_MCP][:2]
    return max(distance_2d(wrist, mid_mcp), 1.0)


def detect_pinch(
    landmarks: List[Tuple[float, float, float]],
    prev_pinching: bool,
) -> PinchState:
    thumb = landmarks[THUMB_TIP][:2]
    index = landmarks[INDEX_TIP][:2]

    dist_px   = distance_2d(thumb, index)
    hand_size = _hand_size(landmarks)
    norm_dist = dist_px / hand_size

    threshold   = PINCH_OFF_THRESHOLD if prev_pinching else PINCH_ON_THRESHOLD
    is_pinching = norm_dist < threshold

    midpoint = ((thumb[0] + index[0]) / 2.0, (thumb[1] + index[1]) / 2.0)
    angle    = math.atan2(index[1] - thumb[1], index[0] - thumb[0])

    return PinchState(
        is_pinching=is_pinching,
        distance_px=dist_px,
        normalized_distance=norm_dist,
        midpoint=midpoint,
        angle=angle,
    )
