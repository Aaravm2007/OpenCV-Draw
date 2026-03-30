"""
Finger open/closed detection.

Uses landmark geometry: a finger is considered extended when its tip sits
above (lower y value) its PIP joint.  The thumb is a special case — it
moves laterally, so we compare x coordinates against the IP joint,
accounting for which hand it is (after horizontal flip).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from utils.constants import (
    THUMB_TIP, THUMB_IP,
    INDEX_TIP, INDEX_PIP,
    MIDDLE_TIP, MIDDLE_PIP,
    RING_TIP, RING_PIP,
    PINKY_TIP, PINKY_PIP,
)


@dataclass
class FingerState:
    thumb:  bool
    index:  bool
    middle: bool
    ring:   bool
    pinky:  bool

    @property
    def extended_count(self) -> int:
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])

    def as_tuple(self) -> Tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def __str__(self) -> str:
        names = []
        if self.thumb:  names.append("T")
        if self.index:  names.append("I")
        if self.middle: names.append("M")
        if self.ring:   names.append("R")
        if self.pinky:  names.append("P")
        return "".join(names) if names else "—"


def detect_finger_state(
    landmarks: List[Tuple[float, float, float]],
    handedness: str,
) -> FingerState:
    """
    Detect which fingers are extended from smoothed pixel-space landmarks.

    Args:
        landmarks:  21-point list of (x, y, z) in pixel coordinates.
        handedness: 'Left' or 'Right' (as reported by MediaPipe after flip).
    """
    # Fingers: tip y < pip y  →  extended  (y increases downward in image space)
    index_ext  = landmarks[INDEX_TIP][1]  < landmarks[INDEX_PIP][1]
    middle_ext = landmarks[MIDDLE_TIP][1] < landmarks[MIDDLE_PIP][1]
    ring_ext   = landmarks[RING_TIP][1]   < landmarks[RING_PIP][1]
    pinky_ext  = landmarks[PINKY_TIP][1]  < landmarks[PINKY_PIP][1]

    # Thumb: horizontal movement after flip.
    # Right hand → thumb tip is to the LEFT of IP when extended (lower x).
    # Left hand  → thumb tip is to the RIGHT of IP when extended (higher x).
    if handedness == "Right":
        thumb_ext = landmarks[THUMB_TIP][0] < landmarks[THUMB_IP][0]
    else:
        thumb_ext = landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0]

    return FingerState(
        thumb=thumb_ext,
        index=index_ext,
        middle=middle_ext,
        ring=ring_ext,
        pinky=pinky_ext,
    )
