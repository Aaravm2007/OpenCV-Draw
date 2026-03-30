"""
Landmark extraction and accessors.

HandData    — dataclass that carries one hand's pixel-space landmarks,
              handedness string, and detection confidence.

Helpers to convert raw MediaPipe NormalizedLandmarkList to pixel coordinates
and to fetch individual landmark positions conveniently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HandData:
    """All per-hand data produced each frame."""
    landmarks: List[Tuple[float, float, float]]  # pixel (x, y, z)
    handedness: str                               # 'Left' or 'Right'
    confidence: float                            # MediaPipe classification score
    smoothed_landmarks: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # If no smoothed data was supplied, fall back to raw landmarks
        if not self.smoothed_landmarks:
            self.smoothed_landmarks = list(self.landmarks)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_landmarks(
    hand_landmarks,           # mediapipe NormalizedLandmarkList
    frame_width: int,
    frame_height: int,
) -> List[Tuple[float, float, float]]:
    """
    Convert MediaPipe normalised landmarks to pixel coordinates.

    z is already relative to the wrist and roughly proportional to depth;
    we scale it by frame_width to keep the unit consistent with x/y.
    """
    result: List[Tuple[float, float, float]] = []
    for lm in hand_landmarks.landmark:
        x = lm.x * frame_width
        y = lm.y * frame_height
        z = lm.z * frame_width   # depth, scaled to pixel-space order of magnitude
        result.append((x, y, z))
    return result


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_landmark(
    landmarks: List[Tuple[float, float, float]],
    idx: int,
) -> Tuple[float, float, float]:
    """Return the 3-D landmark at index idx."""
    return landmarks[idx]


def get_landmark_2d(
    landmarks: List[Tuple[float, float, float]],
    idx: int,
) -> Tuple[int, int]:
    """Return (x, y) pixel coordinates as integers for drawing."""
    lm = landmarks[idx]
    return (int(lm[0]), int(lm[1]))
