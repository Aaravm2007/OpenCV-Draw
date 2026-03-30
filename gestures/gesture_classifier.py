"""
Maps raw finger/pinch state to a named Gesture enum value.

Priority order (highest first):
  1. PINCH      — thumb+index close together
  2. INDEX_ONLY — only index extended (draw mode)
  3. OPEN_PALM  — all four fingers extended
  4. FIST       — all fingers closed
  5. UNKNOWN    — anything else
"""

from __future__ import annotations
from enum import Enum, auto

from gestures.finger_state import FingerState
from gestures.pinch import PinchState


class Gesture(Enum):
    UNKNOWN    = auto()
    FIST       = auto()
    OPEN_PALM  = auto()
    INDEX_ONLY = auto()   # draw mode trigger
    PINCH      = auto()   # object / scene manipulation trigger


class GestureClassifier:
    """Stateless classifier — call classify() every frame."""

    def classify(
        self,
        finger_state: FingerState,
        pinch_state: PinchState,
    ) -> Gesture:
        # 1. Pinch overrides everything
        if pinch_state.is_pinching:
            return Gesture.PINCH

        idx = finger_state.index
        mid = finger_state.middle
        rng = finger_state.ring
        pnk = finger_state.pinky

        # 2. Index only → draw mode
        if idx and not mid and not rng and not pnk:
            return Gesture.INDEX_ONLY

        # 3. All four fingers up → open palm
        if idx and mid and rng and pnk:
            return Gesture.OPEN_PALM

        # 4. All four fingers down → fist
        if not idx and not mid and not rng and not pnk:
            return Gesture.FIST

        return Gesture.UNKNOWN
