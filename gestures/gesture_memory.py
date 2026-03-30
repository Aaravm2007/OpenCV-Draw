"""
Gesture debounce / temporal memory.

A gesture must appear for GESTURE_DEBOUNCE_FRAMES consecutive frames
before it is reported as confirmed.  This eliminates single-frame noise
and brief mis-classifications between stable gestures.
"""

from __future__ import annotations

from gestures.gesture_classifier import Gesture
from config import GESTURE_DEBOUNCE_FRAMES


class GestureMemory:
    """
    Per-hand debounce buffer.

    Usage each frame::

        raw = classifier.classify(finger_state, pinch_state)
        confirmed = memory.update(raw)
    """

    def __init__(self, debounce_frames: int = GESTURE_DEBOUNCE_FRAMES) -> None:
        self._debounce = debounce_frames
        self._candidate: Gesture = Gesture.UNKNOWN
        self._candidate_count: int = 0
        self._confirmed: Gesture = Gesture.UNKNOWN

    # ------------------------------------------------------------------

    def update(self, raw: Gesture) -> Gesture:
        """
        Feed the raw gesture for this frame.
        Returns the stable (debounced) gesture.
        """
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1

        if self._candidate_count >= self._debounce:
            self._confirmed = self._candidate

        return self._confirmed

    def reset(self) -> None:
        """Call when the hand disappears so state starts fresh on re-entry."""
        self._candidate = Gesture.UNKNOWN
        self._candidate_count = 0
        self._confirmed = Gesture.UNKNOWN

    @property
    def confirmed(self) -> Gesture:
        return self._confirmed
