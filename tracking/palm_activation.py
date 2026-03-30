"""
Palm activation detector.

Fires when both hands show OPEN_PALM for ACTIVATION_STABLE_FRAMES
consecutive frames.  Exposes a progress float (0→1) so the renderer
can display a countdown bar.

Stability design
----------------
The detector accepts both *raw* (unfiltered) and *confirmed* (debounced)
gesture values. It advances the counter when EITHER hand shows a raw
OPEN_PALM, but only if the other hand is also showing at least a raw
OPEN_PALM.  This lets the progress bar start immediately as the user
opens both hands, rather than waiting 4+4 extra debounce frames.

A 1-frame glitch (single frame where one palm dips to UNKNOWN) is forgiven
by the `_grace_frames` window, preventing annoying resets from minor jitter.
"""

from __future__ import annotations

from gestures.gesture_classifier import Gesture
from config import ACTIVATION_STABLE_FRAMES

# Allow up to this many consecutive non-palm frames before resetting the
# activation counter.  Keeps UX smooth when tracking briefly hiccups.
_GRACE_FRAMES: int = 3


class PalmActivationDetector:
    """
    Watches the gesture on both hands each frame.
    Call update() every frame; returns True exactly once per activation event,
    then resets so re-activation is possible after a new session.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._grace: int = 0   # frames of forgiveness remaining

    def update(self, left_gesture: Gesture, right_gesture: Gesture) -> bool:
        """
        Feed this frame's gestures (raw or debounced — caller's choice).
        Returns True when the stable-hold threshold is reached.
        """
        both_open = (
            left_gesture  == Gesture.OPEN_PALM and
            right_gesture == Gesture.OPEN_PALM
        )

        if both_open:
            self._count += 1
            self._grace  = _GRACE_FRAMES   # replenish grace on good frames
        else:
            if self._grace > 0:
                # Brief interruption — forgive and keep the counter alive
                self._grace  -= 1
                self._count  += 1          # still advancing (slowly)
            else:
                self._count = 0

        if self._count >= ACTIVATION_STABLE_FRAMES:
            self._count = 0   # reset so it doesn't keep firing
            self._grace = 0
            return True

        return False

    @property
    def progress(self) -> float:
        """0.0 … 1.0 — how far toward the activation threshold."""
        return min(self._count / max(ACTIVATION_STABLE_FRAMES, 1), 1.0)

    def reset(self) -> None:
        self._count = 0
        self._grace = 0
