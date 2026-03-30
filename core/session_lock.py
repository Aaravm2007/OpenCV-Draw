"""
Session lock.

After activation, records the centroid between both wrists.
Each frame it checks:
  - whether both hands are still present (lost-count tracking)
  - whether any detected hands are plausibly the same user (zone check)

This prevents a passer-by's hand briefly entering frame from hijacking
the active session.
"""

from __future__ import annotations
from typing import Optional, Tuple

from utils.constants import WRIST
from utils.geometry import distance_2d, midpoint_2d
from config import USER_LOST_TIMEOUT_FRAMES


class SessionLock:

    # A hand is considered "foreign" if its wrist is more than this many
    # times the original inter-wrist distance away from the lock centre.
    _ZONE_MULTIPLIER = 2.5

    def __init__(self) -> None:
        self.locked: bool = False
        self._lock_center: Optional[Tuple[float, float]] = None
        self._lock_radius: float = 0.0
        self._lost_count: int = 0

    # ------------------------------------------------------------------

    def lock(self, tracking_result) -> None:
        """
        Snapshot the current hand positions as the 'home zone'.
        Call once when transitioning into ACTIVE_TRACKING.
        """
        left  = tracking_result.left_hand
        right = tracking_result.right_hand

        if left and right:
            lw = left.smoothed_landmarks[WRIST][:2]
            rw = right.smoothed_landmarks[WRIST][:2]
            self._lock_center = midpoint_2d(lw, rw)
            self._lock_radius = distance_2d(lw, rw) * self._ZONE_MULTIPLIER
        elif left:
            self._lock_center = tuple(left.smoothed_landmarks[WRIST][:2])
            self._lock_radius = 200.0
        elif right:
            self._lock_center = tuple(right.smoothed_landmarks[WRIST][:2])
            self._lock_radius = 200.0

        self.locked = True
        self._lost_count = 0

    def unlock(self) -> None:
        """Release the lock; call when returning to IDLE."""
        self.locked = False
        self._lock_center = None
        self._lock_radius = 0.0
        self._lost_count = 0

    # ------------------------------------------------------------------

    def update(self, tracking_result) -> bool:
        """
        Call every frame while ACTIVE_TRACKING.
        Returns True when the user has been absent long enough to trigger
        a USER_LOST transition.
        """
        has_hands = len(tracking_result.hands) > 0

        if has_hands and self.is_within_lock_zone(tracking_result):
            self._lost_count = 0
            return False

        self._lost_count += 1
        return self._lost_count >= USER_LOST_TIMEOUT_FRAMES

    def is_within_lock_zone(self, tracking_result) -> bool:
        """
        True if all detected wrists are within the lock zone.
        Always True when no lock center is set (e.g. before first lock).
        """
        if not self.locked or self._lock_center is None:
            return True

        for hand in tracking_result.hands:
            wrist = hand.smoothed_landmarks[WRIST][:2]
            if distance_2d(wrist, self._lock_center) > self._lock_radius:
                return False
        return True

    @property
    def lost_frames(self) -> int:
        return self._lost_count
