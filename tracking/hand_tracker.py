"""
MediaPipe Hands wrapper.

HandTracker.process() returns a TrackingResult containing structured
HandData for left and right hands, plus the raw MediaPipe result needed
for built-in landmark drawing.

Smoothing is applied per-hand via LandmarkSmoother instances so that
downstream code always receives jitter-reduced coordinates.
"""

from __future__ import annotations

import mediapipe as mp
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    MP_MAX_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
    HANDEDNESS_STABLE_FRAMES,
    HANDEDNESS_MIN_CONFIDENCE,
)
from core.smoothing import LandmarkSmoother
from tracking.hand_landmarks import HandData, extract_landmarks
from tracking.handedness import get_handedness_label, get_handedness_score
from utils.constants import HAND_LEFT, HAND_RIGHT


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrackingResult:
    """All hand-tracking data produced in a single frame."""
    hands: List[HandData] = field(default_factory=list)
    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    raw_result: object = None   # mediapipe Hands result, kept for drawing


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class HandTracker:
    """
    Wraps mediapipe.solutions.hands and returns structured TrackingResult
    objects each frame.
    """

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_HANDS,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )

        # One smoother per possible hand slot (keyed by handedness label)
        self._smoothers: dict[str, LandmarkSmoother] = {
            HAND_LEFT: LandmarkSmoother(),
            HAND_RIGHT: LandmarkSmoother(),
        }
        # Track which hands were visible last frame to reset smoothers on re-entry
        self._prev_visible: set[str] = set()

        # ------------------------------------------------------------------
        # Handedness debounce
        # ------------------------------------------------------------------
        # A new label must appear for HANDEDNESS_STABLE_FRAMES consecutive
        # frames (or with high confidence) before we route the hand to the
        # corresponding controller slot.  This prevents single-frame label
        # flips from swapping left/right roles.
        #
        # _stable_label[slot] = currently confirmed label for this slot
        # _candidate[slot]    = (candidate_label, consecutive_count)
        # Slot keys are integers (index into multi_hand_landmarks).
        # We use per-slot tracking because MediaPipe slot ordering can vary.
        self._slot_confirmed: dict[int, str]        = {}
        self._slot_candidate: dict[int, tuple]      = {}  # (label, count)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        frame_rgb,           # uint8 RGB frame (writeable=False is fine)
        frame_w: int,
        frame_h: int,
    ) -> TrackingResult:
        """
        Process one RGB frame and return a TrackingResult.

        The caller is responsible for converting BGR → RGB before calling.
        """
        result = TrackingResult()
        mp_result = self._hands.process(frame_rgb)
        result.raw_result = mp_result

        if not mp_result.multi_hand_landmarks:
            # No hands: reset smoothers and clear debounce state
            for label in self._prev_visible:
                self._smoothers[label].reset()
            self._prev_visible.clear()
            self._slot_confirmed.clear()
            self._slot_candidate.clear()
            return result

        current_visible: set[str] = set()
        # slots present this frame (to prune stale slot state)
        current_slots: set[int] = set()

        for i, hand_landmarks in enumerate(mp_result.multi_hand_landmarks):
            raw_label = get_handedness_label(mp_result.multi_handedness, i)
            score     = get_handedness_score(mp_result.multi_handedness, i)
            current_slots.add(i)

            # ----------------------------------------------------------
            # Handedness debounce
            # ----------------------------------------------------------
            # High-confidence labels are accepted immediately.
            # Low-confidence labels must be seen for HANDEDNESS_STABLE_FRAMES
            # consecutive frames before they are promoted.
            confirmed_label = self._resolve_label(i, raw_label, score)
            if confirmed_label is None:
                # Label is not yet stable enough to route — skip this hand
                # but still process landmarks so skeleton draw works.
                raw_pixels = extract_landmarks(hand_landmarks, frame_w, frame_h)
                # Build a minimal HandData with the raw label so rendering works
                hand_data = HandData(
                    landmarks=raw_pixels,
                    handedness=raw_label,
                    confidence=score,
                    smoothed_landmarks=raw_pixels,   # unsmoothed until stable
                )
                result.hands.append(hand_data)
                continue

            # Confirmed label — route to the correct slot
            # Reset smoother if this label just (re-)appeared
            if confirmed_label not in self._prev_visible:
                self._smoothers[confirmed_label].reset()

            raw_pixels = extract_landmarks(hand_landmarks, frame_w, frame_h)
            smoothed   = self._smoothers[confirmed_label].update(raw_pixels)

            hand_data = HandData(
                landmarks=raw_pixels,
                handedness=confirmed_label,
                confidence=score,
                smoothed_landmarks=smoothed,
            )

            result.hands.append(hand_data)
            current_visible.add(confirmed_label)

            if confirmed_label == HAND_LEFT:
                result.left_hand = hand_data
            else:
                result.right_hand = hand_data

        # Prune slot debounce state for slots that disappeared this frame
        for slot in list(self._slot_confirmed.keys()):
            if slot not in current_slots:
                del self._slot_confirmed[slot]
        for slot in list(self._slot_candidate.keys()):
            if slot not in current_slots:
                del self._slot_candidate[slot]

        # Reset smoothers for labels that dropped out this frame
        for label in self._prev_visible - current_visible:
            self._smoothers[label].reset()

        self._prev_visible = current_visible
        return result

    # ------------------------------------------------------------------
    # Handedness debounce helper
    # ------------------------------------------------------------------

    def _resolve_label(self, slot: int, raw_label: str, score: float) -> "str | None":
        """
        Return the confirmed label for this detection slot, or None if the
        label is not yet stable enough to be used for routing.

        High-confidence readings (≥ HANDEDNESS_MIN_CONFIDENCE) are accepted
        immediately.  Lower-confidence readings must persist for
        HANDEDNESS_STABLE_FRAMES consecutive frames.
        """
        # High-confidence: accept immediately and reset any pending candidate
        if score >= HANDEDNESS_MIN_CONFIDENCE:
            self._slot_confirmed[slot] = raw_label
            self._slot_candidate[slot] = (raw_label, HANDEDNESS_STABLE_FRAMES)
            return raw_label

        # Low confidence: require stability
        prev_candidate, prev_count = self._slot_candidate.get(slot, (None, 0))

        if raw_label == prev_candidate:
            new_count = prev_count + 1
        else:
            # New candidate — restart count
            new_count = 1

        self._slot_candidate[slot] = (raw_label, new_count)

        if new_count >= HANDEDNESS_STABLE_FRAMES:
            self._slot_confirmed[slot] = raw_label
            return raw_label

        # Not yet stable — return last confirmed if available
        return self._slot_confirmed.get(slot, None)

    # ------------------------------------------------------------------
    # Rendering helper
    # ------------------------------------------------------------------

    def draw_landmarks(self, frame, raw_result) -> None:
        """
        Draw the standard MediaPipe hand skeleton onto frame in-place.
        Uses the built-in coloured landmark/connection styles.
        """
        if raw_result is None or not raw_result.multi_hand_landmarks:
            return
        for hand_landmarks in raw_result.multi_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
