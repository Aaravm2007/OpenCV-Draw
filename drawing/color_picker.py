"""
Colour-picker UI.

Renders a vertical column of colour circles in the top-right corner.
The right-hand index fingertip selects a colour by dwelling over it
for COLOR_DWELL_FRAMES frames (prevents accidental selections while drawing
near the edge of the screen).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import (
    DRAW_COLORS,
    COLOR_PICKER_RADIUS,
    COLOR_PICKER_X,
    COLOR_PICKER_Y_START,
    COLOR_PICKER_SPACING,
    COLOR_HOVER_TOLERANCE,
)
from utils.geometry import distance_2d


# Frames the fingertip must dwell over a swatch to confirm selection.
_DWELL_FRAMES = 6


class ColorPicker:

    def __init__(self, frame_width: int) -> None:
        self._frame_w = frame_width
        self._colors: List[Tuple[int, int, int]] = list(DRAW_COLORS)
        self._selected: int = 0   # index into _colors

        # Dwell tracking
        self._dwell_candidate: Optional[int] = None
        self._dwell_count: int = 0

        # Pre-compute circle centres (updated if frame size changes)
        self._centres: List[Tuple[int, int]] = self._compute_centres()

    # ------------------------------------------------------------------

    def update(
        self,
        fingertip: Optional[Tuple[float, float]],
        is_index_up: bool,
    ) -> bool:
        """
        Call every frame with the right-hand index-tip position.

        Args:
            fingertip:   (x, y) in pixel space, or None if hand not visible.
            is_index_up: True when right hand is in INDEX_ONLY gesture.

        Returns True if a new colour was just selected (so the caller can
        suppress drawing on that frame).
        """
        if fingertip is None or not is_index_up:
            self._dwell_candidate = None
            self._dwell_count = 0
            return False

        hit = self._hit_test(fingertip)

        if hit is None:
            self._dwell_candidate = None
            self._dwell_count = 0
            return False

        if hit == self._dwell_candidate:
            self._dwell_count += 1
        else:
            self._dwell_candidate = hit
            self._dwell_count = 1

        if self._dwell_count >= _DWELL_FRAMES:
            self._selected = hit
            self._dwell_candidate = None
            self._dwell_count = 0
            return True   # colour just changed

        return False   # still dwelling, not yet confirmed

    def draw(self, frame: np.ndarray) -> None:
        """Render colour swatches onto frame in-place."""
        for i, (cx, cy) in enumerate(self._centres):
            color = self._colors[i]
            is_selected = (i == self._selected)

            # Outer ring: white if selected, grey otherwise
            ring_color  = (255, 255, 255) if is_selected else (120, 120, 120)
            ring_thick  = 3               if is_selected else 1
            cv2.circle(frame, (cx, cy), COLOR_PICKER_RADIUS + 3, ring_color, ring_thick)

            # Filled swatch
            cv2.circle(frame, (cx, cy), COLOR_PICKER_RADIUS, color, -1)

            # Dwell progress arc
            if self._dwell_candidate == i and self._dwell_count > 0:
                angle = int(360 * self._dwell_count / _DWELL_FRAMES)
                cv2.ellipse(frame, (cx, cy),
                            (COLOR_PICKER_RADIUS + 6, COLOR_PICKER_RADIUS + 6),
                            -90, 0, angle, (0, 255, 200), 2)

    # ------------------------------------------------------------------

    @property
    def active_color(self) -> Tuple[int, int, int]:
        return self._colors[self._selected]

    @property
    def active_color_name(self) -> str:
        names = ["White", "Red", "Green", "Blue", "Yellow", "Magenta", "Orange"]
        if self._selected < len(names):
            return names[self._selected]
        return f"Color {self._selected}"

    def is_over_picker(self, point: Tuple[float, float]) -> bool:
        """True if point is near any swatch (use to suppress drawing)."""
        return self._hit_test(point) is not None

    # ------------------------------------------------------------------

    def _hit_test(self, pt: Tuple[float, float]) -> Optional[int]:
        r = COLOR_PICKER_RADIUS + COLOR_HOVER_TOLERANCE
        for i, centre in enumerate(self._centres):
            if distance_2d(pt, centre) <= r:
                return i
        return None

    def _compute_centres(self) -> List[Tuple[int, int]]:
        centres = []
        for i in range(len(self._colors)):
            cx = self._frame_w - COLOR_PICKER_X
            cy = COLOR_PICKER_Y_START + i * COLOR_PICKER_SPACING
            centres.append((cx, cy))
        return centres
