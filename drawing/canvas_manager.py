"""
Canvas manager.

Owns the list of completed strokes and the in-progress stroke.
Renders all strokes onto the frame each tick.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from drawing.stroke import Stroke
from config import DRAW_STROKE_THICKNESS


class CanvasManager:

    def __init__(self) -> None:
        self._strokes:  List[Stroke]     = []
        self._current:  Optional[Stroke] = None

    # ------------------------------------------------------------------
    # Stroke control
    # ------------------------------------------------------------------

    def start_stroke(
        self,
        point: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = DRAW_STROKE_THICKNESS,
    ) -> None:
        """Begin a new in-progress stroke at point."""
        self._current = Stroke(color=color, thickness=thickness)
        self._current.add_point(point)

    def update_stroke(self, point: Tuple[int, int]) -> None:
        """Extend the in-progress stroke with a new point."""
        if self._current is not None:
            self._current.add_point(point)

    def end_stroke(self) -> None:
        """Finalise the in-progress stroke and move it to the completed list."""
        if self._current is not None:
            if self._current.is_drawable():
                self._strokes.append(self._current)
            self._current = None

    def cancel_stroke(self) -> None:
        """Discard the in-progress stroke without saving it."""
        self._current = None

    def clear(self) -> None:
        """Remove all strokes including any in-progress one."""
        self._strokes.clear()
        self._current = None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, frame: np.ndarray) -> None:
        """Draw all strokes (including in-progress) onto frame in-place."""
        for stroke in self._strokes:
            self._draw_stroke(frame, stroke)
        if self._current is not None:
            self._draw_stroke(frame, self._current)

    # ------------------------------------------------------------------

    @staticmethod
    def _draw_stroke(frame: np.ndarray, stroke: Stroke) -> None:
        pts = stroke.points
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i],
                     stroke.color, stroke.thickness, cv2.LINE_AA)
        # Draw a filled circle at the very tip of an in-progress single-point stroke
        if len(pts) == 1:
            cv2.circle(frame, pts[0], stroke.thickness // 2, stroke.color, -1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def add_stroke(self, stroke: "Stroke") -> None:
        """Append a pre-built stroke (used by load)."""
        self._strokes.append(stroke)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_drawing(self) -> bool:
        return self._current is not None

    @property
    def stroke_count(self) -> int:
        return len(self._strokes)

    @property
    def strokes(self) -> List[Stroke]:
        return list(self._strokes)
