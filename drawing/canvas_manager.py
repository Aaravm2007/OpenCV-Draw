"""
Canvas manager.

Owns the list of completed strokes and the in-progress stroke.
Renders all strokes onto the frame each tick.

Strokes are stored in *canonical* pixel space (pan=0, zoom=1).
Pass pan_px and zoom to render() so they appear correctly under the
current scene transform.  The drawing caller is responsible for
inverse-transforming the tip position before storing.
"""

from __future__ import annotations

import math
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
        """Begin a new in-progress stroke at point (canonical space)."""
        self._current = Stroke(color=color, thickness=thickness)
        self._current.add_point(point)

    def update_stroke(self, point: Tuple[int, int]) -> None:
        """Extend the in-progress stroke (canonical space)."""
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

    def render(
        self,
        frame: np.ndarray,
        pan_px: Tuple[float, float] = (0.0, 0.0),
        zoom: float = 1.0,
    ) -> None:
        """Draw all strokes onto frame, applying pan_px and zoom."""
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        for stroke in self._strokes:
            self._draw_stroke(frame, stroke, cx, cy, pan_px, zoom)
        if self._current is not None:
            self._draw_stroke(frame, self._current, cx, cy, pan_px, zoom)

    # ------------------------------------------------------------------

    @staticmethod
    def _draw_stroke(
        frame: np.ndarray,
        stroke: Stroke,
        cx: float = 0.0,
        cy: float = 0.0,
        pan_px: Tuple[float, float] = (0.0, 0.0),
        zoom: float = 1.0,
    ) -> None:
        def tf(p: Tuple[int, int]) -> Tuple[int, int]:
            return (
                int(cx + (p[0] - cx) * zoom + pan_px[0]),
                int(cy + (p[1] - cy) * zoom + pan_px[1]),
            )

        pts = stroke.points
        for i in range(1, len(pts)):
            cv2.line(frame, tf(pts[i - 1]), tf(pts[i]),
                     stroke.color, stroke.thickness, cv2.LINE_AA)
        if len(pts) == 1:
            cv2.circle(frame, tf(pts[0]), stroke.thickness // 2, stroke.color, -1)

    # ------------------------------------------------------------------
    # Hit-testing
    # ------------------------------------------------------------------

    def find_nearest_stroke(
        self,
        screen_point: Tuple[float, float],
        pan_px: Tuple[float, float] = (0.0, 0.0),
        zoom: float = 1.0,
        cx: float = 0.0,
        cy: float = 0.0,
        threshold_px: float = 40.0,
    ) -> Optional[Stroke]:
        """Return the topmost stroke whose rendered position is within
        threshold_px of screen_point, or None."""
        px, py = screen_point
        best: Optional[Stroke] = None
        best_d = threshold_px
        for stroke in reversed(self._strokes):  # top-most (last drawn) first
            for pt in stroke.points:
                rx = cx + (pt[0] - cx) * zoom + pan_px[0]
                ry = cy + (pt[1] - cy) * zoom + pan_px[1]
                d = math.hypot(rx - px, ry - py)
                if d < best_d:
                    best_d = d
                    best = stroke
        return best

    # ------------------------------------------------------------------
    # Load support
    # ------------------------------------------------------------------

    def add_stroke(self, stroke: "Stroke") -> None:
        """Append a pre-built stroke (used by load)."""
        self._strokes.append(stroke)

    def remove_stroke(self, stroke: "Stroke") -> None:
        """Remove a specific stroke (used by dustbin delete)."""
        try:
            self._strokes.remove(stroke)
        except ValueError:
            pass

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
