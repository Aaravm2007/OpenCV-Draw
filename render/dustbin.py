"""
Dustbin (trash) UI widget.

Rendered as a labelled rectangle in the bottom-right corner.
When an object is being dragged over it the widget highlights red,
signalling that releasing the pinch will delete the object.
"""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


class Dustbin:

    W = 70
    H = 60
    MARGIN  = 16   # distance from frame edge
    BANNER_H = 30  # mode banner at bottom — dustbin sits above it

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.x = frame_width  - self.W - self.MARGIN
        self.y = frame_height - self.H - self.MARGIN - self.BANNER_H
        self.highlight: bool = False   # True when a dragged object is hovering over it

    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> None:
        color    = (0, 60, 220)  if self.highlight else (100, 100, 100)
        bg_color = (0, 20, 80)   if self.highlight else (30, 30, 30)

        # Background fill
        cv2.rectangle(frame,
                      (self.x, self.y),
                      (self.x + self.W, self.y + self.H),
                      bg_color, -1)
        # Border
        thick = 2 if self.highlight else 1
        cv2.rectangle(frame,
                      (self.x, self.y),
                      (self.x + self.W, self.y + self.H),
                      color, thick)

        # Icon lines (simple bin silhouette)
        mx = self.x + self.W // 2
        # Lid
        cv2.line(frame, (self.x + 8, self.y + 14),
                 (self.x + self.W - 8, self.y + 14), color, 2)
        cv2.line(frame, (mx - 8, self.y + 8),
                 (mx + 8, self.y + 8), color, 2)
        # Body outline
        cv2.rectangle(frame,
                      (self.x + 10, self.y + 16),
                      (self.x + self.W - 10, self.y + self.H - 8),
                      color, 1)
        # Vertical stripes inside body
        for lx in (mx - 7, mx, mx + 7):
            cv2.line(frame,
                     (lx, self.y + 19),
                     (lx, self.y + self.H - 11),
                     color, 1)

        # Label
        label_color = (0, 120, 255) if self.highlight else (160, 160, 160)
        cv2.putText(frame, "DEL",
                    (self.x + 12, self.y + self.H - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)

    def is_over(self, point: Tuple[float, float]) -> bool:
        x, y = int(point[0]), int(point[1])
        return self.x <= x <= self.x + self.W and self.y <= y <= self.y + self.H
