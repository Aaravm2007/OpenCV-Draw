"""
Tutorial / controls-guide overlay.

Shown for TUTORIAL_DISPLAY_SECONDS after activation (or on demand via 't').
Fades out smoothly over the last second.
"""

from __future__ import annotations

import time
import cv2
import numpy as np

from config import TUTORIAL_DISPLAY_SECONDS


# (text, BGR colour, font-scale)
_LINES = [
    ("CONTROLS GUIDE",                      (0,   255, 255), 0.62),
    ("",                                     None,            0.0),
    ("RIGHT HAND",                           (180, 180, 255), 0.50),
    ("  Index up     Draw strokes",          (220, 220, 220), 0.46),
    ("  Pinch        Select / move object",  (220, 220, 220), 0.46),
    ("  Pinch dist   Scale object",          (220, 220, 220), 0.46),
    ("  Pinch twist  Rotate object (Z)",     (220, 220, 220), 0.46),
    ("  Hover bin    Delete selected",       (220, 220, 220), 0.46),
    ("",                                     None,            0.0),
    ("LEFT HAND",                            (255, 180, 180), 0.50),
    ("  Pinch+drag   Pan scene",             (220, 220, 220), 0.46),
    ("  Pinch dist   Zoom",                  (220, 220, 220), 0.46),
    ("  Pinch twist  Rotate scene (Z)",      (220, 220, 220), 0.46),
    ("",                                     None,            0.0),
    ("ACTIVATION",                           (255, 220, 120), 0.50),
    ("  Both palms   Start / re-activate",   (220, 220, 220), 0.46),
    ("",                                     None,            0.0),
    ("KEYBOARD",                             (180, 255, 180), 0.50),
    ("  d  debug    c  clear    r  reset",   (200, 200, 200), 0.43),
    ("  n  new cube t  tutorial s  save",    (200, 200, 200), 0.43),
    ("  l  load     q  quit",                (200, 200, 200), 0.43),
]

_LINE_H  = 24
_PAD_X   = 22
_PAD_Y   = 16
_PANEL_W = 400
_PANEL_H = len(_LINES) * _LINE_H + _PAD_Y * 2


class TutorialOverlay:

    def __init__(self) -> None:
        self._active: bool  = False
        self._start:  float = 0.0

    def activate(self) -> None:
        """Show (or restart) the overlay."""
        self._active = True
        self._start  = time.time()

    def draw(self, frame: np.ndarray) -> None:
        if not self._active:
            return

        elapsed = time.time() - self._start
        if elapsed >= TUTORIAL_DISPLAY_SECONDS:
            self._active = False
            return

        # Alpha: fully opaque for first (dur-1)s, then fade to 0
        fade_start = TUTORIAL_DISPLAY_SECONDS - 1.0
        alpha = 1.0 - max(0.0, elapsed - fade_start)  # clamp to [0,1]

        h, w = frame.shape[:2]
        # Position: right half of frame so it doesn't cover the debug overlay
        px = w - _PANEL_W - _PAD_X * 2
        py = max(_PAD_Y, h // 2 - _PANEL_H // 2)

        # Draw semi-transparent dark panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px - _PAD_X, py - _PAD_Y),
            (px + _PANEL_W, py + _PANEL_H),
            (12, 12, 12), -1,
        )
        cv2.rectangle(
            overlay,
            (px - _PAD_X, py - _PAD_Y),
            (px + _PANEL_W, py + _PANEL_H),
            (70, 70, 70), 1,
        )
        cv2.addWeighted(overlay, 0.80 * alpha, frame, 1.0 - 0.80 * alpha, 0, frame)

        # Draw each text line
        for i, (text, color, scale) in enumerate(_LINES):
            if not text or color is None:
                continue
            y_pos   = py + i * _LINE_H + _LINE_H
            blended = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, text, (px, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (px, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        blended, 1, cv2.LINE_AA)

    @property
    def is_active(self) -> bool:
        return self._active
