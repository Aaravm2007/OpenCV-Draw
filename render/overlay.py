"""
Debug overlay.

Draws a semi-transparent HUD showing:
- FPS, app state
- Canvas stroke count / draw status, active colour
- Object count, selected object
- Scene transform (pan, zoom, rotation)
- Per-hand: gesture, finger string, pinch distance
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

import config
from config import (
    DEBUG_FONT_SCALE,
    DEBUG_THICKNESS,
    DEBUG_LINE_HEIGHT,
    COLOR_DEBUG_TEXT,
    COLOR_DEBUG_SHADOW,
)


class DebugOverlay:

    FONT          = cv2.FONT_HERSHEY_SIMPLEX
    PANEL_X       = 10
    PANEL_Y_START = 28
    PANEL_W       = 340   # widened to prevent long lines from clipping

    def __init__(self) -> None:
        self._fps_prev_time: float = time.time()
        self._fps: float = 0.0

    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        state_manager,
        tracking_result,
        left_gesture=None,
        right_gesture=None,
        left_finger=None,
        right_finger=None,
        left_pinch=None,
        right_pinch=None,
        canvas=None,
        color_picker=None,
        object_manager=None,
        scene_controller=None,
        session_lock=None,
        active_mode: str = "IDLE",
        extra_lines: Optional[list] = None,
    ) -> None:
        if not config.DEBUG_MODE:
            return

        self._update_fps()
        lines = self._build_lines(
            state_manager, tracking_result,
            left_gesture, right_gesture,
            left_finger, right_finger,
            left_pinch, right_pinch,
            canvas, color_picker,
            object_manager, scene_controller,
            session_lock, active_mode,
        )
        if extra_lines:
            lines += [str(l) for l in extra_lines]

        self._render_panel(frame, lines, active_mode)
        if color_picker is not None:
            self._draw_active_color_swatch(frame, color_picker)

    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        now = time.time()
        dt  = now - self._fps_prev_time
        if dt > 0:
            # alpha=0.25 converges in ~4 frames — fast enough for display,
            # slow enough to avoid single-frame spikes dominating the number.
            self._fps = 0.75 * self._fps + 0.25 / dt
        self._fps_prev_time = now

    # Colour-codes for active_mode in the HUD (BGR)
    _MODE_COLORS = {
        "OBJECT_MANIPULATE": (80,  200, 255),
        "SCENE_CONTROL":     (255, 180,  60),
        "COLOR_PICK":        (255, 255,  80),
        "DRAW":              (80,  255, 180),
        "IDLE":              (160, 160, 160),
    }

    def _build_lines(
        self, state_manager, tracking_result,
        left_gesture, right_gesture,
        left_finger, right_finger,
        left_pinch, right_pinch,
        canvas, color_picker,
        object_manager, scene_controller,
        session_lock=None, active_mode: str = "IDLE",
    ) -> list:
        lines = []
        lines.append(f"FPS: {self._fps:.1f}  |  {state_manager.state.name}")
        lines.append(f"Mode: {active_mode}")

        # Session lock info
        if session_lock is not None:
            lock_str = (
                f"locked  lost={session_lock.lost_frames}f"
                if session_lock.locked
                else "unlocked"
            )
            lines.append(f"Lock: {lock_str}")

        if canvas is not None:
            status = "DRAWING" if canvas.is_drawing else "idle"
            lines.append(f"Canvas: {canvas.stroke_count} strokes  [{status}]")

        if color_picker is not None:
            lines.append(f"Color: {color_picker.active_color_name}")

        if object_manager is not None:
            sel     = object_manager.selected
            sel_str = f"{sel.object_type}#{sel.uid}" if sel else "none"
            lines.append(f"Objects: {len(object_manager.objects)}  sel={sel_str}")

        if scene_controller is not None:
            sc = scene_controller
            px, py = sc.pan[0], sc.pan[1]
            rz_deg = math.degrees(sc.rotation[2])
            lines.append(
                f"Scene: pan=({px:.1f},{py:.1f}) "
                f"zoom={sc.zoom:.2f} rot={rz_deg:.0f}\u00b0"
            )

        lines.append("\u2500" * 26)

        def hand_block(label, hand, gesture, finger, pinch):
            if hand is None:
                lines.append(f"{label}: \u2014")
                return
            gname = gesture.name if gesture is not None else "?"
            lines.append(f"{label}: {gname}  conf={hand.confidence:.2f}")
            if finger is not None:
                lines.append(f"  fingers: {finger}")
            if pinch is not None:
                pstr = "\u25cf YES" if pinch.is_pinching else "no"
                ang  = math.degrees(pinch.angle)
                lines.append(
                    f"  pinch: {pstr}  "
                    f"n={pinch.normalized_distance:.3f}  "
                    f"ang={ang:.0f}\u00b0"
                )

        hand_block("Left ", tracking_result.left_hand,
                   left_gesture, left_finger, left_pinch)
        hand_block("Right", tracking_result.right_hand,
                   right_gesture, right_finger, right_pinch)

        return lines

    def _draw_active_color_swatch(self, frame: np.ndarray, color_picker) -> None:
        sx = self.PANEL_X + self.PANEL_W - 20
        sy = self.PANEL_Y_START + 42
        color = color_picker.active_color
        cv2.rectangle(frame, (sx, sy), (sx + 14, sy + 14), color, -1)
        cv2.rectangle(frame, (sx, sy), (sx + 14, sy + 14), (200, 200, 200), 1)

    # ------------------------------------------------------------------
    # Mode banner  (always-visible, independent of DEBUG_MODE)
    # ------------------------------------------------------------------

    # State → (label, BGR colour)
    _BANNER_STYLES = {
        "IDLE":                  ("IDLE — show open palms to begin",  (120, 120, 120)),
        "WAITING_FOR_BOTH_PALMS": ("ACTIVATING…",                     (60,  200, 255)),
        "ACTIVE_TRACKING":       ("ACTIVE",                           (60,  220,  60)),
        "USER_LOST":             ("USER LOST — show hands to resume", (40,   80, 255)),
    }

    # active_mode → short display text
    _MODE_LABELS = {
        "OBJECT_MANIPULATE": "OBJ",
        "SCENE_CONTROL":     "SCN",
        "COLOR_PICK":        "CLR",
        "DRAW":              "DRAW",
        "IDLE":              "",
    }

    def draw_mode_banner(
        self,
        frame: np.ndarray,
        state_name: str,
        left_gesture_name: str = "",
        right_gesture_name: str = "",
        active_mode: str = "IDLE",
        flash_text: str = "",
    ) -> None:
        """Draw a slim always-on status bar at the bottom of the frame."""
        if not config.MODE_BANNER_ENABLED:
            return

        h, w = frame.shape[:2]
        bar_h = 30
        bar_y = h - bar_h

        label, color = self._BANNER_STYLES.get(
            state_name,
            (state_name, (160, 160, 160)),
        )

        # Background strip
        cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)

        # State label
        cv2.putText(frame, label,
                    (12, bar_y + 20),
                    self.FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label,
                    (12, bar_y + 20),
                    self.FONT, 0.55, color, 1, cv2.LINE_AA)

        # Active interaction mode badge (shown after state label)
        mode_tag = self._MODE_LABELS.get(active_mode, "")
        if mode_tag:
            mode_color = self._MODE_COLORS.get(active_mode, (200, 200, 200))
            (state_tw, _), _ = cv2.getTextSize(label, self.FONT, 0.55, 1)
            badge_x = 12 + state_tw + 12
            cv2.putText(frame, f"[{mode_tag}]",
                        (badge_x + 1, bar_y + 20),
                        self.FONT, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"[{mode_tag}]",
                        (badge_x, bar_y + 20),
                        self.FONT, 0.48, mode_color, 1, cv2.LINE_AA)

        # Per-hand gesture hints — right-aligned pair
        _skip = {"UNKNOWN", ""}
        hints = []
        if left_gesture_name  and left_gesture_name  not in _skip:
            hints.append(f"L:{left_gesture_name}")
        if right_gesture_name and right_gesture_name not in _skip:
            hints.append(f"R:{right_gesture_name}")
        if hints:
            hint_str = "  ".join(hints)
            (tw, _), _ = cv2.getTextSize(hint_str, self.FONT, 0.45, 1)
            cv2.putText(frame, hint_str,
                        (w - tw - 12, bar_y + 20),
                        self.FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hint_str,
                        (w - tw - 12, bar_y + 20),
                        self.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Transient flash message (centred)
        if flash_text:
            (tw, _), _ = cv2.getTextSize(flash_text, self.FONT, 0.50, 1)
            tx = (w - tw) // 2
            cv2.putText(frame, flash_text,
                        (tx + 1, bar_y + 21),
                        self.FONT, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, flash_text,
                        (tx, bar_y + 20),
                        self.FONT, 0.50, (0, 255, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------

    def _render_panel(
        self,
        frame: np.ndarray,
        lines: list,
        active_mode: str = "IDLE",
    ) -> None:
        panel_h = len(lines) * DEBUG_LINE_HEIGHT + 10
        x = self.PANEL_X - 4
        y = self.PANEL_Y_START - 20

        sub = frame[max(0, y): y + panel_h, x: x + self.PANEL_W]
        if sub.size > 0:
            frame[max(0, y): y + panel_h, x: x + self.PANEL_W] = (
                sub * 0.4
            ).astype(np.uint8)

        mode_color = self._MODE_COLORS.get(active_mode, COLOR_DEBUG_TEXT)

        for i, line in enumerate(lines):
            pos = (self.PANEL_X, self.PANEL_Y_START + i * DEBUG_LINE_HEIGHT)
            # The "Mode:" line (index 1) gets a unique colour for quick scanning
            text_color = mode_color if line.startswith("Mode:") else COLOR_DEBUG_TEXT
            cv2.putText(frame, line, (pos[0] + 1, pos[1] + 1),
                        self.FONT, DEBUG_FONT_SCALE, COLOR_DEBUG_SHADOW,
                        DEBUG_THICKNESS + 1, cv2.LINE_AA)
            cv2.putText(frame, line, pos,
                        self.FONT, DEBUG_FONT_SCALE, text_color,
                        DEBUG_THICKNESS, cv2.LINE_AA)
