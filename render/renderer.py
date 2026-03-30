"""
Main rendering pipeline.

Layers (in draw order):
  1. Canvas strokes
  2. 3-D wireframe objects  (with scene matrix applied)
  3. MediaPipe hand skeleton
  4. Handedness labels
  5. Gesture visual feedback
  6. Colour picker swatches
  7. Dustbin widget
  8. Activation progress bar
  9. Tutorial overlay
  10. Debug HUD
"""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

from config import COLOR_LEFT_HAND, COLOR_RIGHT_HAND
from core.state_manager import StateManager, AppState
from gestures.gesture_classifier import Gesture
from tracking.hand_tracker import HandTracker, TrackingResult
from render.overlay import DebugOverlay
from render.tutorial_overlay import TutorialOverlay
from utils.constants import INDEX_TIP


class Renderer:

    def __init__(self, hand_tracker: HandTracker) -> None:
        self._hand_tracker  = hand_tracker
        self._debug_overlay = DebugOverlay()
        self._tutorial      = TutorialOverlay()

    # ------------------------------------------------------------------

    def activate_tutorial(self) -> None:
        self._tutorial.activate()

    def render(
        self,
        frame: np.ndarray,
        state_manager: StateManager,
        tracking_result: TrackingResult,
        # gestures
        left_gesture=None,
        right_gesture=None,
        left_finger=None,
        right_finger=None,
        left_pinch=None,
        right_pinch=None,
        # drawing
        canvas=None,
        color_picker=None,
        # 3-D scene
        object_manager=None,
        camera=None,
        scene_matrix=None,
        dustbin=None,
        scene_controller=None,
        # activation
        activation_progress: float = 0.0,
        extra_debug: Optional[list] = None,
        flash_text: str = "",
        session_lock=None,
        active_mode: str = "IDLE",
    ) -> np.ndarray:

        # 1 — canvas strokes (below everything)
        if canvas is not None:
            canvas.render(frame)

        # 2 — 3-D wireframes
        if object_manager is not None and camera is not None:
            object_manager.render(frame, camera, scene_matrix)

        # 3 — hand skeleton
        self._hand_tracker.draw_landmarks(frame, tracking_result.raw_result)

        # 4 — handedness labels
        self._draw_handedness_labels(frame, tracking_result)

        # 5 — gesture feedback
        self._draw_gesture_feedback(frame, tracking_result,
                                    left_gesture, right_gesture,
                                    left_pinch, right_pinch)

        # 6 — colour picker
        if color_picker is not None:
            color_picker.draw(frame)

        # 7 — dustbin
        if dustbin is not None:
            dustbin.draw(frame)

        # 8 — activation bar
        if state_manager.state == AppState.WAITING_FOR_BOTH_PALMS:
            self._draw_activation_bar(frame, activation_progress)

        # 9 — tutorial
        self._tutorial.draw(frame)

        # 10 — debug HUD
        self._debug_overlay.draw(
            frame, state_manager, tracking_result,
            left_gesture=left_gesture,   right_gesture=right_gesture,
            left_finger=left_finger,     right_finger=right_finger,
            left_pinch=left_pinch,       right_pinch=right_pinch,
            canvas=canvas,               color_picker=color_picker,
            object_manager=object_manager,
            scene_controller=scene_controller,
            session_lock=session_lock,
            active_mode=active_mode,
            extra_lines=extra_debug,
        )

        # 11 — mode banner (always visible, drawn last so it's never occluded)
        lg_name = left_gesture.name  if left_gesture  is not None else ""
        rg_name = right_gesture.name if right_gesture is not None else ""
        self._debug_overlay.draw_mode_banner(
            frame,
            state_name=state_manager.state.name,
            left_gesture_name=lg_name,
            right_gesture_name=rg_name,
            active_mode=active_mode,
            flash_text=flash_text,
        )

        return frame

    # ------------------------------------------------------------------

    def _draw_handedness_labels(self, frame, tracking_result):
        for hand in tracking_result.hands:
            wrist = hand.smoothed_landmarks[0]
            x, y  = int(wrist[0]), int(wrist[1])
            color = COLOR_LEFT_HAND if hand.handedness == "Left" else COLOR_RIGHT_HAND
            label = f"{hand.handedness} ({hand.confidence:.2f})"
            cv2.putText(frame, label, (x - 30, y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (x - 30, y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

    def _draw_gesture_feedback(self, frame, tracking_result,
                                left_gesture, right_gesture,
                                left_pinch, right_pinch):
        pairs = [
            (tracking_result.left_hand,  left_gesture,  left_pinch),
            (tracking_result.right_hand, right_gesture, right_pinch),
        ]
        for hand, gesture, pinch in pairs:
            if hand is None or gesture is None:
                continue
            lm = hand.smoothed_landmarks

            if gesture == Gesture.PINCH and pinch is not None:
                mx, my = int(pinch.midpoint[0]), int(pinch.midpoint[1])
                cv2.circle(frame, (mx, my), 10, (0, 255, 255), -1)
                cv2.circle(frame, (mx, my), 12, (0,   0,   0),  2)

            elif gesture == Gesture.INDEX_ONLY:
                tip = lm[INDEX_TIP]
                cv2.circle(frame, (int(tip[0]), int(tip[1])), 8,  (0, 140, 255), -1)
                cv2.circle(frame, (int(tip[0]), int(tip[1])), 10, (0,   0,   0),  2)

    def _draw_activation_bar(self, frame: np.ndarray, progress: float) -> None:
        h, w   = frame.shape[:2]
        bar_h  = 12
        # Keep above the 30-px mode banner at the bottom
        bar_y  = h - 30 - bar_h - 14
        bar_w  = int(w * 0.60)
        bar_x  = (w - bar_w) // 2
        filled = int(bar_w * progress)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        if filled > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + filled, bar_y + bar_h), (0, 220, 120), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

        pct   = int(progress * 100)
        label = f"Hold open palms to activate... {pct}%"
        cv2.putText(frame, label, (bar_x + 1, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (bar_x, bar_y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1, cv2.LINE_AA)
