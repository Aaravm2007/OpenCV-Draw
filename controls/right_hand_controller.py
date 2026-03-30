"""
Right-hand object interaction controller.

Gesture priority (checked in order each frame):
  1. Pinch over dustbin while object selected  → delete on release
  2. Pinch starts near an object               → select it
  3. Pinch held with selected object           → move / scale / rotate
  4. Pinch released                            → deselect / finalise

Transform math
--------------
All deltas are computed relative to the state captured at pinch-START so
transforms never drift — the object simply follows the hand offset from
where the pinch began.

  Translation : screen delta → world delta using object's Z depth
  Scale       : distance ratio (current / start)
  Rotation Z  : angle delta (current angle − start angle)
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from objects.selection import find_closest_object
from utils.geometry import angle_diff
from config import FOCAL_LENGTH, OBJ_MIN_SCALE, OBJ_MAX_SCALE, OBJ_MIN_Z, OBJ_RELEASE_DEBOUNCE


class RightHandController:

    def __init__(self, object_manager, dustbin) -> None:
        self._mgr     = object_manager
        self._dustbin = dustbin

        self._active = False
        self._selected_obj = None

        # Release debounce: require OBJ_RELEASE_DEBOUNCE consecutive non-pinch
        # frames before actually dropping the held object.  Prevents 1-2 frame
        # jitter from accidentally deselecting an object.
        self._release_count: int = 0

        # Captured at pinch start
        self._start_mid:       Optional[np.ndarray] = None
        self._start_dist:      float = 1.0
        self._start_angle:     float = 0.0
        self._start_obj_pos:   Optional[np.ndarray] = None
        self._start_obj_scale: Optional[np.ndarray] = None
        self._start_obj_rot:   Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def update(self, right_pinch, camera, scene_matrix) -> None:
        """Call every frame with the right-hand PinchState (or None)."""
        pinch_active = (right_pinch is not None and right_pinch.is_pinching)

        if not pinch_active:
            if self._active:
                # Release debounce: require OBJ_RELEASE_DEBOUNCE consecutive
                # non-pinch frames before actually dropping the object.
                self._release_count += 1
                if self._release_count >= OBJ_RELEASE_DEBOUNCE:
                    self._on_pinch_release(right_pinch)
            return

        # Pinch is active — reset release counter
        self._release_count = 0

        if not self._active:
            self._on_pinch_start(right_pinch, camera, scene_matrix)
        else:
            self._on_pinch_hold(right_pinch)

    # ------------------------------------------------------------------

    def _on_pinch_start(self, pinch, camera, scene_matrix) -> None:
        self._active = True

        # Try to select the nearest object
        obj = find_closest_object(
            self._mgr.objects, pinch.midpoint, camera, scene_matrix
        )
        self._mgr.select(obj)
        self._selected_obj = obj

        if obj is not None:
            self._start_mid       = np.array(pinch.midpoint, dtype=float)
            self._start_dist      = max(pinch.distance_px, 1.0)
            self._start_angle     = pinch.angle
            self._start_obj_pos   = obj.transform.position.copy()
            self._start_obj_scale = obj.transform.scale.copy()
            self._start_obj_rot   = obj.transform.rotation.copy()

    def _on_pinch_hold(self, pinch) -> None:
        if self._selected_obj is None:
            return

        mid = pinch.midpoint
        self._dustbin.highlight = self._dustbin.is_over(mid)

        if self._dustbin.highlight:
            return  # preview delete — don't move while over bin

        self._manipulate(pinch)

    def _on_pinch_release(self, pinch) -> None:
        self._active        = False
        self._release_count = 0

        if self._selected_obj is not None and self._dustbin.highlight:
            self._mgr.remove(self._selected_obj)

        self._dustbin.highlight = False
        self._mgr.select(None)
        self._selected_obj = None
        self._start_mid    = None

    # ------------------------------------------------------------------
    # Public state queries (used for mode arbitration in main.py)
    # ------------------------------------------------------------------

    @property
    def is_manipulating(self) -> bool:
        """True while a pinch is active AND an object is held.

        Used by main.py to suppress scene control so the two transform
        systems never run simultaneously.
        """
        return self._active and self._selected_obj is not None

    def _manipulate(self, pinch) -> None:
        obj   = self._selected_obj
        obj_z = max(float(obj.transform.position[2]), OBJ_MIN_Z)

        # ── Translation ──────────────────────────────────────────────
        current_mid = np.array(pinch.midpoint, dtype=float)
        delta_px    = current_mid - self._start_mid
        # Convert screen-pixel delta to world-unit delta at obj's Z depth
        dx_w =  delta_px[0] * obj_z / FOCAL_LENGTH
        dy_w = -delta_px[1] * obj_z / FOCAL_LENGTH   # flip Y
        obj.transform.position = self._start_obj_pos + np.array([dx_w, dy_w, 0.0])
        # Prevent pushing behind camera
        obj.transform.position[2] = max(obj.transform.position[2], OBJ_MIN_Z)

        # ── Scale ────────────────────────────────────────────────────
        dist_ratio = pinch.distance_px / self._start_dist
        new_scale  = np.clip(
            self._start_obj_scale * dist_ratio,
            OBJ_MIN_SCALE, OBJ_MAX_SCALE,
        )
        obj.transform.scale = new_scale

        # ── Rotation Z (pinch twist) ──────────────────────────────────
        delta_angle = angle_diff(pinch.angle, self._start_angle)
        obj.transform.rotation    = self._start_obj_rot.copy()
        obj.transform.rotation[2] = self._start_obj_rot[2] + delta_angle


