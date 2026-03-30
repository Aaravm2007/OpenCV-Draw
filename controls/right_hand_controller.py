"""
Right-hand object interaction controller.

Gesture priority (checked in order each frame):
  1. Pinch over dustbin while object selected  → delete on release
  2. Pinch starts near a 3-D object            → select and drag it
  3. Pinch starts near a 2-D stroke            → drag the stroke
  4. Pinch held                                → move / scale / rotate
  5. Pinch released                            → deselect / finalise

Transform math
--------------
All deltas are computed relative to the state captured at pinch-START so
transforms never drift.

  3-D objects : screen delta → world delta using object's Z depth
  2-D strokes : screen delta / zoom → canonical-pixel delta
"""

from __future__ import annotations

from typing import Optional, Tuple
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

        # Stroke drag state
        self._selected_stroke      = None
        self._start_stroke_points  = None   # copy of stroke.points at drag start
        self._canvas               = None   # updated each frame, used in release

        # Release debounce: require OBJ_RELEASE_DEBOUNCE consecutive non-pinch
        # frames before actually dropping the held object.
        self._release_count: int = 0

        # Zoom captured at pinch start (needed for canonical-space stroke drag)
        self._zoom: float = 1.0

        # Captured at pinch start
        self._start_mid:       Optional[np.ndarray] = None
        self._start_dist:      float = 1.0
        self._start_angle:     float = 0.0
        self._start_obj_pos:   Optional[np.ndarray] = None
        self._start_obj_scale: Optional[np.ndarray] = None
        self._start_obj_rot:   Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def update(
        self,
        right_pinch,
        camera,
        scene_matrix,
        canvas=None,
        pan_px: Tuple[float, float] = (0.0, 0.0),
        zoom: float = 1.0,
        screen_center: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Call every frame with the right-hand PinchState (or None)."""
        pinch_active = (right_pinch is not None and right_pinch.is_pinching)

        if not pinch_active:
            if self._active:
                self._release_count += 1
                if self._release_count >= OBJ_RELEASE_DEBOUNCE:
                    self._on_pinch_release(right_pinch)
            return

        # Pinch is active — reset release counter and cache canvas ref
        self._release_count = 0
        self._canvas = canvas

        if not self._active:
            self._on_pinch_start(right_pinch, camera, scene_matrix,
                                 canvas, pan_px, zoom, screen_center)
        else:
            self._on_pinch_hold(right_pinch)

    # ------------------------------------------------------------------

    def _on_pinch_start(self, pinch, camera, scene_matrix,
                        canvas, pan_px, zoom, screen_center) -> None:
        self._active = True
        self._zoom   = zoom

        # ── Try 3-D objects first ──────────────────────────────────────
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
            return

        # ── Fall back to 2-D strokes ───────────────────────────────────
        if canvas is not None:
            cx, cy = screen_center
            stroke = canvas.find_nearest_stroke(
                pinch.midpoint, pan_px, zoom, cx, cy, threshold_px=40.0
            )
            self._selected_stroke = stroke
            if stroke is not None:
                self._start_mid          = np.array(pinch.midpoint, dtype=float)
                self._start_stroke_points = list(stroke.points)

    def _on_pinch_hold(self, pinch) -> None:
        mid = pinch.midpoint
        if self._selected_obj is not None:
            self._dustbin.highlight = self._dustbin.is_over(mid)
            if not self._dustbin.highlight:
                self._manipulate(pinch)

        elif self._selected_stroke is not None:
            self._dustbin.highlight = self._dustbin.is_over(mid)
            if not self._dustbin.highlight:
                self._drag_stroke(pinch)

    def _on_pinch_release(self, pinch) -> None:
        self._active        = False
        self._release_count = 0

        if self._dustbin.highlight:
            if self._selected_obj is not None:
                self._mgr.remove(self._selected_obj)
            elif self._selected_stroke is not None and self._canvas is not None:
                self._canvas.remove_stroke(self._selected_stroke)

        self._dustbin.highlight    = False
        self._mgr.select(None)
        self._selected_obj         = None
        self._selected_stroke      = None
        self._start_stroke_points  = None
        self._start_mid            = None

    # ------------------------------------------------------------------
    # Public state queries
    # ------------------------------------------------------------------

    @property
    def is_manipulating(self) -> bool:
        """True while pinch is active AND something is held (object or stroke)."""
        return self._active and (
            self._selected_obj is not None or
            self._selected_stroke is not None
        )

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _manipulate(self, pinch) -> None:
        obj   = self._selected_obj
        obj_z = max(float(obj.transform.position[2]), OBJ_MIN_Z)

        # Translation
        current_mid = np.array(pinch.midpoint, dtype=float)
        delta_px    = current_mid - self._start_mid
        dx_w =  delta_px[0] * obj_z / FOCAL_LENGTH
        dy_w = -delta_px[1] * obj_z / FOCAL_LENGTH
        obj.transform.position = self._start_obj_pos + np.array([dx_w, dy_w, 0.0])
        obj.transform.position[2] = max(obj.transform.position[2], OBJ_MIN_Z)

        # Scale
        dist_ratio = pinch.distance_px / self._start_dist
        new_scale  = np.clip(
            self._start_obj_scale * dist_ratio,
            OBJ_MIN_SCALE, OBJ_MAX_SCALE,
        )
        obj.transform.scale = new_scale

        # Rotation Z
        delta_angle = angle_diff(pinch.angle, self._start_angle)
        obj.transform.rotation    = self._start_obj_rot.copy()
        obj.transform.rotation[2] = self._start_obj_rot[2] + delta_angle

    def _drag_stroke(self, pinch) -> None:
        """Translate a 2-D stroke by the pinch delta in canonical pixel space."""
        delta_screen = np.array(pinch.midpoint, dtype=float) - self._start_mid
        # Strokes are stored in canonical space; screen delta / zoom gives
        # canonical delta (pan is a constant offset that cancels out in delta).
        zoom_safe = max(self._zoom, 0.01)
        dx = int(delta_screen[0] / zoom_safe)
        dy = int(delta_screen[1] / zoom_safe)
        stroke = self._selected_stroke
        stroke.points = [(x + dx, y + dy) for x, y in self._start_stroke_points]
