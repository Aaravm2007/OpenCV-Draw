"""
Gesture 3D Draw — entry point.

Run from inside the gesture_3d_draw/ directory:
    python main.py

Key bindings:
    q  — quit
    d  — toggle debug overlay
    c  — clear canvas strokes
    r  — reset scene transform
    n  — spawn a new cube (random colour / pose)
    t  — show tutorial / controls guide
    s  — save scene to disk
    l  — load scene from disk
"""

import math
import random
import sys
import cv2
import numpy as np

from utils.serialization import save_scene, load_scene

from config import (
    WINDOW_NAME, FLIP_HORIZONTAL,
    USER_LOST_IDLE_FRAMES, FOCAL_LENGTH,
    PINCH_GRACE_FRAMES,
)
import config
from core.camera import CameraCapture
from core.state_manager import StateManager, AppState
from core.session_lock import SessionLock
from tracking.hand_tracker import HandTracker
from tracking.palm_activation import PalmActivationDetector
from gestures.finger_state import detect_finger_state
from gestures.pinch import detect_pinch
from gestures.gesture_classifier import GestureClassifier, Gesture
from gestures.gesture_memory import GestureMemory
from drawing.canvas_manager import CanvasManager
from drawing.color_picker import ColorPicker
from objects.object_manager import ObjectManager
from objects.wireframe_shapes import create_cube, create_plane, create_prism
from controls.scene_controller import SceneController
from controls.right_hand_controller import RightHandController
from controls.left_hand_controller import LeftHandController
from render.renderer import Renderer
from render.projection import PerspectiveCamera
from render.dustbin import Dustbin
from utils.constants import INDEX_TIP


def _spawn_default_objects(manager: ObjectManager) -> None:
    manager.add(create_cube(
        color=(180, 180, 255),
        pos=(-2.0, 0.2, 6.0),
        rot=(0.35, 0.55, 0.1),
        scl=1.4,
    ))
    manager.add(create_prism(
        color=(100, 210, 255),
        pos=(2.0, 0.2, 6.0),
        rot=(0.2, -0.4, 0.0),
        scl=1.4,
    ))
    manager.add(create_plane(
        color=(80, 160, 80),
        pos=(0.0, -1.2, 6.0),
        rot=(0.0, 0.0, 0.0),
        scl=(6.0, 1.0, 4.0),
    ))


def main() -> None:
    # ------------------------------------------------------------------
    # Initialise subsystems
    # ------------------------------------------------------------------
    try:
        camera = CameraCapture()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    W, H = camera.width, camera.height

    state_manager  = StateManager()
    hand_tracker   = HandTracker()
    session_lock   = SessionLock()
    palm_activator = PalmActivationDetector()
    renderer       = Renderer(hand_tracker)
    classifier     = GestureClassifier()
    canvas         = CanvasManager()
    color_picker   = ColorPicker(frame_width=W)
    object_manager = ObjectManager()
    dustbin        = Dustbin(frame_width=W, frame_height=H)
    scene          = SceneController()
    cam3d          = PerspectiveCamera(focal_length=FOCAL_LENGTH, cx=W/2, cy=H/2)

    right_ctrl = RightHandController(object_manager, dustbin)
    left_ctrl  = LeftHandController(scene)

    # _spawn_default_objects(object_manager)

    left_memory  = GestureMemory()
    right_memory = GestureMemory()

    # Pinch hysteresis state (per-hand previous frame)
    left_pinching_prev  = False
    right_pinching_prev = False
    # Grace-period counters: when a hand disappears briefly, keep its
    # prev_pinching state alive for PINCH_GRACE_FRAMES frames so the strict
    # ON threshold isn't applied on immediate re-appearance.
    left_pinch_grace    = 0
    right_pinch_grace   = 0

    user_lost_idle_count = 0
    active_mode  = "IDLE"   # current interaction mode (for debug + arbitration)
    _flash_text  = ""
    _flash_timer = 0   # frames remaining to show flash

    print(f"=== {WINDOW_NAME} ===")
    print(f"Camera: {camera}")
    print("Keys: q=quit  d=debug  c=clear  r=reset scene  n=new cube")
    print("      t=tutorial  s=save  l=load")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W, H)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[ERROR] Camera read failed.")
                break

            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            tracking_result = hand_tracker.process(frame_rgb, W, H)
            frame_rgb.flags.writeable = True

            # Render on a black canvas instead of the video feed
            frame = np.zeros((H, W, 3), dtype=np.uint8)

            hands_present = len(tracking_result.hands) > 0

            # ----------------------------------------------------------
            # Gesture processing (always, regardless of state)
            # ----------------------------------------------------------
            left_finger = left_pinch = None
            right_finger = right_pinch = None
            left_gesture  = Gesture.UNKNOWN
            right_gesture = Gesture.UNKNOWN

            left_raw  = Gesture.UNKNOWN
            right_raw = Gesture.UNKNOWN

            if tracking_result.left_hand:
                lm = tracking_result.left_hand.smoothed_landmarks
                left_finger = detect_finger_state(lm, "Left")
                left_pinch  = detect_pinch(lm, left_pinching_prev)
                left_pinching_prev = left_pinch.is_pinching
                left_pinch_grace   = PINCH_GRACE_FRAMES   # replenish
                left_raw     = classifier.classify(left_finger, left_pinch)
                left_gesture = left_memory.update(left_raw)
            else:
                left_memory.reset()
                # Grace period: keep prev_pinching alive for a few frames so
                # the strict ON-threshold isn't applied on immediate re-entry
                if left_pinch_grace > 0:
                    left_pinch_grace -= 1
                else:
                    left_pinching_prev = False

            if tracking_result.right_hand:
                lm = tracking_result.right_hand.smoothed_landmarks
                right_finger = detect_finger_state(lm, "Right")
                right_pinch  = detect_pinch(lm, right_pinching_prev)
                right_pinching_prev = right_pinch.is_pinching
                right_pinch_grace   = PINCH_GRACE_FRAMES  # replenish
                right_raw     = classifier.classify(right_finger, right_pinch)
                right_gesture = right_memory.update(right_raw)
            else:
                right_memory.reset()
                if right_pinch_grace > 0:
                    right_pinch_grace -= 1
                else:
                    right_pinching_prev = False
                if canvas.is_drawing:
                    canvas.end_stroke()

            # ----------------------------------------------------------
            # State machine
            # ----------------------------------------------------------
            state = state_manager.state
            activation_progress = 0.0

            if state == AppState.IDLE:
                palm_activator.reset()
                if hands_present:
                    state_manager.transition_to(AppState.WAITING_FOR_BOTH_PALMS)

            elif state == AppState.WAITING_FOR_BOTH_PALMS:
                if not hands_present:
                    palm_activator.reset()
                    state_manager.transition_to(AppState.IDLE)
                else:
                    activation_progress = palm_activator.progress
                    # Use raw (pre-debounce) gestures so the progress bar
                    # responds immediately as the user opens both palms,
                    # rather than waiting an extra ~4 debounce frames.
                    if palm_activator.update(left_raw, right_raw):
                        session_lock.lock(tracking_result)
                        state_manager.transition_to(AppState.ACTIVE_TRACKING)
                        renderer.activate_tutorial()
                        print("[INFO] Session activated.")

            elif state == AppState.ACTIVE_TRACKING:
                user_lost_idle_count = 0
                if session_lock.update(tracking_result):
                    canvas.end_stroke()
                    session_lock.unlock()
                    state_manager.transition_to(AppState.USER_LOST)
                    print("[INFO] User lost.")

            elif state == AppState.USER_LOST:
                if hands_present:
                    session_lock.lock(tracking_result)
                    state_manager.transition_to(AppState.ACTIVE_TRACKING)
                    user_lost_idle_count = 0
                    print("[INFO] User returned.")
                else:
                    user_lost_idle_count += 1
                    if user_lost_idle_count >= USER_LOST_IDLE_FRAMES:
                        session_lock.unlock()           # clean up lock state
                        state_manager.transition_to(AppState.IDLE)
                        left_memory.reset()
                        right_memory.reset()
                        palm_activator.reset()
                        user_lost_idle_count = 0
                        print("[INFO] Returned to IDLE.")

            # ----------------------------------------------------------
            # Active-session interaction
            # ----------------------------------------------------------
            scene_matrix = scene.get_matrix()

            if state_manager.state == AppState.ACTIVE_TRACKING:

                # ── Right-hand tip position (used by drawing + picker) ─
                right_hand = tracking_result.right_hand
                right_tip  = None
                if right_hand:
                    tip       = right_hand.smoothed_landmarks[INDEX_TIP]
                    right_tip = (tip[0], tip[1])

                is_index_only = (right_gesture == Gesture.INDEX_ONLY)

                # ── Determine if right hand is pinching ───────────────
                right_pinch_gated = (
                    right_pinch if right_gesture == Gesture.PINCH else None
                )

                # ── Step 1: Object manipulation (right hand, PINCH) ───
                # Must run BEFORE scene control so is_manipulating reflects
                # the correct state for this frame's arbitration.
                right_ctrl.update(right_pinch_gated, cam3d, scene_matrix)

                # ── Step 2: Scene control (left hand, PINCH) ──────────
                # SUPPRESSED while an object is being actively dragged.
                # This prevents the scene from moving while you hold an object,
                # which would break the object's screen-space tracking.
                obj_being_dragged = right_ctrl.is_manipulating
                left_pinch_gated  = (
                    left_pinch
                    if (left_gesture == Gesture.PINCH and not obj_being_dragged)
                    else None
                )
                left_ctrl.update(left_pinch_gated)
                scene_matrix = scene.get_matrix()   # refresh after potential update

                # ── Step 3: Colour picker (index-only, not dragging) ──
                # Do NOT advance the dwell counter while an object is held —
                # prevents accidental colour changes during manipulation.
                color_changed = color_picker.update(
                    right_tip,
                    is_index_only and not obj_being_dragged,
                )
                over_picker = (
                    right_tip is not None and
                    color_picker.is_over_picker(right_tip)
                )

                # ── Step 4: Drawing (index-only, lowest priority) ─────
                # Active only when: index up AND not over picker AND
                # no color just changed AND no object selected/held.
                obj_selected = object_manager.selected is not None
                can_draw = (
                    is_index_only
                    and not over_picker
                    and not color_changed
                    and not obj_selected
                    and not obj_being_dragged
                )

                if can_draw and right_tip is not None:
                    tip_px = (int(right_tip[0]), int(right_tip[1]))
                    if canvas.is_drawing:
                        canvas.update_stroke(tip_px)
                    else:
                        canvas.start_stroke(tip_px, color_picker.active_color)
                else:
                    if canvas.is_drawing:
                        canvas.end_stroke()

                # ── Compute active mode for debug display ─────────────
                if obj_being_dragged:
                    active_mode = "OBJECT_MANIPULATE"
                elif left_pinch_gated is not None and left_pinch_gated.is_pinching:
                    active_mode = "SCENE_CONTROL"
                elif over_picker:
                    active_mode = "COLOR_PICK"
                elif can_draw:
                    active_mode = "DRAW"
                else:
                    active_mode = "IDLE"

            else:
                # Outside active session — release everything gracefully
                if canvas.is_drawing:
                    canvas.end_stroke()
                right_ctrl.update(None, cam3d, scene_matrix)
                left_ctrl.update(None)
                active_mode = "IDLE"

            # ----------------------------------------------------------
            # Flash timer tick
            # ----------------------------------------------------------
            if _flash_timer > 0:
                _flash_timer -= 1
            else:
                _flash_text = ""

            # ----------------------------------------------------------
            # Render
            # ----------------------------------------------------------
            renderer.render(
                frame, state_manager, tracking_result,
                left_gesture=left_gesture,
                right_gesture=right_gesture,
                left_finger=left_finger,
                right_finger=right_finger,
                left_pinch=left_pinch,
                right_pinch=right_pinch,
                canvas=canvas,
                color_picker=color_picker,
                object_manager=object_manager,
                camera=cam3d,
                scene_matrix=scene_matrix,
                dustbin=dustbin,
                scene_controller=scene,
                activation_progress=activation_progress,
                flash_text=_flash_text,
                session_lock=session_lock,
                active_mode=active_mode,
            )

            cv2.imshow(WINDOW_NAME, frame)

            # ----------------------------------------------------------
            # Key bindings
            # ----------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                config.DEBUG_MODE = not config.DEBUG_MODE
                print(f"[INFO] Debug {'ON' if config.DEBUG_MODE else 'OFF'}")
            elif key == ord("c"):
                canvas.clear()
                print("[INFO] Canvas cleared.")
            elif key == ord("r"):
                scene.reset()
                print("[INFO] Scene transform reset.")
            elif key == ord("n"):
                object_manager.add(create_cube(
                    color=(
                        random.randint(80, 255),
                        random.randint(80, 255),
                        random.randint(80, 255),
                    ),
                    pos=(random.uniform(-2, 2), random.uniform(-0.5, 1), 6.0),
                    rot=(random.uniform(0, math.pi), random.uniform(0, math.pi), 0),
                    scl=1.2,
                ))
                print(f"[INFO] Spawned cube. Total: {len(object_manager.objects)}")
            elif key == ord("t"):
                renderer.activate_tutorial()
                print("[INFO] Tutorial overlay triggered.")
            elif key == ord("s"):
                try:
                    path = save_scene(object_manager, canvas)
                    msg  = f"Saved — {len(object_manager.objects)} obj, {canvas.stroke_count} strokes"
                    print(f"[INFO] {msg}  →  {path}")
                    _flash_text  = "Scene saved"
                    _flash_timer = 90
                except OSError as exc:
                    print(f"[ERROR] Save failed: {exc}")
                    _flash_text  = "Save FAILED"
                    _flash_timer = 90
            elif key == ord("l"):
                try:
                    n_obj, n_strk = load_scene(object_manager, canvas)
                    msg = f"Loaded {n_obj} objects, {n_strk} strokes"
                    print(f"[INFO] {msg}")
                    _flash_text  = "Scene loaded"
                    _flash_timer = 90
                except FileNotFoundError:
                    print("[WARN] No save file found.")
                    _flash_text  = "No save file"
                    _flash_timer = 90
                except (ValueError, KeyError) as exc:
                    print(f"[ERROR] Load failed: {exc}")
                    _flash_text  = "Load FAILED"
                    _flash_timer = 90

    finally:
        camera.release()
        hand_tracker.close()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
