"""
Central configuration file. All tuneable constants live here.
Do not import from project modules — this file has no dependencies.

Tuning guide
------------
Jittery tracking    → lower SMOOTHING_ALPHA (e.g. 0.35)
Laggy tracking      → raise  SMOOTHING_ALPHA (e.g. 0.65)
Gestures misfire    → raise  GESTURE_DEBOUNCE_FRAMES
Gestures feel slow  → lower  GESTURE_DEBOUNCE_FRAMES
Pinch too hard      → lower  PINCH_ON_THRESHOLD
Pinch too easy      → raise  PINCH_ON_THRESHOLD
Objects hard to grab→ raise  OBJ_SELECT_THRESHOLD_PX
Scene pans too fast → lower  SCENE_PAN_SENSITIVITY
"""

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
CAMERA_FPS: int = 30

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
WINDOW_NAME: str = "Gesture 3D Draw"
FLIP_HORIZONTAL: bool = True   # Mirror so the view feels natural

# ---------------------------------------------------------------------------
# MediaPipe Hands
# ---------------------------------------------------------------------------
MP_MAX_HANDS: int = 2
MP_MIN_DETECTION_CONFIDENCE: float = 0.70
MP_MIN_TRACKING_CONFIDENCE:  float = 0.60
MP_MODEL_COMPLEXITY: int = 1   # 0 = lite (faster), 1 = full (more accurate)

# ---------------------------------------------------------------------------
# Handedness stability
# ---------------------------------------------------------------------------
# A new handedness label must be seen for this many consecutive frames before
# the hand is re-routed to a different controller slot.
# Prevents single-frame label flips (hands crossing, partial occlusion, etc.)
HANDEDNESS_STABLE_FRAMES:   int   = 2
# Below this confidence the handedness label is considered unreliable.
# The hand is still tracked but will NOT cause a slot reassignment.
HANDEDNESS_MIN_CONFIDENCE:  float = 0.70

# ---------------------------------------------------------------------------
# Smoothing  (EMA: new = alpha*raw + (1-alpha)*prev)
# ---------------------------------------------------------------------------
SMOOTHING_ALPHA: float = 0.45  # Lower = smoother but laggier; higher = more responsive

# ---------------------------------------------------------------------------
# Pinch hysteresis grace period
# ---------------------------------------------------------------------------
# When a hand briefly disappears (1-PINCH_GRACE_FRAMES frames), keep its
# prev_pinching state alive rather than resetting to False.
# This prevents the strict ON-threshold from applying when hands re-appear
# mid-pinch, which would cause the pinch to drop momentarily.
PINCH_GRACE_FRAMES: int = 3

# ---------------------------------------------------------------------------
# Gesture detection
# ---------------------------------------------------------------------------
GESTURE_DEBOUNCE_FRAMES: int = 4    # Frames a gesture must hold before confirmed
                                     # Lower = more responsive; higher = more stable
PINCH_ON_THRESHOLD:  float = 0.15   # Normalised thumb-index dist to ENTER pinch
PINCH_OFF_THRESHOLD: float = 0.22   # Normalised dist to EXIT pinch (hysteresis gap)

# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------
ACTIVATION_STABLE_FRAMES: int   = 18    # Frames both open palms must be held
USER_LOST_TIMEOUT_FRAMES: int   = 45    # Frames without hands → USER_LOST
USER_LOST_IDLE_FRAMES:    int   = 30    # Frames in USER_LOST → IDLE
TUTORIAL_DISPLAY_SECONDS: float = 5.0   # Controls guide display duration

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
DRAW_STROKE_THICKNESS: int = 4
COLOR_PICKER_RADIUS:   int = 22     # Swatch circle radius
COLOR_PICKER_X:        int = 40     # Distance from right edge to swatch centre
COLOR_PICKER_Y_START:  int = 30     # Y of first swatch
COLOR_PICKER_SPACING:  int = 58     # Vertical spacing between swatches
COLOR_HOVER_TOLERANCE: int = 10     # Extra px added to swatch hit-test radius
DRAW_COLORS: list = [
    (255, 255, 255),   # White
    (50,   50, 255),   # Red
    (50,  220,  50),   # Green
    (255, 100,  50),   # Blue
    (0,   220, 220),   # Yellow
    (220,   0, 220),   # Magenta
    (0,   165, 255),   # Orange
]

# ---------------------------------------------------------------------------
# 3-D rendering
# ---------------------------------------------------------------------------
FOCAL_LENGTH:            float = 800.0  # Perspective focal length in pixels
                                         # ~640 = wide, ~1000 = narrow FOV
WIREFRAME_THICKNESS:     int   = 2
WIREFRAME_SELECTED_COLOR       = (0, 255, 255)
WIREFRAME_DEPTH_DIM:     float = 0.45   # 0 = flat, 1 = black at far end
SCENE_Z_NEAR:            float = 0.5    # Near clip distance

# ---------------------------------------------------------------------------
# Object interaction
# ---------------------------------------------------------------------------
OBJ_SELECT_THRESHOLD_PX: int   = 130   # Screen-pixel radius for pinch-to-select
OBJ_MIN_SCALE:           float = 0.12
OBJ_MAX_SCALE:           float = 9.0
OBJ_MIN_Z:               float = 1.5   # Prevents objects going behind camera
# Frames of non-pinch required before confirming an object release.
# Prevents 1-2 frame jitter from accidentally dropping the held object.
OBJ_RELEASE_DEBOUNCE:    int   = 2

# ---------------------------------------------------------------------------
# Scene control
# ---------------------------------------------------------------------------
SCENE_PAN_SENSITIVITY: float = 1.0
SCENE_ZOOM_MIN:        float = 0.15
SCENE_ZOOM_MAX:        float = 6.0
SCENE_CENTER_Z:        float = 5.5    # Rotation pivot depth

# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------
DEBUG_MODE: bool = True
DEBUG_FONT_SCALE:  float = 0.55
DEBUG_THICKNESS:   int   = 1
DEBUG_LINE_HEIGHT: int   = 22

# ---------------------------------------------------------------------------
# Mode banner (always-visible status line, independent of DEBUG_MODE)
# ---------------------------------------------------------------------------
MODE_BANNER_ENABLED: bool = True

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
COLOR_DEBUG_TEXT   = (0, 255, 0)
COLOR_DEBUG_SHADOW = (0, 0, 0)
COLOR_LEFT_HAND    = (255, 100, 100)
COLOR_RIGHT_HAND   = (100, 100, 255)

# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
SAVE_DIR:   str = "saves"
SAVE_FILE:  str = "scene.json"
