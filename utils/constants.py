"""
App-wide symbolic constants. No logic here — just names for magic numbers/strings.
Import these instead of using bare literals so meaning is always clear.
"""

# ---------------------------------------------------------------------------
# App states (mirrors AppState enum in core/state_manager.py)
# These strings are also used in the debug overlay.
# ---------------------------------------------------------------------------
STATE_IDLE = "IDLE"
STATE_WAITING_FOR_BOTH_PALMS = "WAITING_FOR_BOTH_PALMS"
STATE_USER_LOCKED = "USER_LOCKED"
STATE_ACTIVE_TRACKING = "ACTIVE_TRACKING"
STATE_USER_LOST = "USER_LOST"

# ---------------------------------------------------------------------------
# Handedness labels (as returned by MediaPipe)
# ---------------------------------------------------------------------------
HAND_LEFT = "Left"
HAND_RIGHT = "Right"

# ---------------------------------------------------------------------------
# MediaPipe hand landmark indices
# Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
# ---------------------------------------------------------------------------
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP  = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Convenience groups
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
FINGER_MCPS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
