"""
Application state machine.

States
------
IDLE
    No hands detected. The system is dormant.

WAITING_FOR_BOTH_PALMS
    At least one hand is visible but the activation criteria are not yet met.
    (Used in Phase 3 — kept as a defined state from the start.)

USER_LOCKED
    Both open palms were detected stably; the user is now locked in.
    The system is ready to accept interaction.

ACTIVE_TRACKING
    The session is live. Gestures, drawing, and object manipulation are active.

USER_LOST
    The locked user's hands disappeared for longer than the timeout.
    The system is about to return to IDLE.
"""

from enum import Enum, auto
from typing import Optional


class AppState(Enum):
    IDLE = auto()
    WAITING_FOR_BOTH_PALMS = auto()
    USER_LOCKED = auto()
    ACTIVE_TRACKING = auto()
    USER_LOST = auto()


# Valid forward transitions. None means "any target is allowed".
_ALLOWED_TRANSITIONS: dict[AppState, set[AppState]] = {
    AppState.IDLE:                  {AppState.WAITING_FOR_BOTH_PALMS},
    AppState.WAITING_FOR_BOTH_PALMS:{AppState.IDLE, AppState.ACTIVE_TRACKING},
    AppState.ACTIVE_TRACKING:       {AppState.USER_LOST, AppState.IDLE},
    AppState.USER_LOST:             {AppState.IDLE, AppState.ACTIVE_TRACKING},
    # USER_LOCKED is reserved for a future identity-lock upgrade; unused for now.
    AppState.USER_LOCKED:           {AppState.ACTIVE_TRACKING, AppState.IDLE},
}


class StateManager:
    """
    Owns the current AppState and enforces valid transitions.
    Other modules call transition_to() — they never write _state directly.
    """

    def __init__(self) -> None:
        self._state: AppState = AppState.IDLE
        self._prev_state: Optional[AppState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def previous_state(self) -> Optional[AppState]:
        return self._prev_state

    def transition_to(self, new_state: AppState) -> bool:
        """
        Attempt to transition to new_state.

        Returns True if the transition happened, False if it was rejected
        (same state or invalid target).
        """
        if new_state == self._state:
            return False

        allowed = _ALLOWED_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            # Log a warning but do not raise — we prefer robustness at runtime.
            print(
                f"[StateManager] Rejected transition "
                f"{self._state.name} → {new_state.name}"
            )
            return False

        self._prev_state = self._state
        self._state = new_state
        return True

    def is_active(self) -> bool:
        """True when the system is locked onto a user and processing gestures."""
        return self._state in (AppState.USER_LOCKED, AppState.ACTIVE_TRACKING)

    def __str__(self) -> str:
        return self._state.name

    def __repr__(self) -> str:
        return f"StateManager(state={self._state.name})"
