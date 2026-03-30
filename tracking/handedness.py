"""
Handedness helpers.

MediaPipe reports handedness from the perspective of the *camera*, not the
person.  When FLIP_HORIZONTAL is True the image is mirrored, so the labels
come out as the user experiences them (left hand = left side of screen).
"""

from utils.constants import HAND_LEFT, HAND_RIGHT


def get_handedness_label(handedness_result, idx: int) -> str:
    """
    Return 'Left' or 'Right' for the hand at position idx in the result list.

    Args:
        handedness_result: multi_handedness from a MediaPipe Hands result.
        idx:               Index into the result (matches multi_hand_landmarks).
    """
    return handedness_result[idx].classification[0].label


def get_handedness_score(handedness_result, idx: int) -> float:
    """Return the classification confidence [0, 1] for the hand at idx."""
    return handedness_result[idx].classification[0].score


def is_left(label: str) -> bool:
    return label == HAND_LEFT


def is_right(label: str) -> bool:
    return label == HAND_RIGHT
