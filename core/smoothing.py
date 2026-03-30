"""
Temporal smoothing utilities.

ExponentialSmoother  — EMA for a single scalar or NumPy array.
LandmarkSmoother     — Per-landmark EMA for a full 21-point hand.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from config import SMOOTHING_ALPHA


class ExponentialSmoother:
    """
    Exponential Moving Average smoother.

    new_value = alpha * raw + (1 - alpha) * previous
    Higher alpha → tracks faster, less smoothing.
    Lower alpha  → more lag, more smoothing.
    """

    def __init__(self, alpha: float = SMOOTHING_ALPHA) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._value: Optional[np.ndarray] = None

    def update(self, raw: Union[float, List, Tuple, np.ndarray]) -> np.ndarray:
        """Feed a new raw sample; return the smoothed estimate."""
        raw_arr = np.array(raw, dtype=float)
        if self._value is None:
            self._value = raw_arr.copy()
        else:
            self._value = self.alpha * raw_arr + (1.0 - self.alpha) * self._value
        return self._value.copy()

    def reset(self) -> None:
        """Clear state — next update() will seed the smoother fresh."""
        self._value = None

    @property
    def value(self) -> Optional[np.ndarray]:
        return self._value if self._value is None else self._value.copy()

    def has_value(self) -> bool:
        return self._value is not None


class LandmarkSmoother:
    """
    Smooths a set of hand landmarks independently per landmark.

    Expects landmarks as a list of (x, y, z) tuples and returns a list
    of the same shape with smoothed coordinates.
    """

    def __init__(
        self,
        alpha: float = SMOOTHING_ALPHA,
        num_landmarks: int = 21,
    ) -> None:
        self.alpha = alpha
        self.num_landmarks = num_landmarks
        self._smoothers: List[ExponentialSmoother] = [
            ExponentialSmoother(alpha) for _ in range(num_landmarks)
        ]

    def update(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        """
        Accept raw landmarks, return smoothed landmarks.
        landmarks must have exactly num_landmarks entries.
        """
        if len(landmarks) != self.num_landmarks:
            raise ValueError(
                f"Expected {self.num_landmarks} landmarks, got {len(landmarks)}"
            )
        smoothed = []
        for smoother, lm in zip(self._smoothers, landmarks):
            arr = smoother.update(lm)
            smoothed.append((float(arr[0]), float(arr[1]), float(arr[2])))
        return smoothed

    def reset(self) -> None:
        """Reset all per-landmark smoothers (call when a hand re-appears)."""
        for s in self._smoothers:
            s.reset()
