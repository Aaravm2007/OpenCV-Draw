"""
Camera capture wrapper.
Encapsulates OpenCV VideoCapture so the rest of the app never touches it directly.
"""

import cv2
import numpy as np
from typing import Optional

from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class CameraCapture:
    """Opens a webcam and exposes a simple read() / release() interface."""

    def __init__(self) -> None:
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {CAMERA_INDEX}. "
                "Check that the webcam is connected and not in use."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """
        Capture and return the next BGR frame, or None on failure.
        The caller should treat None as a fatal error and exit the loop.
        """
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        self._cap.release()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    def __repr__(self) -> str:
        return (
            f"CameraCapture(index={CAMERA_INDEX}, "
            f"{self.width}×{self.height} @ {self.fps:.0f}fps)"
        )
