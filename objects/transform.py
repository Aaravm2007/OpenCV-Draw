"""
Per-object 3-D transform.

Stores position, rotation (Euler XYZ in radians), and scale as NumPy arrays
and computes the combined 4×4 homogeneous matrix on demand.

Convention: T * Rz * Ry * Rx * S  (scale first, then rotate, then translate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from utils.math3d import (
    translation, rotation_x, rotation_y, rotation_z, scale_matrix, identity
)


@dataclass
class Transform:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    scale:    np.ndarray = field(default_factory=lambda: np.ones(3,  dtype=float))

    # ------------------------------------------------------------------

    def get_matrix(self) -> np.ndarray:
        """Return the combined 4×4 model matrix."""
        S  = scale_matrix(*self.scale)
        Rx = rotation_x(self.rotation[0])
        Ry = rotation_y(self.rotation[1])
        Rz = rotation_z(self.rotation[2])
        T  = translation(*self.position)
        return T @ Rz @ Ry @ Rx @ S

    # ------------------------------------------------------------------
    # Convenience mutators (return self for chaining)

    def translate(self, dx: float, dy: float, dz: float) -> "Transform":
        self.position += np.array([dx, dy, dz], dtype=float)
        return self

    def rotate(self, drx: float, dry: float, drz: float) -> "Transform":
        self.rotation += np.array([drx, dry, drz], dtype=float)
        return self

    def scale_by(self, factor: float) -> "Transform":
        self.scale *= factor
        return self

    # ------------------------------------------------------------------

    @staticmethod
    def make(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0),
        scl=(1.0, 1.0, 1.0),
    ) -> "Transform":
        return Transform(
            position=np.array(pos, dtype=float),
            rotation=np.array(rot, dtype=float),
            scale=np.array(scl, dtype=float),
        )
