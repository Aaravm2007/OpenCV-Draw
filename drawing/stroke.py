"""
Stroke data model.

A Stroke is a sequence of 2-D screen points with a colour and thickness.
Points are accumulated while the user is drawing and frozen when they stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Stroke:
    color:     Tuple[int, int, int]            # BGR
    thickness: int = 4
    points:    List[Tuple[int, int]] = field(default_factory=list)

    def add_point(self, pt: Tuple[int, int]) -> None:
        """Append a screen-space point."""
        # Skip duplicate points (hand didn't move) to keep the list lean.
        if self.points and self.points[-1] == pt:
            return
        self.points.append(pt)

    def is_drawable(self) -> bool:
        """A stroke needs at least 2 points to produce a visible line."""
        return len(self.points) >= 2
