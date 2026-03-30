"""
Scene serialisation — JSON save / load.

Format
------
{
  "version": 1,
  "objects": [
    {
      "type": "cube",          # cube | plane | prism
      "color": [B, G, R],
      "position": [x, y, z],
      "rotation": [rx, ry, rz],
      "scale": [sx, sy, sz]
    },
    ...
  ],
  "strokes": [
    {
      "color": [B, G, R],
      "thickness": 4,
      "points": [[x, y], ...]
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import os
from typing import List, TYPE_CHECKING

from config import SAVE_DIR, SAVE_FILE

if TYPE_CHECKING:
    from objects.object_manager import ObjectManager
    from drawing.canvas_manager import CanvasManager

_SAVE_VERSION = 1


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_scene(object_manager: "ObjectManager", canvas: "CanvasManager") -> str:
    """Serialise the current scene to JSON and write to disk.

    Returns the absolute path of the written file.
    Raises OSError on write failure.
    """
    data = {
        "version": _SAVE_VERSION,
        "objects": _serialise_objects(object_manager),
        "strokes": _serialise_strokes(canvas),
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, SAVE_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return os.path.abspath(path)


def _serialise_objects(manager: "ObjectManager") -> list:
    result = []
    for obj in manager.objects:
        result.append({
            "type":     obj.object_type,
            "color":    list(obj.color),
            "position": obj.transform.position.tolist(),
            "rotation": obj.transform.rotation.tolist(),
            "scale":    obj.transform.scale.tolist(),
        })
    return result


def _serialise_strokes(canvas: "CanvasManager") -> list:
    result = []
    for stroke in canvas.strokes:
        result.append({
            "color":     list(stroke.color),
            "thickness": stroke.thickness,
            "points":    [list(p) for p in stroke.points],
        })
    return result


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_scene(
    object_manager: "ObjectManager",
    canvas: "CanvasManager",
) -> tuple[int, int]:
    """Load a scene from disk into the provided managers.

    Clears existing objects and strokes before loading.
    Returns (n_objects_loaded, n_strokes_loaded).
    Raises FileNotFoundError if the save file does not exist.
    Raises ValueError on unsupported version or malformed data.
    """
    path = os.path.join(SAVE_DIR, SAVE_FILE)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    version = data.get("version", 0)
    if version != _SAVE_VERSION:
        raise ValueError(f"Unsupported save version: {version}")

    # Clear current state
    object_manager.clear()
    canvas.clear()

    n_obj = _deserialise_objects(data.get("objects", []), object_manager)
    n_strk = _deserialise_strokes(data.get("strokes", []), canvas)
    return n_obj, n_strk


def _deserialise_objects(raw: list, manager: "ObjectManager") -> int:
    from objects.wireframe_shapes import create_cube, create_plane, create_prism

    _factories = {
        "cube":  create_cube,
        "plane": create_plane,
        "prism": create_prism,
    }
    count = 0
    for item in raw:
        obj_type = item.get("type", "cube")
        factory  = _factories.get(obj_type, create_cube)
        color    = tuple(item["color"])
        pos      = tuple(item["position"])
        rot      = tuple(item["rotation"])
        scl_raw  = item["scale"]

        # Factories accept either a float or a 3-tuple for scale
        if obj_type == "plane":
            obj = factory(color=color, pos=pos, rot=rot, scl=tuple(scl_raw))
        else:
            # cube / prism: uniform scale stored as [sx, sx, sx]
            obj = factory(color=color, pos=pos, rot=rot, scl=scl_raw[0])

        # Restore exact scale (may differ from uniform if edited)
        import numpy as np
        obj.transform.scale = np.array(scl_raw, dtype=float)

        manager.add(obj)
        count += 1
    return count


def _deserialise_strokes(raw: list, canvas: "CanvasManager") -> int:
    from drawing.stroke import Stroke

    count = 0
    for item in raw:
        stroke = Stroke(
            color=tuple(item["color"]),
            thickness=item.get("thickness", 4),
        )
        for pt in item.get("points", []):
            stroke.points.append(tuple(pt))
        if stroke.is_drawable():
            canvas.add_stroke(stroke)
            count += 1
    return count
