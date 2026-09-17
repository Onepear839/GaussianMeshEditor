"""World-space axis-aligned bounding boxes for Gaussian and Mesh objects.

Both KIRI Gaussian representations are supported:
- EMPTY proxy carrying 'gaussian_data' bytes (columns 0..2 = positions)
- MESH whose vertices are the splat centers (obj.bound_box)

All results are in world space: local bounds are transformed by
matrix_world, matching how KIRI renders (positions @ matrix_world).
"""

import numpy as np
from mathutils import Vector


def _corners_to_world_aabb(corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    bmin = Vector((min(xs), min(ys), min(zs)))
    bmax = Vector((max(xs), max(ys), max(zs)))
    return bmin, bmax


def _local_to_world(local, matrix):
    """Transform an (N, 3) array of local positions by a 4x4 matrix."""
    rot = np.array(matrix.to_3x3(), dtype=np.float64)
    trans = np.array(matrix.to_translation(), dtype=np.float64)
    return local @ rot.T + trans


def gaussian_world_points(obj):
    """Return an (N, 3) array of world-space splat positions, or None.

    Both KIRI representations are covered:
    - EMPTY proxy carrying 'gaussian_data' bytes (columns 0..2)
    - MESH whose vertices are the splat centers
    """
    if obj is None:
        return None
    data = obj.get("gaussian_data")
    if data is not None:
        count = int(obj.get("gaussian_count", 0))
        # Guard against corrupt byte properties: np.frombuffer raises
        # ValueError when len(data) is not a multiple of 4, and Blender 5.2
        # can truncate large IDProperty bytes on overwrite.
        if count > 0 and isinstance(data, (bytes, bytearray)) and len(data) == count * 59 * 4:
            arr = np.frombuffer(data, dtype=np.float32).reshape(count, 59)
            local = arr[:, :3].astype(np.float64)
            return _local_to_world(local, obj.matrix_world)
    if obj.type == "MESH":
        local = np.array([list(v.co) for v in obj.data.vertices], dtype=np.float64)
        if local.size == 0:
            return None
        return _local_to_world(local, obj.matrix_world)
    return None


def gaussian_world_bounds(obj):
    """Return (min, max) world-space corners, or None if not a Gaussian."""
    pts = gaussian_world_points(obj)
    if pts is None or len(pts) == 0:
        return None
    bmin = Vector(pts.min(axis=0))
    bmax = Vector(pts.max(axis=0))
    return bmin, bmax


def mesh_world_bounds(obj):
    """Return (min, max) world-space corners of a MESH object, or None."""
    if obj is None or obj.type != "MESH":
        return None
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return _corners_to_world_aabb(corners)


def bounds_center_size(bmin, bmax):
    """center = (min + max) / 2, size = max - min (component-wise)."""
    center = (bmin + bmax) / 2.0
    size = bmax - bmin
    return center, size
