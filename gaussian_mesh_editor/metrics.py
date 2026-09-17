"""Alignment metrics.

Phase 2 ships Center Distance and Size Ratio. Surface / Chamfer /
Hausdorff distance slots are added here in a later research phase.
"""

from mathutils import Vector


def center_distance(gaussian_center, mesh_center):
    """L2 distance between the two world-space centers."""
    return (Vector(gaussian_center) - Vector(mesh_center)).length


def size_ratio(gaussian_size, mesh_size):
    """Uniform scale factor mapping mesh max-dimension onto gaussian's."""
    g = max(gaussian_size)
    m = max(mesh_size)
    if m <= 0.0:
        return 1.0
    return g / m
