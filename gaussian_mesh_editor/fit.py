"""Phase 7: fit proxy vertices onto the Gaussian cloud.

The whole loop runs in world space (so a manually rotated or
non-uniformly scaled proxy still behaves), then local coordinates are
written back through the inverse object matrix.

Two target rules:
- NEAREST: pull each vertex toward the nearest splat center.
- PLANE: fit a tangent plane through the k nearest splats and slide the
  vertex onto it (along its own normal, with an orthogonal fallback).

A light Laplacian smoothing between rounds keeps the mesh from
crumpling while vertices are being pulled.
"""

import mathutils
import numpy as np
from mathutils import Vector

from . import bbox

_NEAREST = "NEAREST"


def gaussian_kdtree(g_obj):
    """mathutils KDTree over the Gaussian's world-space splat positions."""
    pts = bbox.gaussian_world_points(g_obj)
    if pts is None or len(pts) == 0:
        return None
    kd = mathutils.kdtree.KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(Vector(p), i)
    kd.balance()
    return kd


def _nearest_target(v, kd):
    co, _idx, _dist = kd.find(Vector(v))
    return co


def _plane_target(v, n_v, kd, k):
    found = kd.find_n(v, k)
    if not found:
        return None
    if len(found) == 1:
        return found[0][0]
    pts = np.array([co for co, _idx, _dist in found], dtype=np.float64)
    centroid = pts.mean(axis=0)
    cov = (pts - centroid).T @ (pts - centroid)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_p = eigvecs[:, 0]  # smallest-variance direction = plane normal
    a = np.asarray(v, dtype=np.float64)
    denom = float(np.dot(n_v, n_p))
    if abs(denom) < 1e-8:
        # Vertex normal is parallel to the plane: fall back to the
        # orthogonal projection (foot of the perpendicular).
        return Vector(centroid - np.dot(a - centroid, n_p) * n_p)
    t = float(np.dot(centroid - a, n_p)) / denom
    return Vector(a + t * n_v)


def _mean_nn_distance(kd, world):
    total = 0.0
    for v in world:
        _co, _idx, dist = kd.find(Vector(v))
        total += dist
    return total / len(world)


def _laplacian_blend(me, world, weight):
    """Blend every vertex with the average of its topological neighbors."""
    adj = [[] for _ in range(len(me.vertices))]
    for e in me.edges:
        a, b = e.vertices
        adj[a].append(b)
        adj[b].append(a)
    result = np.array(world, dtype=np.float64, copy=True)
    for i, nb in enumerate(adj):
        if not nb:
            continue
        avg = np.mean(world[nb], axis=0)
        result[i] = world[i] * (1.0 - weight) + avg * weight
    return result


def _world_normal(mat, local_normal):
    n = mat @ Vector((local_normal[0], local_normal[1], local_normal[2], 0.0))
    n = Vector((n[0], n[1], n[2]))
    if n.length_squared > 1e-12:
        n.normalize()
    return np.asarray(n, dtype=np.float64)


def fit_mesh(obj, g_obj, iterations, strength, mode, k, smooth):
    """Snap + smooth loop. Returns (mean_nn_before, mean_nn_after)."""
    kd = gaussian_kdtree(g_obj)
    if kd is None:
        raise ValueError("No Gaussian positions available.")
    mat = obj.matrix_world
    mat_inv = mat.inverted()

    me = obj.data
    verts = me.vertices
    if len(verts) == 0:
        return 0.0, 0.0

    world = np.array([list(mat @ Vector(v.co)) for v in verts], dtype=np.float64)
    normals = (
        np.array([_world_normal(mat, v.normal) for v in verts], dtype=np.float64)
        if mode != _NEAREST
        else None
    )

    before = _mean_nn_distance(kd, world)
    for _ in range(iterations):
        targets = np.array(world, dtype=np.float64, copy=True)
        for i, v in enumerate(world):
            if normals is not None:
                t = _plane_target(v, normals[i], kd, k)
            else:
                t = _nearest_target(v, kd)
            if t is not None:
                targets[i] = t
        moved = world + strength * (targets - world)
        if smooth > 0.0:
            moved = _laplacian_blend(me, moved, smooth)
        world = moved
    after = _mean_nn_distance(kd, world)

    for i, v in enumerate(verts):
        v.co = mat_inv @ Vector(world[i])
    me.update()
    return before, after
