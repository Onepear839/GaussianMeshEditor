"""Step 3-4: camera rays -> BBX corners -> proxy mesh intersection.

Paper: 'we cast rays from these cameras toward the center of a Gaussian'
(UniMGS*) and 'each camera casts 8 rays toward the corners of a
Gaussian's BBX ... For each ray, we retain the face closest to the
Gaussian' (UniMGS). We implement the BBX version: every corner of every
Gaussian gets one ray from every camera; the best (closest to the
Gaussian center) hit across cameras is kept.

Design:
  - The camera set is a CameraSource protocol so real training cameras
    (COLMAP) can replace Blender cameras later without touching the
    binding logic.
  - Intersection goes through a caster protocol. In Blender the caster
    wraps mathutils.bvhtree.BVHTree (O(log M) per ray). A pure-NumPy
    Moller-Trumbore caster doubles as the ground-truth oracle for tests.
  - mathutils is imported lazily so this module stays importable (and
    unit-testable) outside Blender.
"""

import numpy as np

_EPS = 1e-12


class CameraSource:
    """Protocol: yield camera ray origins for the current scene.

    Subclasses implement origins(scene) -> (C, 3) float64 array.
    Rays are cast from each origin toward each BBX corner; for a COLMAP
    source only the camera centers are needed as well.
    """

    name = "CameraSource"

    def origins(self, scene):
        raise NotImplementedError


def ray_directions(corners, origins):
    """(N, 8, 3) corners, (C, 3) origins -> (N, 8, C, 3) unit directions.

    direction = normalize(corner - origin); rays with zero length are
    invalid (corner exactly at the camera origin).
    """
    corners = np.asarray(corners, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    diff = corners[:, :, None, :] - origins[None, None, :, :]
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    bad = norm[..., 0] < _EPS
    dirs = np.zeros_like(diff)
    nz = ~bad
    dirs[nz] = diff[nz] / norm[nz]
    return dirs, bad


def moller_trumbore(origins, dirs, tris):
    """Double-sided Moller-Trumbore ray-triangle intersection (oracle).

    origins: (R, 3), dirs: (R, 3) unit, tris: (M, 3, 3) rows (V0,V1,V2).
    Returns (hit, face_id, bary, t):
      hit: (R,) bool; face_id: (R,) int32 (-1 where miss);
      bary: (R, 3) (b0,b1,b2) with P = b0 V0 + b1 V1 + b2 V2;
      t: (R,) float (distance along the ray, inf where miss).
    Double-sided: back-face hits are accepted, matching mathutils BVH.
    """
    origins = np.asarray(origins, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.float64)
    R = len(origins)
    M = len(tris)
    o = origins[:, None, :]  # (R, 1, 3)
    d = dirs[:, None, :]
    v0 = tris[None, :, 0]
    v1 = tris[None, :, 1]
    v2 = tris[None, :, 2]
    e1 = v1 - v0
    e2 = v2 - v0
    p = np.cross(d, np.tile(e2, (R, 1, 1)))  # (R, M, 3)
    det = np.einsum("rmj,rmj->rm", e1, p)
    s = o - v0
    u_all = np.einsum("rmj,rmj->rm", s, p)
    q = np.cross(s, np.tile(e1, (R, 1, 1)))
    v_all = np.einsum("rmj,rmj->rm", d, q)
    t_all = np.einsum("rmj,rmj->rm", e2, q)

    det_ok = np.abs(det) > _EPS
    inv = np.zeros_like(det)
    inv[det_ok] = 1.0 / det[det_ok]
    u = u_all * inv
    v = v_all * inv
    t = t_all * inv

    in_tri = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    hit_any = det_ok & in_tri & (t >= 0.0)  # (R, M)
    # nearest hit per ray
    t_safe = np.where(hit_any, t, np.inf)
    best = np.argmin(t_safe, axis=-1)  # (R,)
    hit = hit_any[np.arange(R), best]
    face_id = np.where(hit, best, -1).astype(np.int32)
    uu = u[np.arange(R), best]
    vv = v[np.arange(R), best]
    tt = t[np.arange(R), best]
    bary = np.stack([1.0 - uu - vv, uu, vv], axis=-1)
    tt = np.where(hit, tt, np.inf)
    return hit, face_id, bary, tt


class MollerTrumboreCaster:
    """Numpy oracle caster; also used as the fallback outside Blender."""

    def __init__(self, tris):
        self.tris = np.asarray(tris, dtype=np.float64)

    def cast(self, origins, dirs, progress=None):
        hit, face_id, bary, t = moller_trumbore(origins, dirs, self.tris)
        loc = origins + dirs * np.where(np.isfinite(t), t, 0.0)[:, None]
        if progress is not None:
            progress(len(origins), len(origins))
        return hit, face_id, loc, t


class BVHCaster:
    """mathutils.bvhtree.BVHTree caster for use inside Blender.

    Construction from numpy triangles is O(M) Python; the per-ray query
    is O(log M). mathutils is imported lazily.
    """

    def __init__(self, tris):
        import mathutils
        from mathutils.bvhtree import BVHTree

        self.tris = np.asarray(tris, dtype=np.float64)
        verts = [mathutils.Vector(tuple(v)) for v in self.tris.reshape(-1, 3)]
        faces = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(self.tris))]
        self.tree = BVHTree.FromPolygons(verts, faces)

    def cast(self, origins, dirs, progress=None):
        """Cast all rays; call progress(done, total) every ~0.5%."""
        origins = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)
        R = len(origins)
        hit = np.zeros(R, dtype=bool)
        face_id = np.full(R, -1, dtype=np.int32)
        loc = np.zeros((R, 3), dtype=np.float64)
        t = np.full(R, np.inf, dtype=np.float64)
        report_every = max(1, R // 200)
        for i in range(R):
            o = tuple(origins[i])
            d = tuple(dirs[i])
            res = self.tree.ray_cast(o, d)
            # Blender 5.2 returns (None, None, None, None) on a miss
            # instead of None, so test the location element.
            if res is not None and res[0] is not None:
                location, _normal, index, distance = res
                hit[i] = True
                face_id[i] = index
                loc[i] = (location.x, location.y, location.z)
                t[i] = distance
            if progress is not None and (i % report_every == 0 or i == R - 1):
                progress(i + 1, R)
        return hit, face_id, loc, t


def compute_barycentric(points, tris, ids):
    """Barycentric coords (b0,b1,b2) of points on triangles.

    points: (R, 3); tris: (M, 3, 3); ids: (R,) int triangle index.
    P = b0 V0 + b1 V1 + b2 V2. Missing ids (-1) -> zeros.
    """
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int32)
    R = len(points)
    out = np.zeros((R, 3), dtype=np.float64)
    ok = ids >= 0
    if not ok.any():
        return out
    idx = ids[ok]
    v0 = tris[idx, 0]
    v1 = tris[idx, 1]
    v2 = tris[idx, 2]
    e1 = v1 - v0
    e2 = v2 - v0
    p = points[ok] - v0
    a11 = np.einsum("ri,ri->r", e1, e1)
    a12 = np.einsum("ri,ri->r", e1, e2)
    a22 = np.einsum("ri,ri->r", e2, e2)
    b1 = np.einsum("ri,ri->r", p, e1)
    b2 = np.einsum("ri,ri->r", p, e2)
    det = a11 * a22 - a12 * a12
    safe = np.abs(det) > _EPS
    u = np.zeros_like(b1)
    v = np.zeros_like(b1)
    u[safe] = (b1[safe] * a22[safe] - a12[safe] * b2[safe]) / det[safe]
    v[safe] = (a11[safe] * b2[safe] - b1[safe] * a12[safe]) / det[safe]
    out[ok, 0] = 1.0 - u - v
    out[ok, 1] = u
    out[ok, 2] = v
    return out


def bind_corners(corners, origins, tris, caster=None, centers=None,
                 progress=None):
    """Bind every BBX corner to its best proxy face via camera rays.

    corners: (N, 8, 3) world corners; origins: (C, 3) camera centers;
    tris: (M, 3, 3) world triangles; centers: (N, 3) Gaussian centers
    (used to pick the face closest to the Gaussian per the paper);
    progress: optional done/total callback forwarded to the caster.

    Returns dict with keys: face_ids (N, 8) int32 (-1 invalid), bary
    (N, 8, 3), distances (N, 8) (hit-to-Gaussian-center distance, inf
    invalid), hit_points (N, 8, 3), valid (N, 8) bool.
    """
    corners = np.asarray(corners, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    if centers is None:
        centers = corners.mean(axis=1)
    centers = np.asarray(centers, dtype=np.float64)
    N = len(corners)
    C = len(origins)

    dirs, bad = ray_directions(corners, origins)  # (N, 8, C, 3)
    n_rays = N * 8 * C
    flat_orig = np.broadcast_to(origins[None, None, :, :], (N, 8, C, 3)).reshape(n_rays, 3)
    flat_dirs = dirs.reshape(n_rays, 3)
    flat_bad = bad.reshape(n_rays)

    if caster is None:
        caster = MollerTrumboreCaster(tris)
    hit, face_id, loc, t = caster.cast(flat_orig, flat_dirs, progress=progress)
    hit &= ~flat_bad

    hit = hit.reshape(N, 8, C)
    face_id = face_id.reshape(N, 8, C)
    loc = loc.reshape(N, 8, C, 3)

    # distance from the hit point to the Gaussian center (paper: keep
    # the face closest to the Gaussian)
    d2 = np.linalg.norm(loc - centers[:, None, None, :], axis=-1)  # (N, 8, C)
    score = np.where(hit, d2, np.inf)
    best = np.argmin(score, axis=-1)  # (N, 8)

    ii = np.arange(N)[:, None]
    kk = np.arange(8)[None, :]
    valid = hit[ii, kk, best]
    out_face = np.where(valid, face_id[ii, kk, best], -1).astype(np.int32)
    out_hit = np.where(valid[..., None], loc[ii, kk, best, :], 0.0)
    out_bary = compute_barycentric(
        out_hit.reshape(-1, 3), tris, out_face.reshape(-1)
    ).reshape(N, 8, 3)
    return {
        "face_ids": out_face,
        "barycentric": out_bary,
        "distances": np.where(valid, d2[ii, kk, best], np.inf),
        "hit_points": out_hit,
        "valid": valid,
    }
