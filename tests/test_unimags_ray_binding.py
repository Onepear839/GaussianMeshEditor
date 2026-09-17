"""Unit tests for unimags.ray_binding — pure numpy, no Blender required.

Run: python test_unimags_ray_binding.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "gaussian_mesh_editor"))

from unimags.bbx import get_gaussian_bbx  # noqa: E402
from unimags.ray_binding import (  # noqa: E402
    MollerTrumboreCaster,
    bind_corners,
    compute_barycentric,
    moller_trumbore,
    ray_directions,
)

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print("  ok -", msg)


def rng():
    return np.random.default_rng(2026)


def make_cube_tris(half=1.0):
    """6 quad faces of a cube, each split into 2 triangles (12 tris)."""
    h = half
    v = np.array(
        [
            [-h, -h, -h],
            [-h, -h, +h],
            [-h, +h, -h],
            [-h, +h, +h],
            [+h, -h, -h],
            [+h, -h, +h],
            [+h, +h, -h],
            [+h, +h, +h],
        ],
        dtype=np.float64,
    )
    quads = [
        (0, 2, 3, 1),  # x = -1
        (4, 5, 7, 6),  # x = +1
        (0, 1, 5, 4),  # y = -1
        (2, 6, 7, 3),  # y = +1
        (0, 4, 6, 2),  # z = -1
        (1, 3, 7, 5),  # z = +1
    ]
    tris = []
    for a, b, c, d in quads:
        tris.append([v[a], v[b], v[c]])
        tris.append([v[a], v[c], v[d]])
    return np.array(tris)


# ---------------------------------------------------------------- MT oracle
print("moller_trumbore")

tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # xy plane
o = np.array([[0.25, 0.25, 1.0]])  # above
d = np.array([[0.0, 0.0, -1.0]])
hit, fid, bary, t = moller_trumbore(o, d, tri)
ok(hit[0] and t[0] == 1.0, "ray straight down hits the triangle at t=1")
ok(np.allclose(bary[0], [0.5, 0.25, 0.25]), "barycentric of hit point is correct")
ok(fid[0] == 0, "face id reported")

# back-face hit (double sided): triangle normal is +z, so a ray from
# below going up (+z) strikes the back face
hit2, fid2, bary2, t2 = moller_trumbore(
    np.array([[0.25, 0.25, -1.0]]), np.array([[0.0, 0.0, 1.0]]), tri
)
ok(hit2[0] and abs(t2[0] - 1.0) < 1e-12, "back-face ray also hits (double-sided)")
ok(np.allclose(bary2[0], [0.5, 0.25, 0.25]), "back-face barycentric is identical")

# parallel ray misses
hit3, _, _, _ = moller_trumbore(np.array([[2.0, 2.0, 1.0]]), np.array([[0.0, 1.0, 0.0]]), tri)
ok(not hit3[0], "parallel/offset ray misses")

# outside the triangle silhouette misses
hit4, _, _, _ = moller_trumbore(np.array([[2.0, 2.0, 1.0]]), d, tri)
ok(not hit4[0], "ray beyond triangle edges misses")

# multiple triangles: nearest hit wins
tris2 = np.array(
    [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # z=0, near
        [[0.0, 0.0, -5.0], [1.0, 0.0, -5.0], [0.0, 1.0, -5.0]],  # z=-5, far
    ]
)
hit5, fid5, _, t5 = moller_trumbore(np.array([[0.25, 0.25, 1.0]]), d, tris2)
ok(hit5[0] and fid5[0] == 0 and t5[0] == 1.0, "nearest triangle is chosen")

# ---------------------------------------------------------------- directions
print("ray_directions")

corners = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
origins = np.array([[0.0, 0.0, 0.0]])
dirs, bad = ray_directions(corners, origins)
ok(np.allclose(dirs[0, 0], [1, 0, 0]) and np.allclose(dirs[0, 1], [0, 1, 0]),
   "directions point from origin to corner")
ok(np.linalg.norm(dirs, axis=-1).shape == (1, 2, 1) and np.allclose(np.linalg.norm(dirs, axis=-1), 1.0),
   "directions are unit length")

# ---------------------------------------------------------------- barycentric
print("compute_barycentric")

pts = np.array([[0.25, 0.25, 0.0]])
ids = np.array([0])
b = compute_barycentric(pts, tri, ids)
ok(np.allclose(b[0], [0.5, 0.25, 0.25]), "barycentric reconstructs known point")
ok(np.allclose(b[0, 0] * tri[0, 0] + b[0, 1] * tri[0, 1] + b[0, 2] * tri[0, 2], pts[0]),
   "P == b0 V0 + b1 V1 + b2 V2")
b_miss = compute_barycentric(np.array([[0.0, 0.0, 0.0]]), tri, np.array([-1]))
ok(np.allclose(b_miss[0], 0.0), "invalid ids yield zero barycentric")

# ---------------------------------------------------------------- bind_corners
print("bind_corners")

cube = make_cube_tris(1.0)
# Gaussian at the cube center, axis-aligned, tiny scale: all 8 corners
# are near the center, so every camera ray toward a corner pierces a
# cube face -> all corners bound
mu = np.array([[0.0, 0.0, 0.0]])
scale = np.array([[0.1, 0.1, 0.1]])
R = np.eye(3)[None]
corners8 = get_gaussian_bbx(mu, R, scale, k=1.0)
cams = np.array([[0.0, 0.0, 5.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
res = bind_corners(corners8, cams, cube)
ok(res["face_ids"].shape == (1, 8) and res["barycentric"].shape == (1, 8, 3),
   "output shapes (N, 8) / (N, 8, 3)")
ok(res["valid"].all(), "all 8 corners of a centered Gaussian bind to the cube")
ok((res["face_ids"] >= 0).all() and (res["face_ids"] < 12).all(), "face ids in range")
ok(np.allclose(res["barycentric"].sum(axis=-1), 1.0), "barycentric rows sum to 1")
# hit points lie on the cube surface (|x| or |y| or |z| == 1)
hp = res["hit_points"][0]
on_surface = (np.abs(np.abs(hp) - 1.0).min(axis=-1) < 1e-9)
ok(on_surface.all(), "hit points lie on the proxy surface")

# Gaussian far outside the cube: corner rays miss -> unbound corners
mu2 = np.array([[10.0, 10.0, 10.0]])
corners2 = get_gaussian_bbx(mu2, R, scale, k=1.0)
res2 = bind_corners(corners2, cams, cube)
ok(not res2["valid"].any(), "Gaussian outside the proxy silhouette binds nothing")

# two cameras, one blocked: the visible camera's face wins
mu3 = np.array([[0.0, 0.0, 0.0]])
corners3 = get_gaussian_bbx(mu3, R, scale, k=1.0)
# one camera far on +x, another far on -x: a corner at +x is nearer to
# the +x camera's hit (z=0 plane behind it is farther) -> face ids differ
cams3 = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
res3 = bind_corners(corners3, cams3, cube)
ok(res3["valid"].all(), "both cameras bind all corners")
# paper criterion: the stored hit must be the camera hit closest to the
# Gaussian center, not merely the nearest face along a single ray
corner = corners3[0, 4]
best_dist = np.inf
best_hit = None
for cam in cams3:
    dvec = corner - cam
    dvec = dvec / np.linalg.norm(dvec)
    hit, _fid, _bary, t = moller_trumbore(cam[None], dvec[None], cube)
    if hit[0]:
        pt = cam + dvec * t[0]
        dd = np.linalg.norm(pt - mu3[0])
        if dd < best_dist:
            best_dist = dd
            best_hit = pt
ok(np.allclose(res3["hit_points"][0, 4], best_hit), "stored hit is closest to the Gaussian across cameras")
ok(np.allclose(res3["distances"][0, 4], best_dist), "stored distance is the min over cameras")

# MollerTrumboreCaster path (explicit caster) matches default
caster = MollerTrumboreCaster(cube)
res4 = bind_corners(corners8, cams, cube, caster=caster)
ok(np.array_equal(res4["face_ids"], res["face_ids"]) and
   np.allclose(res4["barycentric"], res["barycentric"]),
   "explicit MollerTrumboreCaster == default caster")

print("\n%d checks passed" % PASS)
