"""Blender test: BVHCaster (mathutils BVHTree) == numpy MollerTrumbore oracle.

Run inside Blender headless:
  blender --background --python test_unimags_bvh.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "gaussian_mesh_editor"))

from unimags.ray_binding import (  # noqa: E402
    BVHCaster,
    MollerTrumboreCaster,
    bind_corners,
)

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print("  ok -", msg)


def make_cube_tris(half=1.0):
    h = half
    v = np.array(
        [
            [-h, -h, -h], [-h, -h, +h], [-h, +h, -h], [-h, +h, +h],
            [+h, -h, -h], [+h, -h, +h], [+h, +h, -h], [+h, +h, +h],
        ],
        dtype=np.float64,
    )
    quads = [
        (0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
        (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5),
    ]
    tris = []
    for a, b, c, d in quads:
        tris.append([v[a], v[b], v[c]])
        tris.append([v[a], v[c], v[d]])
    return np.array(tris)


print("BVHCaster vs MollerTrumboreCaster")

r = np.random.default_rng(42)
cube = make_cube_tris(1.0)
bv = BVHCaster(cube)
mt = MollerTrumboreCaster(cube)

# random rays around the cube (some hit, some miss)
origins = r.uniform(-4, 4, size=(500, 3))
dirs = r.normal(size=(500, 3))
dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

hit_b, fid_b, loc_b, t_b = bv.cast(origins, dirs)
hit_m, fid_m, loc_m, t_m = mt.cast(origins, dirs)

ok(np.array_equal(hit_b, hit_m), "hit flags identical (500 random rays)")
ok(np.array_equal(fid_b, fid_m), "face ids identical")
# BVHTree is float32 internally, the numpy oracle is float64 -> ~1e-6
# differences; 1e-4 is ample for binding purposes. Miss rays carry
# placeholder locations (0 vs origin), so compare per-ray on hits only.
same = (np.linalg.norm(loc_b - loc_m, axis=-1) <= 1e-4) | (~hit_b)
ok(same.all(), "hit locations identical (tol 1e-4)")
ok(np.allclose(t_b[hit_b], t_m[hit_m], atol=1e-4), "hit distances identical (tol 1e-4)")

# rays from a camera cluster through a target cluster (binding-like rays)
cams = r.uniform(-6, -3, size=(30, 3))
targets = r.normal(scale=0.3, size=(200, 3))
bundle = np.repeat(cams, len(targets), axis=0)
tgt = np.tile(targets, (len(cams), 1))
bd = tgt - bundle
bd /= np.linalg.norm(bd, axis=-1, keepdims=True)
hit_b2, fid_b2, loc_b2, t_b2 = bv.cast(bundle, bd)
hit_m2, fid_m2, loc_m2, t_m2 = mt.cast(bundle, bd)
ok(np.array_equal(hit_b2, hit_m2), "camera-bundle hit flags identical")
ok(np.array_equal(fid_b2, fid_m2), "camera-bundle face ids identical")
ok(np.allclose(t_b2[hit_b2], t_m2[hit_m2], atol=1e-4), "camera-bundle distances identical (tol 1e-4)")

# bind_corners end-to-end with both casters on a 64-Gaussian cloud
from unimags.bbx import get_gaussian_bbx  # noqa: E402
from unimags.rotation import quat_to_matrix  # noqa: E402

qs = r.normal(size=(64, 4))
qs /= np.linalg.norm(qs, axis=-1, keepdims=True)
Rs = quat_to_matrix(qs)
mus = r.normal(scale=0.2, size=(64, 3))
scales = r.uniform(0.05, 0.2, size=(64, 3))
corners = get_gaussian_bbx(mus, Rs, scales)
cams_b = np.array([[4.0, 1.0, 1.0], [-4.0, 1.0, 1.0], [1.0, 4.0, -1.0], [1.0, -4.0, -1.0]])

res_bv = bind_corners(corners, cams_b, cube, caster=bv)
res_mt = bind_corners(corners, cams_b, cube, caster=mt)
ok(np.array_equal(res_bv["face_ids"], res_mt["face_ids"]), "bind_corners face ids identical (BVH vs MT)")
ok(np.array_equal(res_bv["valid"], res_mt["valid"]), "bind_corners valid masks identical")
ok(np.allclose(res_bv["barycentric"], res_mt["barycentric"], atol=1e-4),
   "bind_corners barycentric identical (tol 1e-4)")
ok(np.allclose(res_bv["distances"], res_mt["distances"], atol=1e-4),
   "bind_corners distances identical (tol 1e-4)")
ok(np.allclose(res_bv["hit_points"], res_mt["hit_points"], atol=1e-4),
   "bind_corners hit points identical (tol 1e-4)")

print("\n%d checks passed" % PASS)
