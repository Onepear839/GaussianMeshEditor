"""Unit tests for unimags.deformation + unimags.covariance (pure numpy).

Run outside Blender:
  python test_unimags_deformation.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "gaussian_mesh_editor"))

from unimags.bbx import get_gaussian_bbx  # noqa: E402
from unimags.covariance import apply_deformation, propagate_covariance  # noqa: E402
from unimags.deformation import (  # noqa: E402
    corner_deformations,
    gaussian_updates,
    vertex_deformations,
)
from unimags.ray_binding import MollerTrumboreCaster, bind_corners  # noqa: E402
from unimags.rotation import (  # noqa: E402
    matrix_to_quat,
    quat_to_matrix,
    so3_exp,
)

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print("  ok -", msg)


def cube_mesh(half=1.0):
    """Triangulated cube (8 verts, 12 tris), vertex order as test_binding."""
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
        tris.append([a, b, c])
        tris.append([a, c, d])
    return v, np.array(tris, dtype=np.int32)


def axis_rotation(axis, angle):
    return so3_exp(np.array(axis, dtype=np.float64) * angle)


rng = np.random.default_rng(7)


# ---------------------------------------------------------------- vertex
print("vertex_deformations")
V, T = cube_mesh(1.0)

# 1. pure translation
t = np.array([0.3, -0.2, 0.5])
delta, rot, shear = vertex_deformations(V, V + t, T)
ok(np.allclose(delta, t, atol=1e-12), "translation -> uniform delta")
ok(np.allclose(rot, np.eye(3), atol=1e-9), "translation -> identity rotation")
ok(np.allclose(shear, np.eye(3), atol=1e-9), "translation -> identity shear")

# 2. rigid rotation
Rrot = axis_rotation([0.4, -0.7, 0.2], 0.9)
cur = V @ Rrot.T
delta, rot, shear = vertex_deformations(V, cur, T)
ok(np.allclose(rot, Rrot, atol=1e-9), "rigid rotation -> per-vertex R == Rrot")
ok(np.allclose(shear, np.eye(3), atol=1e-9), "rigid rotation -> identity shear")
ok(np.allclose(delta, V @ Rrot.T - V, atol=1e-12), "rigid rotation -> consistent delta")

# 3. uniform scale (normal pseudo-edge scales too -> exact D = s I)
s = 1.3
delta, rot, shear = vertex_deformations(V, V * s, T)
ok(np.allclose(rot, np.eye(3), atol=1e-9), "uniform scale -> identity rotation")
ok(np.allclose(shear, s * np.eye(3), atol=1e-6), "uniform scale -> shear == s I")
ok(np.allclose(delta, V * (s - 1.0), atol=1e-12), "uniform scale -> consistent delta")

# 4. non-uniform scale: shear picks up the anisotropy, rotation deviates
#    slightly (inherent ACAP one-ring approximation error; the paper's
#    full nonlinear ACAP solves this exactly)
su = np.array([1.2, 0.8, 1.4])
delta, rot, shear = vertex_deformations(V, V * su, T)
ev = np.linalg.eigvalsh(shear)
ok(np.allclose(np.sort(ev, axis=1), np.sort(su), atol=0.3),
   "non-uniform scale -> shear eigenvalues ~= scale factors (ACAP approx)")

# 5. isolated vertex keeps identity
v2 = V[:4].copy() + t
t2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
iso_tris = np.array([[0, 1, 2]], dtype=np.int32)
_, rot_i, shear_i = vertex_deformations(v2, v2 + t2, iso_tris)
ok(np.allclose(rot_i[3], np.eye(3)), "isolated vertex -> identity rotation")
ok(np.allclose(shear_i[3], np.eye(3)), "isolated vertex -> identity shear")

# ------------------------------------------------------------ corners
print("corner_deformations")
N, C = 3, 8
tri_ids = np.full((N, C), 1, dtype=np.int32)   # all corners on triangle 1
bary = np.zeros((N, C, 3))
bary[...] = [1.0, 0.0, 0.0]                     # corner == vertex 0 of the tri
nd = np.arange(len(V)) % 3                      # fake per-vertex motion: (i%3)
delta_v = np.stack([np.arange(len(V)) % 3,
                    (np.arange(len(V)) + 1) % 3,
                    (np.arange(len(V)) + 2) % 3], axis=1).astype(np.float64) * 0.1
rot_v = np.tile(np.eye(3), (len(V), 1, 1))
shear_v = np.tile(np.eye(3), (len(V), 1, 1))
c_d, c_lr, c_s = corner_deformations(tri_ids, bary, T, delta_v, rot_v, shear_v)
v0 = T[1, 0]
ok(np.allclose(c_d, delta_v[v0], atol=1e-12), "bary (1,0,0) -> corner == vertex-0 delta")
ok(np.allclose(c_s, shear_v[v0], atol=1e-12), "bary (1,0,0) -> corner == vertex-0 shear")
ok(np.allclose(c_lr, np.zeros((N, C, 3)), atol=1e-12), "identity rotations -> zero logR")

# mixed barycentric weights average the three vertices
bary2 = np.zeros((N, C, 3))
bary2[...] = [0.5, 0.3, 0.2]
c_d2, _, _ = corner_deformations(tri_ids, bary2, T, delta_v, rot_v, shear_v)
expected = 0.5 * delta_v[T[1, 0]] + 0.3 * delta_v[T[1, 1]] + 0.2 * delta_v[T[1, 2]]
ok(np.allclose(c_d2, expected, atol=1e-12), "bary weights -> convex combination of deltas")

# invalid corners are zeroed
tri_ids_mix = tri_ids.copy()
tri_ids_mix[0, 3] = -1
c_d3, c_lr3, c_s3 = corner_deformations(tri_ids_mix, bary, T, delta_v, rot_v, shear_v)
ok(np.allclose(c_d3[0, 3], 0.0) and np.allclose(c_s3[0, 3], 0.0) and
   np.allclose(c_lr3[0, 3], 0.0), "invalid corner -> zeroed delta/logR/shear")

# ---------------------------------------------------------- gaussian
print("gaussian_updates")
mu = rng.normal(size=(N, 3))
Rg = np.tile(axis_rotation([1, 0, 0], 0.3), (N, 1, 1))
sg = np.tile(np.array([0.5, 1.0, 2.0]), (N, 1))
d_all = np.full((N, C, 3), [0.1, 0.2, 0.3])
lr_all = np.zeros((N, C, 3))
s_all = np.tile(np.eye(3) * 0.5, (N, C, 1, 1))
vm_full = np.ones((N, C), dtype=bool)
mu_n, R_inc, S_inc = gaussian_updates(mu, Rg, sg, d_all, lr_all, s_all, vm_full)
ok(np.allclose(mu_n, mu + [0.1, 0.2, 0.3], atol=1e-12), "full valid -> mu' = mu + mean delta")
ok(np.allclose(S_inc, np.eye(3) * 0.5, atol=1e-12), "full valid -> S' = mean S")
ok(np.allclose(R_inc, np.eye(3), atol=1e-12), "zero logR -> R' = I")

# partial validity averages only the valid corners
vm_part = vm_full.copy()
vm_part[1, :4] = False
lr2 = np.zeros((N, C, 3))
lr2[1, :, 2] = 0.4
mu_n2, R_inc2, S_inc2 = gaussian_updates(mu, Rg, sg, d_all, lr2, s_all, vm_part)
ok(np.allclose(mu_n2[1], mu[1] + [0.1, 0.2, 0.3], atol=1e-12),
   "partial valid -> mean over valid corners only")
w = np.array([0.0, 0.0, 0.4])
ok(np.allclose(R_inc2[1], so3_exp(w), atol=1e-12),
   "partial valid -> logR averaged over valid corners")

# no valid corners -> unchanged position, identity increment
vm_none = np.zeros((N, C), dtype=bool)
mu_n3, R_inc3, S_inc3 = gaussian_updates(mu, Rg, sg, d_all, lr_all, s_all, vm_none)
ok(np.allclose(mu_n3, mu, atol=1e-12), "no valid corner -> mu unchanged")
ok(np.allclose(R_inc3, np.eye(3), atol=1e-12), "no valid corner -> R' = I")
ok(np.allclose(S_inc3, np.eye(3), atol=1e-12), "no valid corner -> S' = I")

# -------------------------------------------------------- covariance
print("propagate_covariance")
Nc = 5
Rc = np.stack([axis_rotation([1, 0, 0], a) for a in rng.uniform(-1, 1, Nc)])
Rc = Rc @ np.stack([axis_rotation([0, 1, 0], a) for a in rng.uniform(-1, 1, Nc)])
sc = rng.uniform(0.3, 2.0, size=(Nc, 3))
I3 = np.eye(3)

# 6. identity increment -> exact preservation
Ro, so = propagate_covariance(Rc, sc, np.tile(I3, (Nc, 1, 1)), np.tile(I3, (Nc, 1, 1)))
ok(np.array_equal(Ro, Rc) and np.array_equal(so, sc),
   "identity increment -> rot/scale preserved exactly")

# 7. pure rotation increment: Sigma' = R' Sigma R'^T (covariance check)
dRinc = axis_rotation([0.2, -0.5, 0.1], 0.7)
Rinc = np.tile(dRinc, (Nc, 1, 1))
Ro, so = propagate_covariance(Rc, sc, Rinc, np.tile(I3, (Nc, 1, 1)))
Sigma = (Rc * sc[:, None, :] ** 2) @ np.swapaxes(Rc, -1, -2)
Sig_expected = Rinc @ Sigma @ np.swapaxes(Rinc, -1, -2)
Sig_out = (Ro * so[:, None, :] ** 2) @ np.swapaxes(Ro, -1, -2)
ok(np.allclose(Sig_out, Sig_expected, atol=1e-9),
   "rotation increment -> propagated covariance matches Eq.13")

# 8. uniform shear increment S' = c I: Sigma' = c^2 Sigma
csh = 1.6
Sinc = np.tile(I3 * csh, (Nc, 1, 1))
Ro, so = propagate_covariance(Rc, sc, np.tile(I3, (Nc, 1, 1)), Sinc)
Sig_out2 = (Ro * so[:, None, :] ** 2) @ np.swapaxes(Ro, -1, -2)
Sig_exp2 = csh * csh * (Rc * sc[:, None, :] ** 2) @ np.swapaxes(Rc, -1, -2)
ok(np.allclose(Sig_out2, Sig_exp2, atol=1e-9),
   "uniform shear -> covariance scaled by c^2")

# 9. random combined increment (rotation + symmetric PSD shear)
for _ in range(3):
    Rr = axis_rotation(rng.normal(size=3), rng.uniform(0.1, 1.5))
    Sr = rng.normal(size=(3, 3))
    Sr = Sr @ Sr.T + I3                     # symmetric PSD
    Ro, so = propagate_covariance(Rc, sc,
                                  np.tile(Rr, (Nc, 1, 1)),
                                  np.tile(Sr, (Nc, 1, 1)))
    Mr = Rr @ Sr @ Rc                       # A = R' S' R
    Sig_e = (Mr * sc[:, None, :] ** 2) @ np.swapaxes(Mr, -1, -2)
    Sig_o = (Ro * so[:, None, :] ** 2) @ np.swapaxes(Ro, -1, -2)
    ok(np.allclose(Sig_o, Sig_e, atol=1e-8), "random increment -> Eq.13 covariance holds")
    ok(np.allclose(np.linalg.det(Ro), 1.0, atol=1e-9), "random increment -> det(R_out) = +1")
    ok((so > 0).all(), "random increment -> positive scales")

# 10. asymmetric S_inc (numerical noise from barycentric averaging)
Snoisy = np.tile(np.array([[1.0, 1e-6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), (Nc, 1, 1))
Ro, so = propagate_covariance(Rc, sc, np.tile(I3, (Nc, 1, 1)), Snoisy)
ok(np.allclose(np.linalg.det(Ro), 1.0, atol=1e-9), "noisy S_inc -> still a rotation")

# 11. end-to-end apply_deformation: invalid gaussians untouched
mu_all = rng.normal(size=(Nc, 3))
c_dE = rng.normal(size=(Nc, 8, 3)) * 0.1
c_lrE = rng.normal(size=(Nc, 8, 3)) * 0.1
c_sE = np.tile(I3, (Nc, 8, 1, 1)) + rng.normal(scale=0.05, size=(Nc, 8, 3, 3))
vm_E = np.ones((Nc, 8), dtype=bool)
vm_E[3] = False
mu_nE, RoE, soE = apply_deformation(mu_all, Rc, sc, c_dE, c_lrE, c_sE, vm_E)
ok(np.allclose(mu_nE[3], mu_all[3], atol=1e-12), "apply_deformation -> unbound mu untouched")
ok(np.array_equal(RoE[3], Rc[3]) and np.array_equal(soE[3], sc[3]),
   "apply_deformation -> unbound rot/scale untouched")
ok(not np.allclose(mu_nE[0], mu_all[0], atol=1e-9), "apply_deformation -> bound mu moved")
ok(np.allclose(np.linalg.det(RoE[0]), 1.0, atol=1e-9), "apply_deformation -> bound rotation proper")

# 12. quaternion round-trip through the KIRI layout
q_new = matrix_to_quat(RoE)
ok(np.allclose(quat_to_matrix(q_new), RoE, atol=1e-9),
   "matrix_to_quat round-trips the propagated rotation")

# ------------------------------------------------- end-to-end pipeline
print("end-to-end (BBX -> bind -> deform -> update)")
v_cube, t_cube = cube_mesh(1.0)
tris3 = v_cube[t_cube]                       # (12, 3, 3)
ng = 24
mus = rng.uniform(-0.8, 0.8, size=(ng, 3))
qs = rng.normal(size=(ng, 4))
qs /= np.linalg.norm(qs, axis=-1, keepdims=True)
Rgs = quat_to_matrix(qs)
scales = rng.uniform(0.05, 0.25, size=(ng, 3))
corners = get_gaussian_bbx(mus, Rgs, scales, k=1.0)
cams = np.array([[5.0, 2.0, 2.0], [-5.0, 2.0, 2.0], [2.0, 5.0, -2.0], [2.0, -5.0, -2.0]])
bind = bind_corners(corners, cams, tris3, caster=MollerTrumboreCaster(tris3), centers=mus)
ok(bind["valid"].sum() > ng * 4, "e2e -> majority of corners bound")
ok(np.isfinite(bind["distances"][bind["valid"]]).all() and
   bind["distances"][bind["valid"]].max() < 3.0,
   "e2e -> hit distances finite and bounded")

# 13. mesh translated: gaussians follow (rotation/shear stay identity)
tvec = np.array([0.4, -0.3, 0.2])
cur3 = tris3 + tvec
cur_v = v_cube + tvec
delta_v, rot_v, shear_v = vertex_deformations(v_cube, cur_v, t_cube)
c_d, c_lr, c_s = corner_deformations(bind["face_ids"], bind["barycentric"], t_cube,
                                     delta_v, rot_v, shear_v)
mu_n, Ro, so = apply_deformation(mus, Rgs, scales, c_d, c_lr, c_s, bind["valid"])
ok(np.allclose(mu_n[bind["valid"].any(axis=1)], mus[bind["valid"].any(axis=1)] + tvec,
               atol=1e-9), "e2e -> bound gaussians translate with the mesh")
ok(np.array_equal(Ro, Rgs) and np.array_equal(so, scales),
   "e2e -> translation preserves rotations and scales exactly")

# 14. mesh unchanged: everything stays put
mu_n2, Ro2, so2 = apply_deformation(mus, Rgs, scales,
                                    np.zeros((ng, 8, 3)), np.zeros((ng, 8, 3)),
                                    np.tile(I3, (ng, 8, 1, 1)), bind["valid"])
ok(np.array_equal(mu_n2, mus) and np.array_equal(Ro2, Rgs) and np.array_equal(so2, scales),
   "e2e -> no deformation leaves every Gaussian untouched")

print()
print("PASS: {} assertions".format(PASS))
