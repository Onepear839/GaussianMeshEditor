"""Unit tests for unimags.rotation — pure numpy, no Blender required.

Run: python test_rotation.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "gaussian_mesh_editor"))

from unimags.rotation import (  # noqa: E402
    matrix_to_quat,
    polar_decompose,
    quat_to_matrix,
    skew,
    so3_exp,
    so3_log,
)

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print("  ok -", msg)


def rng():
    return np.random.default_rng(12345)


def rand_quats(n, r):
    q = r.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def rand_rotations(n, r):
    return quat_to_matrix(rand_quats(n, r))


# ---------------------------------------------------------------- quat/matrix
print("quat_to_matrix / matrix_to_quat")

q0 = np.array([1.0, 0.0, 0.0, 0.0])
ok(np.allclose(quat_to_matrix(q0), np.eye(3)), "identity quaternion -> identity matrix")

# rotation about x by 90 deg: q = (cos45, sin45, 0, 0)
qx = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
Rx = quat_to_matrix(qx)
expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
ok(np.allclose(Rx, expected, atol=1e-12), "90 deg about x matches known matrix")

r = rng()
n = 500
qs = rand_quats(n, r)
Rs = quat_to_matrix(qs)
qs2 = matrix_to_quat(Rs)
Rs2 = quat_to_matrix(qs2)
ok(np.allclose(Rs, Rs2, atol=1e-10), "quat -> R -> quat -> R roundtrip")
# quats may differ by global sign; after sign-normalization (w>=0) they must match
qs_n = np.where(qs[..., :1] < 0, -qs, qs)
qs2_n = np.where(qs2[..., :1] < 0, -qs2, qs2)
ok(np.allclose(qs_n, qs2_n, atol=1e-10), "quat roundtrip matches up to global sign")

# 180-degree rotation (w == 0 path of Shepperd selection)
Rpi = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
qpi = matrix_to_quat(Rpi)
ok(np.allclose(quat_to_matrix(qpi), Rpi, atol=1e-10), "180 deg rotation survives w=0 branch")

# orthogonality of produced rotations
ok(np.allclose(np.einsum("nij,nkj->nik", Rs, Rs), np.tile(np.eye(3), (n, 1, 1)), atol=1e-10),
   "all rotation matrices are orthogonal")

# ---------------------------------------------------------------- so3 exp/log
print("so3_exp / so3_log")

omega = np.array([0.5, -0.3, 0.8])
Rw = so3_exp(omega)
angle = np.linalg.norm(omega)
axis = omega / angle
K = skew(axis)
R_rod = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
ok(np.allclose(Rw, R_rod, atol=1e-12), "so3_exp matches Rodrigues formula")

omegas = r.normal(scale=0.3, size=(200, 3))  # small angles keep branch away from pi
ok(np.allclose(so3_log(so3_exp(omegas)), omegas, atol=1e-9), "so3_log(so3_exp(w)) == w (small angles)")

# so3_exp(0) is identity
ok(np.allclose(so3_exp(np.zeros(3)), np.eye(3), atol=1e-12), "so3_exp(0) == I")

# a larger angle roundtrip (up to ~150 deg, below the pi instability);
# so3_log returns the principal rotation vector with norm <= pi, so
# sample axis-angle with |angle| < pi directly
axes = r.normal(size=(100, 3))
axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
angles = r.uniform(-2.6, 2.6, size=(100, 1))
big = axes * angles
ok(np.allclose(so3_log(so3_exp(big)), big, atol=1e-8), "so3 log/exp roundtrip up to ~150 deg")

# ---------------------------------------------------------------- polar decom
print("polar_decompose")

r = rng()
D = r.normal(size=(200, 3, 3))
R, S = polar_decompose(D)
ok(np.allclose(R @ S, D, atol=1e-10), "D = R @ S reconstruction")
ok(np.allclose(R @ np.swapaxes(R, -1, -2), np.tile(np.eye(3), (200, 1, 1)), atol=1e-10),
   "R is orthogonal")
ok(np.allclose(S, np.swapaxes(S, -1, -2), atol=1e-10), "S is symmetric")
evals = np.linalg.eigvalsh(S)
ok(evals.min() > -1e-9, "S is positive semi-definite")

# det(R) = +1 whenever det(D) > 0 (mesh deformation gradients have
# positive determinant; det(D) < 0 gives the reflection polar form,
# which does not occur for non-inverted mesh edits)
pos = np.linalg.det(D) > 0
det = np.linalg.det(R)
ok(np.allclose(det[pos], 1.0, atol=1e-8), "R has det +1 for det(D) > 0")
ok(np.allclose(det[~pos], -1.0, atol=1e-8), "R is a reflection iff det(D) < 0")

# known factorization: R_true @ S_true (S_true symmetric PSD) is recovered
r = rng()
qs = rand_quats(100, r)
R_true = quat_to_matrix(qs)
S_true = np.zeros((100, 3, 3))
S_true[..., 0, 0] = r.uniform(0.5, 2.0, 100)
S_true[..., 1, 1] = r.uniform(0.5, 2.0, 100)
S_true[..., 2, 2] = r.uniform(0.5, 2.0, 100)
D2 = R_true @ S_true
R2, S2 = polar_decompose(D2)
ok(np.allclose(R2, R_true, atol=1e-9), "polar decomposition recovers the rotation of D = R S")
ok(np.allclose(S2, S_true, atol=1e-9), "polar decomposition recovers the shear of D = R S")

print("\n%d checks passed" % PASS)
