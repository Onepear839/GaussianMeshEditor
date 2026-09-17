"""Unit tests for unimags.bbx — pure numpy, no Blender required.

Run: python test_unimags_bbx.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "gaussian_mesh_editor"))

from unimags.bbx import bbx_from_quats, get_gaussian_bbx  # noqa: E402
from unimags.rotation import quat_to_matrix  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print("  ok -", msg)


def rng():
    return np.random.default_rng(7)


# axis-aligned Gaussian: corners are mu +- k*s component-wise
mu = np.array([[1.0, 2.0, 3.0]])
s = np.array([[1.0, 2.0, 3.0]])
R = np.eye(3)[None]
corners = get_gaussian_bbx(mu, R, s, k=1.0)
ok(corners.shape == (1, 8, 3), "returns (N, 8, 3)")
expected = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 6.0],
        [0.0, 4.0, 0.0],
        [0.0, 4.0, 6.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 6.0],
        [2.0, 4.0, 0.0],
        [2.0, 4.0, 6.0],
    ]
)
ok(np.allclose(corners[0], expected), "identity rotation -> box centered at mu, extents +-s")

# box center is mu
ok(np.allclose(corners.mean(axis=1), mu), "center of 8 corners is mu")

# k scales the half-extents
corners_k = get_gaussian_bbx(mu, R, s, k=2.0)
ok(np.allclose(corners_k - mu[:, None, :], 2.0 * (corners - mu[:, None, :])),
   "k multiplies the half-extents")

# rotated Gaussian: corner offsets are k * R @ (sign * s)
r = rng()
n = 300
mus = r.normal(size=(n, 3))
scales = r.uniform(0.1, 3.0, size=(n, 3))
q = r.normal(size=(n, 4))
q /= np.linalg.norm(q, axis=-1, keepdims=True)
Rs = quat_to_matrix(q)
Cs = get_gaussian_bbx(mus, Rs, scales, k=1.0)
# R^T (C_i - mu) / k must equal sign_i * s for every corner
offsets = Cs - mus[:, None, :]  # (N, 8, 3)
back = np.einsum("nok,nkj->noj", offsets, Rs)  # (N, 8, 3) == offsets @ R == R^T offset
signs = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ],
    dtype=float,
)
expected = signs[None] * scales[:, None, :]
ok(np.allclose(back, expected, atol=1e-9), "corners are aligned with R, extents k*s per axis")

# quat convenience entry matches the matrix entry
Cs2 = bbx_from_quats(mus, q, scales, k=1.0)
ok(np.allclose(Cs2, Cs, atol=1e-12), "bbx_from_quats == get_gaussian_bbx(quat_to_matrix)")

# a flat (scale ~ 0) Gaussian degenerates to its center
flat = get_gaussian_bbx(mu, R, np.array([[1e-12, 1e-12, 1e-12]]))
ok(np.allclose(flat, mu[:, None, :], atol=1e-6), "zero-scale Gaussian -> all corners collapse to mu")

print("\n%d checks passed" % PASS)
