"""Step 1: Gaussian-centric BBX — the 8 corners of each Gaussian's box.

Paper: 'a Gaussian kernel G with its BBX B, parameterized by a mean
vector mu and a covariance matrix Sigma' — the BBX is derived from the
Gaussian's own rotation and scale, NOT an axis-aligned world cube.

We build the oriented box centered at mu, axes along the Gaussian's
rotation R, half-extents k * scale along each principal axis:

    C_i = mu + k * R @ (sign_i * s),   i = 0..7

k is an experimental parameter (paper does not define the exact box
extent); fixed at 1.0 for now, kept as an argument for later sweeps.

All inputs/outputs are world space. mu: (N, 3); rot: (N, 3, 3);
scale: (N, 3) LINEAR scale (KIRI's gaussian_data stores linear scale).
"""

import numpy as np

from .rotation import quat_to_matrix

_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [+1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)


def get_gaussian_bbx(mu, rot, scale, k=1.0):
    """(N, 3) mu, (N, 3, 3) rot, (N, 3) linear scale -> (N, 8, 3) corners.

    Row order of corners is the _SIGNS order above.
    """
    mu = np.asarray(mu, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    rs = rot * scale[:, None, :]  # (N, 3, 3) == R @ diag(s)
    half = np.einsum("nij,oj->noi", rs, _SIGNS)  # (N, 8, 3)
    return mu[:, None, :] + k * half


def bbx_from_quats(mu, quats, scales, k=1.0):
    """Convenience entry point: raw gaussian_data columns.

    mu: (N, 3) world positions; quats: (N, 4) quaternion (w, x, y, z);
    scales: (N, 3) LINEAR scale. Returns (N, 8, 3) world corners.
    """
    return get_gaussian_bbx(mu, quat_to_matrix(quats), scales, k)
