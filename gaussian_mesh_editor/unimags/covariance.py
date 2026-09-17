"""Step 10: covariance propagation (paper Eq.13) and KIRI re-factorization.

Eq.13: Sigma' = R' S' Sigma (R' S')^T, where Sigma = R diag(s^2) R^T is
the Gaussian's own covariance (R its rotation, s its LINEAR scale) and
(R', S') is the corner-averaged deformation *increment* produced by
deformation.gaussian_updates.

The propagated covariance must be re-factorized into
Sigma' = R_out diag(s_out^2) R_out^T so it fits KIRI's (quaternion,
linear scale) layout. We do it without ever forming Sigma explicitly:

  A = R' S' R                one combined 3x3 (Sigma' = A diag(s^2) A^T)
  A = R_p S_p                polar decomposition (R_p rotation, S_p PSD)
  B = S_p diag(s^2) S_p      symmetric PSD
  B = V diag(lambda) V^T     symmetric eigendecomposition
  R_out = R_p V,  s_out = sqrt(lambda)

An identity increment (R' = I, S' = I, i.e. no deformation) keeps the
original factorization bit-for-bit, so an untouched Gaussian never gets
its quaternion rewritten by eigendecomposition noise.

Everything is vectorized over a leading batch axis (N gaussians).
"""

import numpy as np

from .rotation import polar_decompose

_EPS = 1e-12
_IDENT_TOL = 1e-9


def propagate_covariance(rot, scale, R_inc, S_inc):
    """Eq.13: Sigma' = R' S' Sigma (R' S')^T, re-factorized for KIRI.

    rot (N, 3, 3) rest rotations; scale (N, 3) rest LINEAR scales;
    R_inc (N, 3, 3) / S_inc (N, 3, 3) deformation increments.

    Returns (rot_out (N, 3, 3), scale_out (N, 3)) with det(rot_out) = +1
    and Sigma' = rot_out diag(scale_out^2) rot_out^T. Gaussians whose
    increment is identity come back with rot_out = rot, scale_out = scale
    exactly (no re-factorization).
    """
    rot = np.asarray(rot, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    R_inc = np.asarray(R_inc, dtype=np.float64)
    S_inc = np.asarray(S_inc, dtype=np.float64)

    rot_out = rot.copy()
    scale_out = np.array(scale, dtype=np.float64)

    # Identity increment -> keep the rest factorization exactly. This also
    # covers fully-unbound Gaussians, whose increment is forced to I.
    dR = np.max(np.abs(R_inc - np.eye(3)), axis=(-2, -1))
    dS = np.max(np.abs(S_inc - np.eye(3)), axis=(-2, -1))
    active = (dR >= _IDENT_TOL) | (dS >= _IDENT_TOL)
    if not active.any():
        return rot_out, scale_out

    idx = np.flatnonzero(active)
    R = rot[idx]
    s2 = scale[idx] ** 2
    Ri = R_inc[idx]
    Si = S_inc[idx]

    A = Ri @ Si @ R                      # (K, 3, 3), Sigma' = A diag(s2) A^T
    R_p, S_p = polar_decompose(A)

    # B = S_p diag(s^2) S_p  (symmetric PSD)
    B = (S_p * s2[:, None, :]) @ np.swapaxes(S_p, -1, -2)
    lam, V = np.linalg.eigh(B)
    lam = np.clip(lam, _EPS, None)
    # det(V) is arbitrary (+-1); flip the smallest-eigenvalue column so
    # R_out = R_p V is a proper rotation (det = +1).
    V = V.copy()
    flip = np.linalg.det(V) < 0.0
    if flip.any():
        V[flip, :, 0] = -V[flip, :, 0]

    R_out = R_p @ V
    # SVD-based polar decomposition may return det(R_p) = -1 when the
    # deformation gradient is orientation-flipping; correct it too.
    det_p = np.linalg.det(R_p)
    bad = det_p < 0.0
    if bad.any():
        R_out[bad] = -R_out[bad]

    rot_out[idx] = R_out
    scale_out[idx] = np.sqrt(lam)
    return rot_out, scale_out


def apply_deformation(mu, rot, scale, corner_delta, corner_logR, corner_S,
                      valid_mask):
    """End-to-end Eq.13: corner increments -> position + covariance update.

    mu (N, 3) rest centers; rot (N, 3, 3); scale (N, 3);
    corner_delta (N, 8, 3), corner_logR (N, 8, 3), corner_S (N, 8, 3, 3)
    from deformation.corner_deformations; valid_mask (N, 8) bool.

    Returns (mu_new (N, 3), rot_out (N, 3, 3), scale_out (N, 3)) in
    KIRI-compatible form: rot_out is a proper rotation, scale_out is
    LINEAR. Fully-unbound Gaussians are unchanged.
    """
    from .deformation import gaussian_updates

    mu_new, R_inc, S_inc = gaussian_updates(
        mu, rot, scale, corner_delta, corner_logR, corner_S, valid_mask
    )
    rot_out, scale_out = propagate_covariance(rot, scale, R_inc, S_inc)
    return mu_new, rot_out, scale_out
