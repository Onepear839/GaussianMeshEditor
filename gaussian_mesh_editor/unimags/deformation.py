"""Step 6-9: deformation transfer — mesh -> BBX corners -> Gaussian.

Implements the paper's Eq.12 (corner deformation by barycentric
interpolation of per-vertex motion) and Eq.13 (Gaussian update by
averaging over the 8 BBX corners):

  vertex:   motion = offset D^j; D^j = R^j S^j (polar decomposition)
  corner:   Delta_i  = u D1 + v D2 + w D3
            log(R_i) = u log(R1) + v log(R2) + w log(R3)
            S_i      = u S1 + v S2 + w S3
  Gaussian: R' = exp( mean_i log(R_i) )
            S' = mean_i S_i
            mu' = mu + mean_i Delta_i
            Sigma' = R' S' Sigma (R' S')^T      (see covariance.py)

Per-vertex D^j is the one-ring deformation gradient (least squares over
the incident edges, regularized with the area-weighted vertex normal
pseudo-edge), decomposed via polar decomposition. This is an ACAP
approximation: the transfer equations (Eq.12/13) are faithful to the
paper; the vertex-side solver is the interactive substitute for ACAP's
full nonlinear optimization.
"""

import numpy as np

from .rotation import polar_decompose, so3_exp, so3_log


def _vertex_neighbors(tris, vertex_count):
    """Sorted neighbor vertex index list per vertex, from a triangle soup."""
    neigh = [set() for _ in range(vertex_count)]
    for a, b, c in tris:
        neigh[a].update((b, c))
        neigh[b].update((a, c))
        neigh[c].update((a, b))
    return [sorted(s) for s in neigh]


def _vertex_normals(verts, tris):
    """Area-weighted vertex normals (Blender-style)."""
    v = verts
    n = np.zeros_like(v)
    for a, b, c in tris:
        fn = np.cross(v[b] - v[a], v[c] - v[a])
        n[a] += fn
        n[b] += fn
        n[c] += fn
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return n / norms


def vertex_deformations(rest_verts, cur_verts, tris):
    """Per-vertex motion (delta, rotation, shear).

    rest_verts / cur_verts: (V, 3) world positions (same topology).
    tris: (M, 3) int32 vertex indices.

    Returns (delta (V, 3), rot (V, 3, 3), shear (V, 3, 3)).
    For each vertex the one-ring least-squares deformation gradient
    D = H G^{-1} is polar-decomposed: D = R S (ACAP approximation).
    Isolated vertices get identity rotation/shear.
    """
    rest_verts = np.asarray(rest_verts, dtype=np.float64)
    cur_verts = np.asarray(cur_verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int32)
    V = len(rest_verts)

    delta = cur_verts - rest_verts
    rot = np.tile(np.eye(3), (V, 1, 1))
    shear = np.tile(np.eye(3), (V, 1, 1))

    n_rest = _vertex_normals(rest_verts, tris)
    n_cur = _vertex_normals(cur_verts, tris)
    neigh = _vertex_neighbors(tris, V)

    for j in range(V):
        nb = neigh[j]
        if not nb:
            continue
        idx = np.asarray(nb, dtype=np.int32)
        e = rest_verts[idx] - rest_verts[j]  # (K, 3)
        ep = cur_verts[idx] - cur_verts[j]
        w = float(np.mean(np.einsum("ki,ki->k", e, e)))
        wp = float(np.mean(np.einsum("ki,ki->k", ep, ep)))
        # normal pseudo-edge (length sqrt(mean squared edge length)) keeps
        # the rank-2 one-ring fan full-rank; its length must follow the
        # deformed edge scale, otherwise a uniform scale is not recovered
        en = np.sqrt(w) * n_rest[j]
        enp = np.sqrt(wp) * n_cur[j]
        G = np.einsum("ki,kj->ij", e, e) + np.outer(en, en)
        H = np.einsum("ki,kj->ij", ep, e) + np.outer(enp, en)
        try:
            D = np.linalg.solve(G, H.T).T  # H G^{-1}
        except np.linalg.LinAlgError:
            continue
        Rj, Sj = polar_decompose(D[None])
        rot[j] = Rj[0]
        shear[j] = Sj[0]
    return delta, rot, shear


def corner_deformations(tri_ids, bary, tris, delta, rot, shear):
    """Eq.12: barycentric interpolation of vertex motion per BBX corner.

    tri_ids: (N, 8) int32 triangle index per corner (-1 invalid);
    bary: (N, 8, 3) (b0, b1, b2) of the corner's hit point;
    tris: (M, 3) int32 vertex indices;
    delta (V, 3), rot (V, 3, 3), shear (V, 3, 3): per-vertex motion.

    Returns (corner_delta (N, 8, 3), corner_logR (N, 8, 3) rotation
    vectors, corner_S (N, 8, 3, 3)). Rotations are interpolated in the
    SO(3) log domain (never as Euler angles); invalid corners are zero.
    """
    tri_ids = np.asarray(tri_ids, dtype=np.int32)
    bary = np.asarray(bary, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int32)
    N, C = tri_ids.shape

    safe = np.where(tri_ids >= 0, tri_ids, 0)
    tv = tris[safe]  # (N, C, 3) vertex ids

    d = delta[tv]  # (N, C, 3, 3)
    corner_delta = np.einsum("nck,nckd->ncd", bary, d)

    R = rot[tv]  # (N, C, 3, 3, 3)
    logR = so3_log(R)  # (N, C, 3, 3)
    corner_logR = np.einsum("nck,nckd->ncd", bary, logR)

    S = shear[tv]  # (N, C, 3, 3, 3)
    corner_S = np.einsum("nck,nckij->ncij", bary, S)

    valid = tri_ids >= 0
    corner_delta[~valid] = 0.0
    corner_logR[~valid] = 0.0
    corner_S[~valid] = 0.0
    return corner_delta, corner_logR, corner_S


def gaussian_updates(mu, rot, scale, corner_delta, corner_logR, corner_S,
                     valid_mask):
    """Eq.13: average the 8 corner deformations into a Gaussian update.

    mu (N, 3) rest centers; rot (N, 3, 3) rest rotations;
    scale (N, 3) LINEAR rest scales (unused here, kept for symmetry);
    corner_delta (N, 8, 3), corner_logR (N, 8, 3),
    corner_S (N, 8, 3, 3); valid_mask (N, 8) bool.

    Returns (mu_new (N, 3), R_inc (N, 3, 3), S_inc (N, 3, 3)) with
    R_inc = exp(mean log R_i), S_inc = mean S_i (valid corners only).
    R_inc / S_inc are the *increment* applied to the rest covariance in
    Eq.13 (Sigma' = R' S' Sigma (R' S')^T), NOT the final rotation -
    covariance.propagate_covariance re-factorizes them. A Gaussian with
    no valid corner keeps its rest position and gets an identity
    increment so the covariance propagator leaves it untouched.
    """
    mu = np.asarray(mu, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    cnt = valid_mask.sum(axis=1, keepdims=True)  # (N, 1)
    total = np.maximum(cnt, 1.0)
    wd = np.where(valid_mask, 1.0, 0.0)[..., None]  # (N, 8, 1)
    wS = wd[..., None]  # (N, 8, 1, 1)

    mean_delta = (corner_delta * wd).sum(axis=1) / total
    mean_logR = (corner_logR * wd).sum(axis=1) / total
    mean_S = (corner_S * wS).sum(axis=1) / total[..., None]

    mu_new = mu + mean_delta
    R_inc = so3_exp(mean_logR)
    S_inc = mean_S

    unchanged = cnt[:, 0] == 0
    if unchanged.any():
        mu_new[unchanged] = mu[unchanged]
        R_inc[unchanged] = np.eye(3)
        S_inc[unchanged] = np.eye(3)
    return mu_new, R_inc, S_inc
