"""SO(3) / quaternion math for the UniMGS reproduction.

Conventions match KIRI's vert.glsl exactly:
  - quaternion layout is (w, x, y, z); quat_to_matrix uses the 3DGS
    build_rotation formula KIRI's shader uses, so a quaternion written
    back to gaussian_data renders the same rotation it encodes here.
  - scale is LINEAR in gaussian_data (KIRI stores exp'd scale and the
    shader never re-applies exp), so no log/exp appears on scale here.

Everything is vectorized over a leading batch axis and operates on
float64 internally.
"""

import numpy as np

_EPS = 1e-12


def quat_to_matrix(q):
    """(..., 4) quaternion (w, x, y, z) -> (..., 3, 3) rotation matrix.

    Matrix layout is the one in KIRI's computeCov3D / quaternionToMatrix.
    """
    q = np.asarray(q, dtype=np.float64)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[..., 0, 1] = 2.0 * (x * y - w * z)
    out[..., 0, 2] = 2.0 * (x * z + w * y)
    out[..., 1, 0] = 2.0 * (x * y + w * z)
    out[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[..., 1, 2] = 2.0 * (y * z - w * x)
    out[..., 2, 0] = 2.0 * (x * z - w * y)
    out[..., 2, 1] = 2.0 * (y * z + w * x)
    out[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


def matrix_to_quat(R):
    """(..., 3, 3) rotation matrix -> (..., 4) quaternion (w, x, y, z).

    Shepperd-style branch selection so 180-degree rotations (w == 0)
    stay accurate. Recovers the input quaternion up to the global sign;
    the returned quaternion is normalized and has w >= 0.
    """
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]
    r11 = R[..., 1, 1]
    r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    tr = r00 + r11 + r22

    vals = np.stack(
        [
            1.0 + tr,
            1.0 + r00 - r11 - r22,
            1.0 - r00 + r11 - r22,
            1.0 - r00 - r11 + r22,
        ],
        axis=-1,
    )
    m = np.argmax(vals, axis=-1)
    s = 0.5 * np.sqrt(np.take_along_axis(vals, m[..., None], axis=-1)[..., 0])

    q = np.zeros(shp + (4,), dtype=np.float64)
    sel = m == 0
    if sel.any():
        w = s[sel]
        q[sel, 0] = w
        q[sel, 1] = (r21 - r12)[sel] / (4.0 * w)
        q[sel, 2] = (r02 - r20)[sel] / (4.0 * w)
        q[sel, 3] = (r10 - r01)[sel] / (4.0 * w)
    sel = m == 1
    if sel.any():
        x = s[sel]
        q[sel, 1] = x
        q[sel, 0] = (r21 - r12)[sel] / (4.0 * x)
        q[sel, 2] = (r10 + r01)[sel] / (4.0 * x)
        q[sel, 3] = (r20 + r02)[sel] / (4.0 * x)
    sel = m == 2
    if sel.any():
        y = s[sel]
        q[sel, 2] = y
        q[sel, 0] = (r02 - r20)[sel] / (4.0 * y)
        q[sel, 1] = (r10 + r01)[sel] / (4.0 * y)
        q[sel, 3] = (r21 + r12)[sel] / (4.0 * y)
    sel = m == 3
    if sel.any():
        z = s[sel]
        q[sel, 3] = z
        q[sel, 0] = (r10 - r01)[sel] / (4.0 * z)
        q[sel, 1] = (r20 + r02)[sel] / (4.0 * z)
        q[sel, 2] = (r21 + r12)[sel] / (4.0 * z)

    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm[norm < _EPS] = 1.0
    q = q / norm
    flip = q[..., 0] < 0.0
    q[flip] = -q[flip]
    return q


def skew(v):
    """(..., 3) vector -> (..., 3, 3) skew-symmetric matrix."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def so3_exp(omega):
    """Rodrigues: (..., 3) rotation vector -> (..., 3, 3) rotation matrix.

    omega = angle * axis. Series-safe at omega == 0.
    """
    omega = np.asarray(omega, dtype=np.float64)
    theta = np.linalg.norm(omega, axis=-1)
    nz = theta > _EPS
    t = np.maximum(theta, _EPS)
    a = np.where(nz, np.sin(theta) / t, 1.0)
    b = np.where(nz, (1.0 - np.cos(theta)) / (t * t), 0.5)
    K = skew(omega)
    eye = np.eye(3, dtype=np.float64)
    return eye + a[..., None, None] * K + b[..., None, None] * (K @ K)


def so3_log(R):
    """(..., 3, 3) rotation matrix -> (..., 3) rotation vector.

    log(R) = theta / (2 sin theta) * (R - R^T) as a vector; series-safe
    at theta == 0. Used for barycentric interpolation of rotations in
    the paper's Eq.12 (rotations must never be averaged as Euler angles).
    """
    R = np.asarray(R, dtype=np.float64)
    cos_t = np.clip((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)
    sv = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    small = theta < 1e-8
    # small-angle series: log(R) ~= sv/2 * (1 + theta^2/6)
    factor = np.where(
        small,
        0.5 * (1.0 + theta * theta / 6.0),
        theta / (2.0 * np.maximum(np.sin(theta), _EPS)),
    )
    return factor[..., None] * sv


def polar_decompose(D):
    """(..., 3, 3) -> (R, S) with D = R @ S.

    R is a rotation matrix, S is symmetric positive semi-definite
    (rotation * shear, per the paper: 'D^j decomposed into a rotation
    matrix R^j and a shear matrix S^j via polar decomposition').
    Computed via SVD: D = U Sigma V^T -> R = U V^T, S = V Sigma V^T.
    """
    D = np.asarray(D, dtype=np.float64)
    U, s_vals, Vt = np.linalg.svd(D)
    R = U @ Vt
    V = np.swapaxes(Vt, -1, -2)
    S = (V * s_vals[..., None, :]) @ Vt
    return R, S
