"""GPU-accelerated vertex deformation stage (Eq.12 vertex side).

Replaces deformation.vertex_deformations with a GLSL compute-shader
kernel when the user enables GPU acceleration. The seam is exact:

  (delta (V, 3), rot (V, 3, 3), shear (V, 3, 3))
      = gpu_pipeline.vertex_deformations_gpu(rest_verts, cur_verts, tris)
      = deformation.vertex_deformations(rest_verts, cur_verts, tris)

so apply_deformation can swap the implementation without touching the
corner / Gaussian stages.

Implementation notes (Blender 5.2 `gpu` module):
  - push constants and UBOs are not settable from Python, so the scalar
    parameters (V, K, TEX_W) travel in a 1x1 params texture.
  - `gpu.types.GPUTexture` has no `write()`; data is uploaded through
    the constructor `data=` argument.
  - SSBOs are gone; all arrays are packed into tiled 2D RGBA32F
    textures (imageLoad / imageStore), width TEX_W to stay under the
    per-dimension texture limit.
  - Any failure raises RuntimeError; the caller falls back to the CPU
    path.

The kernel re-implements the CPU math in float32:
  - area-weighted vertex normals come from the CPU (vectorized numpy,
    float64) so the GPU kernel stays pure one-ring math;
  - per vertex: G = sum e e^T + en en^T, H = sum ep e^T + enp en^T,
    D = H G^{-1}, then polar decomposition D = R S via the symmetric
    eigendecomposition of D^T D (Jacobi rotations; S = V sqrt(L) V^T,
    R = D S^{-1}). This matches np.linalg.svd's D = U S V^T ->
    R = U V^T up to float32 noise.
  - vertices with no neighbors, singular G, or singular D fall back to
    the identity (same policy as the CPU's LinAlgError branch).
"""

import numpy as np

_IDENT_TOL = 1e-9
_DEFAULT_MAX_K = 24  # one-ring neighbor cap; larger meshes fall back to CPU
_DEFAULT_TEX_W = 4096

_GLSL = r"""
void sym_eigen3(mat3 A, out vec3 eval, out mat3 evec) {
    /* Jacobi rotations on a symmetric 3x3; evec columns are eigenvectors. */
    mat3 V = mat3(1.0);
    for (int it = 0; it < 12; it++) {
        int p = 0, q = 1;
        float amax = abs(A[0][1]);
        if (abs(A[0][2]) > amax) { amax = abs(A[0][2]); p = 0; q = 2; }
        if (abs(A[1][2]) > amax) { amax = abs(A[1][2]); p = 1; q = 2; }
        if (amax < 1e-10) break;
        float app = A[p][p];
        float aqq = A[q][q];
        float apq = A[p][q];
        float tau = (aqq - app) / (2.0 * apq);
        float t = (tau >= 0.0) ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                               : -1.0 / (-tau + sqrt(1.0 + tau * tau));
        float c = 1.0 / sqrt(1.0 + t * t);
        float s = t * c;
        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        for (int k = 0; k < 3; k++) {
            if (k == p || k == q) continue;
            float akp = A[k][p];
            float akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[k][q] = s * akp + c * akq;
            A[p][k] = A[k][p];
            A[q][k] = A[k][q];
        }
        for (int k = 0; k < 3; k++) {
            float vkp = V[k][p];
            float vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
        }
    }
    eval = vec3(A[0][0], A[1][1], A[2][2]);
    evec = V;
}

mat3 outer3(vec3 a, vec3 b) {
    /* Computes b * a^T (NOT a * b^T): GLSL mat3 is column-major, so the
     * constructor arguments are columns, and mat3(b*a.x, b*a.y, b*a.z)
     * has column i = b*a[i]. Callers pass the operands in swapped order
     * to get the outer product they need (see the G/H accumulations). */
    return mat3(b * a.x, b * a.y, b * a.z);
}

void polar_decompose(mat3 D, out mat3 R, out mat3 S) {
    /* D = R S with R rotation, S symmetric PSD, via eig(D^T D). */
    mat3 M = transpose(D) * D;
    vec3 lam;
    mat3 V;
    sym_eigen3(M, lam, V);
    lam = max(lam, vec3(0.0));
    vec3 sq = sqrt(lam);
    vec3 isq;
    for (int i = 0; i < 3; i++) {
        isq[i] = (sq[i] > 1e-20) ? 1.0 / sq[i] : 0.0;
    }
    /* V diag(sq): GLSL V * sq is a matrix-vector product, so scale the
     * columns explicitly (V[0], V[1], V[2] are the columns of V). */
    S = mat3(V[0] * sq.x, V[1] * sq.y, V[2] * sq.z);
    S = S * transpose(V);
    mat3 Sinv = mat3(V[0] * isq.x, V[1] * isq.y, V[2] * isq.z);
    Sinv = Sinv * transpose(V);
    R = D * Sinv;
}

/* Bindless images: Blender 5.x types images as image2D_bindless, which
 * cannot be passed to a function taking image2D (and imageLoad has no
 * image2D overload). Inline the fetch through a macro instead. */
#define FETCHV(img, lin, tex_w) imageLoad((img), ivec2((lin) % (tex_w), (lin) / (tex_w)))

void main() {
    vec4 P = imageLoad(img_params, ivec2(0, 0));
    int V = int(P.x);
    int K = int(P.y);
    int TEX_W = int(P.z);
    int j = int(gl_GlobalInvocationID.x);
    if (j >= V) return;

    vec3 r0 = FETCHV(img_rest, j, TEX_W).xyz;
    vec3 c0 = FETCHV(img_cur, j, TEX_W).xyz;
    vec3 nr = FETCHV(img_nrest, j, TEX_W).xyz;
    vec3 nc = FETCHV(img_ncur, j, TEX_W).xyz;

    mat3 G = mat3(0.0);
    mat3 H = mat3(0.0);
    float cnt = 0.0;
    float wsum = 0.0;
    float wpsum = 0.0;
    for (int k = 0; k < K; k++) {
        int nb = int(FETCHV(img_neigh, j * K + k, TEX_W).x);
        if (nb < 0) continue;
        vec3 e = FETCHV(img_rest, nb, TEX_W).xyz - r0;
        vec3 ep = FETCHV(img_cur, nb, TEX_W).xyz - c0;
        G += outer3(e, e);
        H += outer3(e, ep);
        wsum += dot(e, e);
        wpsum += dot(ep, ep);
        cnt += 1.0;
    }

    mat3 R = mat3(1.0);
    mat3 S = mat3(1.0);
    bool ok = (cnt > 0.5);
    if (ok) {
        float w = wsum / cnt;
        float wp = wpsum / cnt;
        vec3 en = sqrt(w) * nr;
        vec3 enp = sqrt(wp) * nc;
        G += outer3(en, en);
        H += outer3(en, enp);
        float nG = max(max(max(abs(G[0][0]), abs(G[0][1])), abs(G[0][2])),
                       max(max(abs(G[1][0]), abs(G[1][1])), abs(G[1][2])));
        nG = max(nG, max(max(abs(G[2][0]), abs(G[2][1])), abs(G[2][2])));
        ok = (nG > 1e-30) && (abs(determinant(G)) > 1e-10 * nG * nG * nG);
        if (ok) {
            mat3 D = H * inverse(G);
            float nD = max(max(max(abs(D[0][0]), abs(D[0][1])), abs(D[0][2])),
                           max(max(abs(D[1][0]), abs(D[1][1])), abs(D[1][2])));
            nD = max(nD, max(max(abs(D[2][0]), abs(D[2][1])), abs(D[2][2])));
            ok = (nD > 1e-30) && (abs(determinant(D)) > 1e-10 * nD * nD * nD);
            if (ok) {
                polar_decompose(D, R, S);
            }
        }
    }

    /* out R (9) then S (9), 3 RGBA texels each starting at j*3. Column-
     * major element order on the wire; Python rebuilds row-major. */
    imageStore(img_outR, ivec2((j * 3 + 0) % TEX_W, (j * 3 + 0) / TEX_W),
               vec4(R[0][0], R[1][0], R[2][0], R[0][1]));
    imageStore(img_outR, ivec2((j * 3 + 1) % TEX_W, (j * 3 + 1) / TEX_W),
               vec4(R[1][1], R[2][1], R[0][2], R[1][2]));
    imageStore(img_outR, ivec2((j * 3 + 2) % TEX_W, (j * 3 + 2) / TEX_W),
               vec4(R[2][2], 0.0, 0.0, 0.0));
    imageStore(img_outS, ivec2((j * 3 + 0) % TEX_W, (j * 3 + 0) / TEX_W),
               vec4(S[0][0], S[1][0], S[2][0], S[0][1]));
    imageStore(img_outS, ivec2((j * 3 + 1) % TEX_W, (j * 3 + 1) / TEX_W),
               vec4(S[1][1], S[2][1], S[0][2], S[1][2]));
    imageStore(img_outS, ivec2((j * 3 + 2) % TEX_W, (j * 3 + 2) / TEX_W),
               vec4(S[2][2], 0.0, 0.0, 0.0));
}
"""


def _build_neighbor_table(tris, V, max_k):
    """(V * K) float32 flat padded neighbor table; -1 padding.

    Raises RuntimeError when any vertex's one-ring degree exceeds max_k.
    """
    tris = np.asarray(tris, dtype=np.int64)
    a = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    b = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    V64 = np.int64(V)
    keys = np.concatenate([a * V64 + b, b * V64 + a])
    uniq = np.unique(keys)
    src = uniq // V64
    dst = uniq % V64
    cnt = np.bincount(src, minlength=V)
    max_deg = int(cnt.max()) if len(cnt) else 0
    if max_deg > max_k:
        raise RuntimeError(
            "mesh vertex degree {} exceeds GPU cap {} (fall back to CPU)".format(
                max_deg, max_k))
    offsets = np.zeros(V + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(cnt)
    rank = np.arange(len(dst), dtype=np.int64) - offsets[src]
    slot = src * max_k + rank
    table = np.full(V * max_k, -1.0, dtype=np.float32)
    table[slot] = dst.astype(np.float32)
    return table


def _vertex_normals(verts, tris):
    """Area-weighted vertex normals, vectorized (matches deformation.py)."""
    v = np.asarray(verts, dtype=np.float64)
    t = np.asarray(tris, dtype=np.int64)
    fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    n = np.zeros((len(v), 3), dtype=np.float64)
    np.add.at(n, t.ravel(), np.repeat(fn, 3, axis=0))
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return n / norms


# ------------------------------------------------------------------ shader
_shader_cache = {}


def _get_shader(K, tex_w):
    key = (K, tex_w)
    shader = _shader_cache.get(key)
    if shader is not None:
        return shader
    import gpu
    from gpu.types import GPUShaderCreateInfo

    info = GPUShaderCreateInfo()
    info.compute_source(_GLSL)
    info.image(0, "RGBA32F", "FLOAT_2D", "img_params", qualifiers={"READ"})
    info.image(1, "RGBA32F", "FLOAT_2D", "img_rest", qualifiers={"READ"})
    info.image(2, "RGBA32F", "FLOAT_2D", "img_cur", qualifiers={"READ"})
    info.image(3, "RGBA32F", "FLOAT_2D", "img_nrest", qualifiers={"READ"})
    info.image(4, "RGBA32F", "FLOAT_2D", "img_ncur", qualifiers={"READ"})
    info.image(5, "RGBA32F", "FLOAT_2D", "img_neigh", qualifiers={"READ"})
    info.image(6, "RGBA32F", "FLOAT_2D", "img_outR", qualifiers={"WRITE"})
    info.image(7, "RGBA32F", "FLOAT_2D", "img_outS", qualifiers={"WRITE"})
    info.local_group_size(64, 1, 1)
    shader = gpu.shader.create_from_info(info)
    _shader_cache[key] = shader
    return shader


def _make_tex(data, tex_w):
    """RGBA32F tiled 2D texture from a flat float32 array (zero padded so
    the buffer fills the whole texture)."""
    from gpu.types import GPUTexture, Buffer

    total = len(data)  # already a multiple of 4
    h = max(1, (total // 4 + tex_w - 1) // tex_w)
    n_tex = tex_w * h * 4
    if n_tex > total:
        data = np.concatenate([data, np.zeros(n_tex - total, dtype=np.float32)])
    return GPUTexture(size=(tex_w, h), format="RGBA32F",
                      data=Buffer("FLOAT", len(data), data.tolist()))


def vertex_deformations_gpu(rest_verts, cur_verts, tris, max_k=_DEFAULT_MAX_K,
                            tex_w=_DEFAULT_TEX_W):
    """GPU one-ring deformation gradient; same contract as
    deformation.vertex_deformations. Raises RuntimeError on any failure.
    """
    import gpu
    from gpu.types import GPUTexture, Buffer

    rest_verts = np.asarray(rest_verts, dtype=np.float64)
    cur_verts = np.asarray(cur_verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    V = len(rest_verts)
    if V == 0 or len(tris) == 0:
        raise RuntimeError("empty mesh for GPU path")

    # float32 precision: recenter so edge vectors keep relative precision.
    center = rest_verts.mean(axis=0)
    rest_c = (rest_verts - center).astype(np.float32)
    cur_c = (cur_verts - center).astype(np.float32)
    delta = cur_verts - rest_verts  # exact, computed in float64

    n_rest = _vertex_normals(rest_verts, tris).astype(np.float32)
    n_cur = _vertex_normals(cur_verts, tris).astype(np.float32)
    table = _build_neighbor_table(tris, V, max_k)
    K = max_k

    # Stride-4 packing: a RGBA32F texel holds 4 floats, so 3-float vertex
    # data must be padded to 4 floats per vertex (texel j -> vertex j).
    # Otherwise vertex j's data straddles texels and fetchv(j) reads the
    # wrong texel for every j > 0.
    def pack_verts(arr):
        out = np.zeros((len(arr), 4), dtype=np.float32)
        out[:, :3] = arr
        return np.ascontiguousarray(out.ravel())

    def pack_table(tbl):
        out = np.zeros((len(tbl), 4), dtype=np.float32)
        out[:, 0] = tbl
        return np.ascontiguousarray(out.ravel())

    tex_params = GPUTexture(size=(1, 1), format="RGBA32F",
                            data=Buffer("FLOAT", 4,
                                        [float(V), float(K), float(tex_w), 0.0]))
    tex_rest = _make_tex(pack_verts(rest_c), tex_w)
    tex_cur = _make_tex(pack_verts(cur_c), tex_w)
    tex_nr = _make_tex(pack_verts(n_rest), tex_w)
    tex_nc = _make_tex(pack_verts(n_cur), tex_w)
    tex_ne = _make_tex(pack_table(table), tex_w)
    tex_outR = _make_tex(np.zeros(V * 12, dtype=np.float32), tex_w)
    tex_outS = _make_tex(np.zeros(V * 12, dtype=np.float32), tex_w)

    shader = _get_shader(K, tex_w)
    shader.bind()
    shader.image("img_params", tex_params)
    shader.image("img_rest", tex_rest)
    shader.image("img_cur", tex_cur)
    shader.image("img_nrest", tex_nr)
    shader.image("img_ncur", tex_nc)
    shader.image("img_neigh", tex_ne)
    shader.image("img_outR", tex_outR)
    shader.image("img_outS", tex_outS)
    gpu.compute.dispatch(shader, (V + 63) // 64, 1, 1)

    def read_tex(tex):
        lst = tex.read().to_list()
        flat = []
        for row in lst:
            for px in row:
                flat.extend(px)
        return np.asarray(flat, dtype=np.float32)

    r_flat = read_tex(tex_outR)[: V * 9]
    s_flat = read_tex(tex_outS)[: V * 9]

    # Wire order is column-major; remap to numpy row-major (V, 3, 3).
    cm = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=np.int64)
    rot = np.empty((V, 9), dtype=np.float64)
    shear = np.empty((V, 9), dtype=np.float64)
    rot[:, :] = r_flat.reshape(V, 9)[:, cm]
    shear[:, :] = s_flat.reshape(V, 9)[:, cm]
    rot = rot.reshape(V, 3, 3)
    shear = shear.reshape(V, 3, 3)

    # Exactness guard: untouched vertices must keep the exact identity
    # (matches the CPU path bit-for-bit where nothing moved).
    dR = np.max(np.abs(rot - np.eye(3)), axis=(-2, -1))
    dS = np.max(np.abs(shear - np.eye(3)), axis=(-2, -1))
    idle = (dR < _IDENT_TOL) & (dS < _IDENT_TOL)
    if idle.any():
        rot[idle] = np.eye(3)
        shear[idle] = np.eye(3)
    return delta, rot, shear
