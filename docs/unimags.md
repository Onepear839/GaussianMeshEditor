# UniMGS Reproduction

This add-on re-implements the **deformation module** of:

> **UniMGS: Unifying Mesh and 3D Gaussian Splatting with Single-Pass Rasterization and Proxy-Based Deformation**
> Zeyu Xiao, Mingyang Sun, Yimin Cong, Lintao Wang, Dongliang Kou, Zhenyi Wu, Dingkang Yang, Peng Zhai, Zeyu Wang, Lihua Zhang
> AAAI 2026 · [arXiv:2601.19233](https://arxiv.org/html/2601.19233)

The paper contributes two things: (1) a single-pass rasterizer that blends mesh and Gaussian fragments together, and (2) a **Gaussian-centric binding strategy** that deforms 3DGS through a proxy mesh. This project reproduces (2) inside Blender, feeding the deformed result to KIRI 3DGS Render's existing rasterizer. The paper's single-pass renderer itself is **not** reproduced — KIRI already renders meshes and Gaussians in the same viewport.

## Pipeline (paper order)

```
   mu, R, S  ──▶  BBX (8 corners)  ──▶  camera ray casting  ──▶  per-corner
   face binding  ──▶  mesh deformation  ──▶  corner deformation (Eq. 12)  ──▶
   Gaussian update (Eq. 13)  ──▶  KIRI cache write-back
```

### Step 1 — Gaussian-centric BBX (`unimags/bbx.py`)

For each Gaussian with mean `μ`, rotation `R` and (linear) scale `s`, the bounding box is the oriented box:

```
Cᵢ = μ + k · R · (σᵢ ⊙ s),   σᵢ ∈ {±1}³,   i = 0..7
```

`k` is a half-edge multiplier. The paper does not pin down the exact box extent, so `k` is exposed as `BBX k` (default `1.0`).

### Steps 3–4 — Camera-ray binding (`unimags/ray_binding.py`)

> "each camera casts 8 rays toward the corners of a Gaussian's BBX … For each ray, we retain the face closest to the Gaussian."

Every corner of every Gaussian gets one ray from every camera. Rays are cast against the proxy mesh with a BVH (`mathutils.bvhtree.BVHTree` in Blender; a pure-NumPy Moller–Trumbore caster exists as the test oracle). For each corner we keep the **closest** hit across all cameras (face index + barycentric coordinates + hit distance).

**Camera coverage caveat.** A ray from a single viewpoint always hits the front-facing hemisphere first, so back-facing proxy faces can never drive deformation. The add-on therefore creates `Surround Cams` (default 6) virtual camera origins evenly spread on a sphere around the proxy, in addition to the scene cameras.

### Steps 6–8 — Deformation transfer (`unimags/deformation.py`)

Given the current proxy vertices vs. the rest pose captured at bind time:

- **Per-vertex motion** — each vertex's displacement gradient is decomposed via polar decomposition into a rotation `Rⱼ` and a symmetric `Sⱼ`.
- **Corner deformation (Eq. 12)** — a bound corner with barycentric weights `(u, v, w)` on triangle `(1, 2, 3)` deforms by

```
Δᵢ  = u·D₁ + v·D₂ + w·D₃
log Rᵢ = u·log R₁ + v·log R₂ + w·log R₃
Sᵢ  = u·S₁ + v·S₂ + w·S₃
```

- **Gaussian update (Eq. 13)** — average over the 8 corners:

```
R′ = exp( meanᵢ log Rᵢ )
S′ = meanᵢ Sᵢ
μ′ = μ + meanᵢ Δᵢ
```

### Step 10 — Covariance propagation (`unimags/covariance.py`)

```
Σ′ = R′·S′·Σ·(R′·S′)ᵀ,   Σ = R·diag(s²)·Rᵀ
```

The propagated covariance must be re-factorized into `Σ′ = R_out · diag(s_out²) · R_outᵀ` so it fits KIRI's `(quaternion, linear scale)` layout. This is done without ever forming `Σ` explicitly: one combined 3×3 `A = R′S′R`, polar decomposition, then a symmetric eigendecomposition of `B = S_p·diag(s²)·S_pᵀ`.

### Rotation conventions (`unimags/rotation.py`)

Quaternion layout is `(w, x, y, z)` and scale is **linear** — both match KIRI's `vert.glsl` exactly, so a quaternion written back to `gaussian_data` renders the same rotation it encodes. All math is vectorized over the batch axis in float64.

## Fidelity notes

- **Preserved as-is:** Gaussian color, opacity, spherical-harmonic coefficients. Only position, rotation and scale are deformed.
- **Deterministic:** with `GPU Acceleration` off, the whole pipeline is deterministic NumPy; the GPU path is an exact swap of the vertex stage (same outputs, verified in `tests/test_unimags_blender.py`).
- **Binding persistence:** bindings are cached in memory per scene and saved as an `.npz` next to the `.blend` file, so they survive Blender restarts and are re-validated against the current mesh topology at apply time.
