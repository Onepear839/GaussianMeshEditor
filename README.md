# Gaussian Mesh Editor

A Blender add-on that aligns, fits and deforms 3D Gaussian Splatting scenes through a **proxy mesh**, built as an independent companion module for [KIRI 3DGS Render](https://github.com/xiaolongyuan/KIRI3DGS) (the KIRI add-on itself is never modified).

It includes a faithful re-implementation of the **deformation module** of the paper:

> **UniMGS: Unifying Mesh and 3D Gaussian Splatting with Single-Pass Rasterization and Proxy-Based Deformation**
> Zeyu Xiao, Mingyang Sun, Yimin Cong, Lintao Wang, Dongliang Kou, Zhenyi Wu, Dingkang Yang, Peng Zhai, Zeyu Wang, Lihua Zhang
> AAAI 2026 · [arXiv:2601.19233](https://arxiv.org/html/2601.19233)

![Pipeline](docs/images/pipeline.svg)

## Features

- **Proxy-mesh workflow** — designate any scene mesh as a proxy, rough-align it onto the Gaussian cloud (center + uniform scale), then fine-tune with the native `G / R / S` tools.
- **Subdivide & Fit** — linear subdivision plus iterative vertex fitting onto the splat cloud (`NEAREST` snap or `PLANE` tangent-plane projection with Laplacian smoothing).
- **UniMGS reproduction** — Gaussian-centric bounding-box (BBX) ray binding and proxy-based deformation transfer (paper Eq. 12/13), with covariance re-factorization matching KIRI's `(quaternion, linear scale)` layout.
- **Live Auto-Follow** — while you drag the proxy mesh in Edit Mode, bound Gaussians follow in real time (depsgraph handler + poll timer).
- **Optional GPU acceleration** — the per-vertex deformation-gradient stage (the slowest part of Eq. 12) runs as a GLSL compute shader, with automatic CPU fallback.
- **KIRI integrated, not coupled** — Gaussian objects are detected by data (`is_gaussian_splat` / `f_dc_0`), so the add-on installs and runs even when KIRI is absent.
- **Pure-NumPy core** — the entire `unimags` package is `bpy`-free and unit-testable outside Blender.

## Requirements

| Component | Version |
| --- | --- |
| Blender | ≥ 5.1 (tested on 5.2 LTS) |
| Python | ≥ 3.10 (bundled with Blender) |
| NumPy | bundled with Blender |
| KIRI 3DGS Render | optional, ≥ 5.1 (needed for rendering/importing Gaussian data) |

## Installation

### As a Blender extension (recommended, Blender 4.2+)

1. Download the latest release zip (`gaussian_mesh_editor_<version>.zip`).
2. In Blender: **Edit → Preferences → Get Extensions → Install from Disk...** (or drag the zip onto the Extensions window).
3. Enable **Gaussian Mesh Editor** in the *Add-ons* tab.

### As a legacy add-on

1. Unzip `gaussian_mesh_editor/` into your Blender add-ons directory (see the [installation docs](docs/installation.md) for exact paths).
2. Enable **Gaussian Mesh Editor** in **Edit → Preferences → Add-ons**.

## Quick Start

1. **Import or create a 3DGS scene** with KIRI 3DGS Render (or add a plain mesh with an `f_dc_0` point attribute).
2. Select a mesh to act as the proxy, click **Set Active as Proxy Mesh**, then **Rough Align**.
3. Use **Subdivide** and **Fit to Gaussian** to conform the proxy to the splat surface.
4. Open the **UniMGS Deform** panel, click **Build UniMGS Binding**, then deform:

   - Edit the proxy mesh and click **Apply UniMGS Deformation**, or
   - Enable **Auto Follow (live)** and drag vertices in Edit Mode — Gaussians follow instantly.

See [docs/usage.md](docs/usage.md) for the full workflow, parameter reference and troubleshooting.

## How It Works

```
   Gaussian cloud (mu, R, S)
          │  kiri_bridge reads gaussian_data (world space)
          ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Step 1  BBX         8 oriented corners per Gaussian     │
   │ Step 2  Cameras     scene cameras + virtual surround    │
   │ Step 3  Rays        one ray per corner per camera       │
   │ Step 4  Bind        BVH raycast → closest face + bary   │
   │ Step 5  Cache       flat NumPy arrays, npz-persisted    │
   │ Step 6  Mesh deform per-vertex offset → Rⱼ Sⱼ (polar)   │
   │ Step 7  Corner      barycentric interpolation (Eq. 12)  │
   │ Step 8  Gaussian    average over 8 corners (Eq. 13)     │
   │ Step 9  Rot/S       exp(mean log Rᵢ), mean Sᵢ           │
   │ Step 10 Covariance  R′S′Σ(R′S′)ᵀ → re-factorized        │
   └─────────────────────────────────────────────────────────┘
          │  write-back to KIRI runtime cache / IDProperty
          ▼
   Viewport renders the deformed Gaussians
```

The math follows the paper: each Gaussian is spatially associated with the proxy mesh through its 8 BBX corners; deforming the mesh deforms each corner (Eq. 12), and each Gaussian is updated by averaging over its corners (Eq. 13). Full detail: [docs/unimags.md](docs/unimags.md), architecture: [docs/architecture.md](docs/architecture.md).

## Repository Layout

```
gaussian_mesh_editor/
├── __init__.py            add-on entry, registration, bl_info
├── blender_manifest.toml  Blender extension manifest
├── props.py               scene property group
├── operators.py           all operators (align / fit / UniMGS)
├── ui.py                  side-panel UI
├── kiri_bridge.py         KIRI interoperability (data-level detection)
├── bbox.py                world-space bounds
├── export.py              OBJ / PLY export with baked world transform
├── fit.py                 vertex fitting (NEAREST / PLANE)
├── metrics.py             alignment metrics
└── unimags/               pure-NumPy UniMGS reproduction (bpy-free)
    ├── bbx.py             Step 1 — Gaussian BBX corners
    ├── ray_binding.py     Steps 3–4 — camera rays + BVH binding
    ├── binding_data.py    Step 5 — binding cache (npz)
    ├── deformation.py     Steps 6–8 — Eq. 12/13 deformation transfer
    ├── covariance.py      Step 10 — covariance propagation
    ├── rotation.py        SO(3) / quaternion math (KIRI conventions)
    ├── gpu_pipeline.py    optional GLSL compute acceleration
    ├── auto_follow.py     live deformation driver
    └── blender.py         bpy glue: scene ⇄ pipeline ⇄ KIRI
docs/                      architecture, usage, limitations, …
tests/                     unit + headless integration tests
```

## Testing

The test suite is split into pure-NumPy unit tests (no Blender needed) and headless Blender integration tests.

```bash
# Pure-NumPy unit tests (plain Python)
python tests/test_unimags_bbx.py
python tests/test_unimags_binding_data.py
python tests/test_unimags_deformation.py
python tests/test_unimags_ray_binding.py
python tests/test_unimags_rotation.py

# Headless Blender integration tests
blender.exe -b --python tests/test_unimags_blender.py
blender.exe -b --python tests/test_unimags_bvh.py
blender.exe -b --python tests/test_auto_follow.py
blender.exe -b --python tests/test_binding.py
blender.exe -b --python tests/test_deform.py
blender.exe -b --python tests/test_visualize.py
blender.exe -b --python tests/test_phase6.py
blender.exe -b --python tests/test_phase7.py
```

Every test prints `ok - <description>` per assertion and exits non-zero on failure. See [docs/testing.md](docs/testing.md) for a summary of the verified behavior.

## Limitations

- This add-on re-implements only the **deformation** module of UniMGS; the paper's single-pass mesh+3DGS rasterizer is not included (that part lives in KIRI's renderer).
- Binding coverage depends on cameras; a single viewport camera only reaches the facing hemisphere, which is why virtual surround cameras are created by default (`Surround Cams`).
- Gaussian color, opacity and spherical-harmonic coefficients are preserved as-is; only position, rotation and scale are deformed.

See [docs/limitations.md](docs/limitations.md) for the complete list.

## Citation

If you use this add-on or its UniMGS reproduction in your work, please cite:

```bibtex
@inproceedings{xiao2026unimgs,
  title     = {UniMGS: Unifying Mesh and 3D {Gaussian} Splatting with Single-Pass
               Rasterization and Proxy-Based Deformation},
  author    = {Xiao, Zeyu and Sun, Mingyang and Cong, Yimin and Wang, Lintao
               and Kou, Dongliang and Wu, Zhenyi and Yang, Dingkang
               and Zhai, Peng and Wang, Zeyu and Zhang, Lihua},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {13},
  year      = {2026}
}
```

## License

GPL-2.0-or-later — see [LICENSE](LICENSE). The add-on is an independent work; it does not contain or modify KIRI 3DGS Render source code.

## Acknowledgements

- [KIRI 3DGS Render](https://github.com/Kiri-Innovation/3dgs-render-blender-addon) — Gaussian splatting rendering engine this add-on interoperates with.
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (Kerbl et al., SIGGRAPH 2023) — the underlying radiance representation.
