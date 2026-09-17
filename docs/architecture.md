# Architecture

Gaussian Mesh Editor is split into two layers:

1. **`gaussian_mesh_editor/unimags/`** — a pure-NumPy implementation of the UniMGS deformation pipeline. No `bpy` import anywhere in this package, so every module is unit-testable in a plain Python interpreter.
2. **The Blender glue** (`__init__.py`, `operators.py`, `ui.py`, `props.py`, `kiri_bridge.py`, `unimags/blender.py`) — reads Gaussian data out of KIRI objects, feeds it through the pipeline, and writes the result back in a format the viewport renders.

```
                        ┌────────────────────────────────────────────┐
                        │              Blender (bpy)                 │
                        │                                            │
   operators / ui ─────▶│  unimags.blender                           │
                        │    ├─ _resolve_gaussian()  KIRI object     │
                        │    │      │   reads gaussian_data (local)  │
                        │    │      ▼                                │
                        │    ├─ to_world / to_local   matrix_world   │
                        │    │      ▼                                │
                        │    ├─ build_binding()  ──┐                 │
                        │    └─ apply_deformation()│                 │
                        └─────────┬───────────────┼─────────────────┘
                                  │               │
                        ┌─────────▼───────────────▼─────────────────┐
                        │         unimags/ (pure NumPy)             │
                        │                                           │
                        │  bbx           ray_binding  binding_data  │
                        │  deformation   covariance   rotation      │
                        │  gpu_pipeline  auto_follow                │
                        └───────────────────────────────────────────┘
```

## Module responsibilities

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `unimags/bbx.py` | Step 1: 8 oriented BBX corners per Gaussian, `Cᵢ = μ + k·R·(±s)` | `rotation` |
| `unimags/ray_binding.py` | Steps 3–4: camera rays → BVH raycast → closest face + barycentric coords | — |
| `unimags/binding_data.py` | Step 5: flat-array binding cache, `save_npz` / `load_npz` persistence | — |
| `unimags/deformation.py` | Steps 6–8: per-vertex offset → corner deformation (Eq. 12) → Gaussian update (Eq. 13) | `rotation` |
| `unimags/covariance.py` | Step 10: covariance propagation + re-factorization into `(R, s)` | `rotation` |
| `unimags/rotation.py` | SO(3) / quaternion helpers, vectorized, float64 | — |
| `unimags/gpu_pipeline.py` | Optional GLSL compute-shader implementation of the vertex stage | `deformation` |
| `unimags/auto_follow.py` | Live driver: depsgraph handler + poll timer → `apply_deformation` | `blender` |
| `unimags/blender.py` | bpy glue: object resolution, space conversions, binding lifetime, write-back | all of the above |

The add-on-level modules (`operators.py`, `ui.py`, `props.py`) are thin: they translate UI state into calls on `unimags.blender`. `kiri_bridge.py` is the only module that knows how to *detect* KIRI objects.

## KIRI interop (`kiri_bridge.py`)

KIRI is never force-imported. Detection is data-level:

- `EMPTY` object with `obj["is_gaussian_splat"]` set → KIRI GPU proxy, Gaussian data lives in the `gaussian_data` IDProperty (a `bytes` blob of `float32`, N×59).
- `MESH` object with an `f_dc_0` point attribute → KIRI PLY-imported Gaussian mesh, splat centers are the mesh vertices.

`gaussian_data` column layout (matches KIRI's shader):

| Columns | Meaning |
| --- | --- |
| 0:3 | position (local space) |
| 3:7 | quaternion `(w, x, y, z)` — local rotation |
| 7:10 | **linear** scale (no exp/log applied) |

The pipeline runs in **world space** (`matrix_world` maps local→world); results are written back in local space.

## Write-back strategy

Deformed Gaussians are written to KIRI's **runtime cache** by default (what the viewport renders from). Optionally, `Persist to Object` also rewrites the large `gaussian_data` IDProperty — slower, and large overwrites can be fragile on some Blender builds, so it is off by default.

## Extension vs legacy add-on

`blender_manifest.toml` makes the add-on installable as a Blender **extension** (4.2+). The same code installs as a legacy add-on because the package uses a flat layout. `kiri_bridge._find_kiri_prefix()` handles KIRI being loaded under either module prefix (`dgs_render_by_kiri_engine` vs `bl_ext.user_default.dgs_render_by_kiri_engine`).
