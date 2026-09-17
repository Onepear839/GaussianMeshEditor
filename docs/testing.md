# Testing

The test suite is organized in two tiers:

- **Pure-NumPy unit tests** — run with a plain Python interpreter; no Blender involved. They cover the math modules of `unimags/`.
- **Headless Blender integration tests** — run with `blender -b --python …`; they exercise the full pipeline inside Blender, including KIRI-style object read/write and the live Auto-Follow driver.

Every test prints `  ok - <description>` per passing assertion and raises (exiting non-zero) on the first failure.

## Unit tests (no Blender)

```bash
python tests/test_unimags_bbx.py          # BBX corners vs brute force
python tests/test_unimags_binding_data.py # flat-array cache + npz round-trip
python tests/test_unimags_deformation.py  # Eq.12/13 transfer + covariance
python tests/test_unimags_ray_binding.py  # ray binder vs Moller-Trumbore oracle
python tests/test_unimags_rotation.py     # quat<->matrix round-trips
```

## Integration tests (headless Blender)

```bash
blender.exe -b --python tests/test_unimags_blender.py   # full UniMGS pipeline
blender.exe -b --python tests/test_unimags_bvh.py       # BVH caster == numpy oracle
blender.exe -b --python tests/test_auto_follow.py       # live auto-follow
blender.exe -b --python tests/test_binding.py           # legacy binding
blender.exe -b --python tests/test_deform.py            # mesh->Gaussian center deform
blender.exe -b --python tests/test_visualize.py         # visualization + save/load
blender.exe -b --python tests/test_phase6.py            # export correctness
blender.exe -b --python tests/test_phase7.py            # subdivide + fit + refine
```

> On Windows, use the full path to `blender.exe` (e.g. `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`).

## Verified behavior (as of v0.19.15)

All tests below pass on Blender 5.2 LTS — 9 test files, exit code 0:

**Unit tests (pure NumPy, 108 assertions)** — `test_unimags_bbx` (7
checks), `test_unimags_binding_data` (12 checks), 
`test_unimags_deformation` (48 assertions), `test_unimags_ray_binding`
(23 checks), `test_unimags_rotation` (18 checks).

**Integration tests (headless Blender, 32 checks + 9 PASS lines)** —
`test_unimags_blender` (20 checks, full UniMGS pipeline),
`test_unimags_bvh` (12 checks, BVH == numpy oracle), `test_phase6`
(export correctness), `test_phase7` (subdivide + fit + refine +
triangulated export + reset).

Highlights from the UniMGS suite:

- BBX corners are on the Gaussian's own oriented axes, not world-aligned.
- Binding builds from the source mesh; MESH-representation Gaussians bind without a `gaussian_count` property (centers read from mesh vertices).
- A rigid translation of the proxy moves every bound Gaussian by the same vector and preserves covariance.
- A second apply follows a further deformation (snapshot updates).
- `mesh vertices follow the proxy` — source-mesh vertices and the KIRI runtime cache are both updated; the viewport sees the move.
- Topology changes raise a clear `ValueError`; clearing a binding invalidates Apply; bindings can be rebuilt after clear.
- Auto-Follow propagates mesh edits through the binding, and leaves Gaussians untouched when disabled.
- GPU path (compute shader) matches the CPU path output; `gpu_accel` falls back automatically.
- Export writes triangle-only OBJ/PLY with world transform baked.

## CI notes

There is no bundled CI configuration. If you want to run the integration tier in CI, install Blender 5.2 LTS on the runner and invoke `blender -b --python` per test; the suite is deterministic and does not need a display.
