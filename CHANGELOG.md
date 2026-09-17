# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- CI configuration and automated release packaging (planned).

## [0.19.15] - 2026-09-17

### Changed

- `kiri_bridge` import in `unimags/blender.py` — fixes `NameError` when the
  add-on is imported before KIRI is registered.
- Removed the legacy `binding` module; all binding/deformation now goes
  through the UniMGS pipeline.
- Test suite updated accordingly; full unit + headless integration suite
  passes on Blender 5.2 LTS.

## [0.19.x] - 2026-09

### Added

- UniMGS **binding persistence**: bindings are saved as `.npz` next to the
  `.blend` and auto-reload after Blender restarts.
- `mesh vertices follow the proxy`: source-mesh vertices are written back
  together with the KIRI runtime cache, so the viewport sees deformation
  immediately.
- Global texture-rebuild flag for MESH-representation Gaussians.

### Fixed

- MESH-representation Gaussians now bind and deform without a
  `gaussian_count` property (centers read from mesh vertices).

## [0.17.0] - 2026-08

### Added

- UniMGS reproduction, first complete pass: Gaussian BBX corners,
  camera-ray binding (BVH), Eq.12 corner deformation, Eq.13 Gaussian
  update, covariance re-factorization, KIRI write-back.
- `UniMGS Deform` side panel with binding stats.
- Virtual surround cameras for back-face binding coverage.

## [0.16.x] - 2026-08

### Added

- `unimags/` pure-NumPy package skeleton (bbx, ray_binding, deformation,
  covariance, rotation) with unit tests.
- GPU compute-shader prototype for the vertex deformation stage.

## [0.10.0] - 2026-07

### Added

- Phase 7: linear subdivision, iterative vertex fitting (`NEAREST` /
  `PLANE` tangent projection), Laplacian smoothing, one-click
  Subdivide + Fit refine.
- Triangle-only OBJ/PLY export with world transform baked.

## [0.4.0] - 2026-06

### Added

- Phase 6: export aligned proxy mesh (OBJ / PLY).
- Phase 5: reset proxy transform from the set-time snapshot.

## [0.3.x] - 2026-06

### Added

- Phase 4: rough alignment (center + uniform scale on Gaussian bounds).
- Phase 3: live auto-follow of Gaussian deformation while the proxy is
  edited.

## [0.2.0] - 2026-06

### Added

- Phase 3: mesh → Gaussian-center deformation for KIRI EMPTY proxies.
- World-space bounds computation (center / size).

## [0.1.0] - 2026-05

### Added

- Initial add-on skeleton: `bl_info`, flat package layout, registration.
- Phase 2: designate a scene mesh as proxy + green wireframe display.
