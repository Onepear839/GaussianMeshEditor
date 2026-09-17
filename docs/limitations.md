# Limitations

This project is a faithful reproduction of UniMGS's **deformation module** as a Blender add-on. The following are known limitations, by design or by the state of the implementation.

## Scope of the reproduction

- **Single-pass rasterizer not included.** The paper's key rendering contribution — rasterizing mesh and 3DGS fragments together in one anti-aliased pass — is *not* reproduced here. Deformed Gaussians are handed back to KIRI 3DGS Render's existing renderer, which already draws meshes and Gaussians in the same viewport. Occlusion/transparency between the proxy mesh and the splats therefore follows KIRI's behavior, not UniMGS's fused pass.
- **Deformation only.** Only position, rotation and scale are propagated to Gaussians. Color, opacity and spherical-harmonic coefficients are read but never modified.

## Algorithmic caveats

- **Camera coverage drives binding quality.** A ray from one viewpoint always hits the front-facing hemisphere first, so back faces can never be bound by a single camera. The add-on compensates with virtual surround cameras (`Surround Cams`), but coverage is still an approximation of the paper's full training-camera set. If your scene has extreme self-occlusion, corners behind deep concavities may stay unbound.
- **BBX half-edge `k` is a free parameter.** The paper does not define the exact box extent; `BBX k` defaults to `1.0` and is exposed for tuning. Different `k` values change which faces a Gaussian binds to.
- **Topology changes invalidate the binding.** Subdividing or re-meshing the proxy after binding raises a clear error — you must rebuild the binding. This is intentional (the binding stores per-corner triangle ids + barycentric coordinates against the rest topology).
- **Linear scale, KIRI conventions.** The pipeline assumes KIRI's `gaussian_data` layout (quaternion `(w,x,y,z)`, linear scale). Data from other renderers must be converted first.

## Blender / KIRI platform notes

- **Minimum Blender 5.1.** The add-on uses APIs and behaviors tested on 5.2 LTS; older versions are not supported.
- **GPU path needs a GPU context.** The compute-shader acceleration requires a live viewport GPU context; headless renders silently use the CPU path. The CPU path is the reference implementation.
- **`Persist to Object` can be fragile.** Rewriting the large `gaussian_data` IDProperty is slow, and Blender 5.2 can corrupt large overwrites. It is off by default; the runtime cache is what the viewport renders from.
- **KIRI is optional at install time.** Detection is data-level, so the add-on loads without KIRI — but you need KIRI (or data in its format) for real Gaussian scenes.

## Performance

- Binding cost scales linearly with `Gaussians × corners × cameras`; raycast is `O(log M)` per corner per camera thanks to the BVH.
- The per-vertex deformation-gradient stage is the slowest part of Eq. 12 in Python; use **GPU Acceleration** for interactive scenes with many vertices.
- Auto-Follow runs a ~20 Hz poll timer in Edit Mode; on very dense proxies this adds CPU load while dragging.
