# Usage

The add-on adds a **Gaussian Mesh Editor** tab to the 3D Viewport side panel (toggle with `N`).

```
Gaussian Mesh Editor            ← GME_PT_alignment (tab)
  Proxy Mesh
    [Set Active as Proxy Mesh]  [Reset Transform]
  Alignment Target
    [x] Alignment Target        Gaussian: <object>
  Rough Alignment
    [Rough Align]
  Bounds (World Space)
    [Print Bounds]
  Subdivide & Fit
    Levels / Faces  [Subdivide]
    Mode / k / Iterations / Strength / Smoothing
    [Fit to Gaussian]  [Subdivide + Fit]
  Export
    [Export Aligned Mesh]

UniMGS Deform                   ← GME_PT_unimags (same tab, below)
  Build Binding (Paper)
    BBX k / Surround Cams  [Build UniMGS Binding]
  Binding Stats                 (after binding exists)
    Gaussians / Corner Rate / Fully Bound / Partial Bound / Unbound
    [Clear Binding]
  Deformation (Eq.12/13)
    [x] Auto Follow (live)   [x] GPU Acceleration
    [x] Persist to Object (slow)
    [Apply UniMGS Deformation]
```

## 1. Prepare a proxy mesh

1. Import (KIRI) or create a Gaussian scene. The *Alignment* panel shows how many Gaussian objects were detected.
2. Create or pick any mesh that roughly covers the Gaussian cloud (a box or a low-poly cage is a good start).
3. Select the mesh → **Set Active as Proxy Mesh**. The object keeps its name; its transform is snapshotted so **Reset Transform** can restore it.

## 2. Rough alignment

Click **Rough Align**. The add-on computes world-space bounds of both the Gaussian cloud and the proxy, centers the proxy on the Gaussian center, and applies a uniform scale so the bounds match. Fine-tune afterwards with the native `G` / `R` / `S` transforms. **Print Bounds** shows the numbers.

> Tip: re-run **Rough Align** after any manual transform if you want to re-sync.

## 3. Subdivide & fit

- **Subdivide** — linear subdivision of the proxy (faces × 4 per level). Keeps the shape, adds vertices for the fit to work with.
- **Fit to Gaussian** — iterative snap of proxy vertices onto the splat cloud:
  - `Mode = NEAREST` — pull each vertex toward the nearest splat center (aggressive; good for final snapping).
  - `Mode = PLANE` — fit a tangent plane through the `k` nearest splats and slide the vertex onto it (smoother; good for initial passes).
  - `Iterations` / `Strength` / `Smoothing` control how far and how smoothly vertices move per round.
- **Subdivide + Fit** — one click that subdivides then fits.

Repeat until the proxy conforms to the surface you want to deform.

## 4. Export (optional)

**Export Aligned Mesh** writes OBJ / PLY with the world transform baked in and every polygon fan-triangulated, so downstream engines receive triangle-only, world-space geometry.

## 5. UniMGS binding

Open the **UniMGS Deform** section:

1. **BBX k** — half-edge multiplier of the Gaussian bounding box (paper default `1.0`).
2. **Surround Cams** — how many virtual camera origins to spread around the proxy (default `6`; raise it if binding coverage is low, see [unimags.md](unimags.md) for why).
3. Click **Build UniMGS Binding**.

The binding is cached in memory and saved as `.npz` next to your `.blend`, so it survives restarts. The panel then shows binding stats (Gaussian count, corner rate, fully/partially bound counts).

> If **Corner Rate** is low, increase `Surround Cams` or make sure the proxy mesh is closed and covers the Gaussians.

## 6. Deform

Two ways:

- **Manual:** edit the proxy mesh (move vertices / transform it), then click **Apply UniMGS Deformation**. Bound Gaussians move with the mesh.
- **Auto Follow (live):** tick **Auto Follow**. Then simply drag proxy vertices in Edit Mode — a timer (~20×/s) plus a depsgraph handler watch the mesh and push deformation continuously, so Gaussians follow *while you drag*.

Options:

- **GPU Acceleration** — runs the per-vertex deformation-gradient stage on the GPU via a GLSL compute shader; automatically falls back to CPU. Headless renders always use CPU.
- **Persist to Object (slow)** — also rewrites the `gaussian_data` IDProperty instead of only the runtime cache the viewport renders from. Keep it off unless you need the data saved into the object.

## Common workflows

| Goal | Recipe |
| --- | --- |
| Deform a scene with a cage | Set proxy → Rough Align → Subdivide (2–3 levels) → Fit (PLANE) → Build UniMGS Binding → Auto Follow ON → drag |
| Snap proxy exactly onto splats | Fit with `NEAREST`, `Strength 1.0`, `Iterations 1` |
| Keep topology light | Use few subdivision levels; binding cost scales with triangle count |
| Export a deformed proxy | Apply UniMGS Deformation → Export Aligned Mesh |

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| "no Gaussian object found" | The active scene has no `is_gaussian_splat` EMPTY and no `f_dc_0` mesh. Import with KIRI first, or add an `f_dc_0` point attribute for testing. |
| Low corner rate in binding stats | Not enough cameras: increase `Surround Cams`; ensure the proxy covers the Gaussians. |
| Apply raises "binding is invalid / topology changed" | You changed the proxy topology (subdivided) after binding. Rebuild the binding. |
| Gaussians don't follow in Edit Mode | Auto Follow polls the mesh while you drag; if it is off, click **Apply UniMGS Deformation** manually. |
| Large `Persist to Object` writes fail on Blender 5.2 | Known fragility with big IDProperty overwrites — keep persistence off and rely on the runtime cache. |
