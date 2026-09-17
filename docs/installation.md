# Installation

## Requirements

- **Blender ≥ 5.1** (developed and tested on Blender 5.2 LTS)
- **KIRI 3DGS Render ≥ 5.1** — *optional but recommended*; needed to import/`render Gaussian scenes. The add-on also works standalone for testing, because Gaussian objects are detected by data, not by KIRI's presence.
- NumPy (bundled with Blender — no manual install needed)

## Option A — Blender extension (recommended)

Blender 4.2+ ships an extension system; this add-on provides a `blender_manifest.toml` so it installs like any other extension.

1. Download the release zip, e.g. `gaussian_mesh_editor_0.19.15.zip`.
2. Start Blender → **Edit → Preferences → Get Extensions** (or **Extensions** from the splash screen).
3. Click the dropdown arrow next to *Repositories* → **Install from Disk…**
4. Select the zip. **Gaussian Mesh Editor** appears under *Installed Extensions*.
5. Enable the checkbox. The add-on's side panel shows up in the 3D Viewport.

## Option B — legacy add-on

If you prefer a classic add-on install:

1. Unzip `gaussian_mesh_editor/` so you have a folder named `gaussian_mesh_editor` containing `__init__.py`.
2. Copy that folder into your add-ons directory:

   | Platform | Typical path |
   | --- | --- |
   | Windows | `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\` |
   | macOS | `~/Library/Application Support/Blender/<version>/scripts/addons/` |
   | Linux | `~/.config/blender/<version>/scripts/addons/` |
3. **Edit → Preferences → Add-ons**, search "Gaussian Mesh Editor", enable it.

## Install KIRI (for real Gaussian scenes)

KIRI 3DGS Render installs the same way (it is itself a Blender add-on). After both are enabled:

- A KIRI-imported scene appears as an `EMPTY` object carrying `is_gaussian_splat` and a `gaussian_data` blob, or as a `MESH` object with an `f_dc_0` point attribute.
- Gaussian Mesh Editor's *Alignment* panel lists detected Gaussian objects automatically.

## Verifying the install

Run a headless smoke test:

```bash
blender.exe -b --python-expr "import gaussian_mesh_editor; gaussian_mesh_editor.register(); print('GME OK')"
```

Expect `GME OK` on the console. See [testing.md](testing.md) for the full suite.

## Upgrading

- **Extension install:** download the new zip and install from disk again; Blender replaces the old version.
- **Legacy install:** replace the `gaussian_mesh_editor` folder. Bindings are stored as `.npz` next to your `.blend` file (not inside the add-on folder), so they survive upgrades.
