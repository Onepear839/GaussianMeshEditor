# Contributing

Thanks for considering a contribution to Gaussian Mesh Editor!

## Development setup

1. Blender 5.2 LTS (with KIRI 3DGS Render, if you test real Gaussian scenes).
2. A Python 3.10+ interpreter with NumPy for the pure-NumPy unit tests.
3. Clone the repository and link/copy `gaussian_mesh_editor/` into Blender's
   add-ons directory (see [docs/installation.md](docs/installation.md)).

## Workflow

1. **Open an issue** for the change you intend to make, so it can be
   discussed before code is written.
2. **Branch** from `main` with a descriptive name
   (`fix/ray-binding-edge-case`, `feat/scale-handles`, …).
3. **Code + tests.** Keep the `unimags/` package free of `bpy` imports —
   that is what makes the math unit-testable outside Blender. Every
   pipeline change should ship with a unit test, and behavioral changes
   should update the headless integration tests.
4. **Run the suite** (see [docs/testing.md](docs/testing.md)):

   ```bash
   python tests/test_unimags_bbx.py
   python tests/test_unimags_binding_data.py
   python tests/test_unimags_deformation.py
   python tests/test_unimags_ray_binding.py
   python tests/test_unimags_rotation.py
   blender.exe -b --python tests/test_unimags_blender.py
   # …plus the other integration tests in docs/testing.md
   ```

5. **Version bump** for user-visible changes (see below).
6. **Pull request** describing what changed and why, referencing the issue.

## Conventions

- **No `bpy` in `unimags/`.** This is a hard rule; the Blender glue belongs
  in `unimags/blender.py` or the add-on-level modules.
- **World space inside the pipeline.** Space conversions (local ⇄ world)
  happen at the `blender.py` boundary.
- **Match KIRI conventions.** Quaternion layout `(w, x, y, z)`, linear
  scale — see [docs/unimags.md](docs/unimags.md).
- **NumPy, not lists.** All hot loops are vectorized over the batch axis.
- **Comments explain *why*, not *what*.** Favor clear names over comments.

## Versioning

User-visible changes bump the version in **both** places:

- `bl_info["version"]` in `gaussian_mesh_editor/__init__.py`
- `version` in `gaussian_mesh_editor/blender_manifest.toml`

and add an entry to `CHANGELOG.md` under the new version.

- Patch (`0.19.15 → 0.19.16`): bug fixes, no new behavior.
- Minor (`0.19 → 0.20`): new features, behavior changes.
- Major (`0.x → 1.0`): API or format breaks.

## License

By contributing, you agree that your contributions are licensed under
the same terms as the project: **GPL-2.0-or-later** (see [LICENSE](LICENSE)).
