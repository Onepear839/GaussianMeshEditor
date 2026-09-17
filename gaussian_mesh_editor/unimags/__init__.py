"""UniMGS reproduction: Gaussian-centric proxy-mesh binding + deformation.

Faithful re-implementation of the deformation module of
"UniMGS: Unifying Mesh and 3D Gaussian Splatting with Single-Pass
Rasterization and Proxy-Based Deformation" (arXiv:2601.19233).

Pipeline (per the paper):
  mu, R, S -> BBX (8 corners) -> camera ray casting -> per-corner
  face binding -> mesh deformation -> corner deformation (Eq.12) ->
  Gaussian update (Eq.13) -> KIRI cache write-back.

This package is pure NumPy (no bpy) so every module is unit-testable
outside Blender. All geometry is in world space; the caller converts
to/from the KIRI object's local space.
"""

__version__ = (0, 1, 0)
