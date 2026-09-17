"""Step 5: UniMGS binding cache — per-Gaussian 8-corner face bindings.

Storage is flat NumPy arrays (never one Blender object per corner):
  triangle_ids  (N, 8) int32   face index per corner, -1 = invalid
  barycentric   (N, 8, 3)      (b0,b1,b2) of the hit point per corner
  distances     (N, 8)         hit-point-to-Gaussian-center distance
  hit_points    (N, 8, 3)      intersection point per corner
  gaussian_centers (N, 3)      mu at bind time (for stats and debug)
  rest_verts    (V, 3)         proxy mesh world vertices at bind time
  rest_tris     (M, 3) int32   proxy mesh topology at bind time

valid_mask is derived (triangle_ids >= 0) but kept explicit for clarity.
The rest mesh is the reference for deformation gradients (Eq.12/13).
"""

import numpy as np


class UniMGSBindingData:
    def __init__(self, gaussian_centers, triangle_ids, barycentric,
                 distances, hit_points, rest_verts=None, rest_tris=None):
        self.gaussian_centers = np.asarray(gaussian_centers, dtype=np.float64)
        self.triangle_ids = np.asarray(triangle_ids, dtype=np.int32)
        self.barycentric = np.asarray(barycentric, dtype=np.float64)
        self.distances = np.asarray(distances, dtype=np.float64)
        self.hit_points = np.asarray(hit_points, dtype=np.float64)
        self.valid_mask = self.triangle_ids >= 0
        self.rest_verts = (
            np.zeros((0, 3), dtype=np.float64) if rest_verts is None
            else np.asarray(rest_verts, dtype=np.float64)
        )
        self.rest_tris = (
            np.zeros((0, 3), dtype=np.int32) if rest_tris is None
            else np.asarray(rest_tris, dtype=np.int32)
        )

    # ------------------------------------------------------------- stats
    def stats(self):
        """dict: counts and rates of fully / partially / unbound."""
        n = len(self.gaussian_centers)
        per = self.valid_mask.sum(axis=1)  # valid corners per Gaussian
        fully = int((per == 8).sum())
        partial = int(((per > 0) & (per < 8)).sum())
        unbound = int((per == 0).sum())
        corners = int(self.valid_mask.sum())
        return {
            "gaussian_count": n,
            "corner_slots": n * 8,
            "valid_corners": corners,
            "corner_rate": corners / (n * 8) if n else 0.0,
            "fully_bound": fully,
            "partial_bound": partial,
            "unbound": unbound,
            "fully_rate": fully / n if n else 0.0,
            "mean_corners_per_gaussian": per.mean() if n else 0.0,
        }

    # ------------------------------------------------------------- I/O
    def save_npz(self, path):
        np.savez(
            path,
            gaussian_centers=self.gaussian_centers,
            triangle_ids=self.triangle_ids,
            barycentric=self.barycentric,
            distances=self.distances,
            hit_points=self.hit_points,
            rest_verts=self.rest_verts,
            rest_tris=self.rest_tris,
        )

    @classmethod
    def load_npz(cls, path):
        z = np.load(path)
        return cls(
            z["gaussian_centers"],
            z["triangle_ids"],
            z["barycentric"],
            z["distances"],
            z["hit_points"],
            z["rest_verts"] if "rest_verts" in z else None,
            z["rest_tris"] if "rest_tris" in z else None,
        )

    # ------------------------------------------------------------- misc
    def summary_line(self):
        s = self.stats()
        return (
            "Gaussians={} fully={} partial={} unbound={} "
            "corner_rate={:.1%}".format(
                s["gaussian_count"],
                s["fully_bound"],
                s["partial_bound"],
                s["unbound"],
                s["corner_rate"],
            )
        )
