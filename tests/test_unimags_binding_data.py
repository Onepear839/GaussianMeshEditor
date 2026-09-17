"""Unit tests for unimags.binding_data — pure numpy.

Run: python test_unimags_binding_data.py
"""

import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "gaussian_mesh_editor"))

from unimags.binding_data import UniMGSBindingData  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print("  ok -", msg)


def rng():
    return np.random.default_rng(99)


r = rng()
N = 40
centers = r.normal(size=(N, 3))
# 5 fully bound (8/8), 10 partial, rest unbound
ids = np.full((N, 8), -1, dtype=np.int32)
ids[:5] = r.integers(0, 60, size=(5, 8))
for i in range(5, 15):  # partial: each row gets a random 1..7 valid corners
    k = int(r.integers(1, 8))
    ids[i, :k] = r.integers(0, 60, size=k)
bary = np.zeros((N, 8, 3))
bary[ids >= 0] = 1.0 / 3.0
dist = np.where(ids >= 0, r.uniform(0.0, 1.0, size=(N, 8)), np.inf)
hits = r.normal(size=(N, 8, 3))

print("UniMGSBindingData")

data = UniMGSBindingData(centers, ids, bary, dist, hits)
ok(np.array_equal(data.valid_mask, ids >= 0), "valid_mask derived from triangle_ids")

s = data.stats()
ok(s["gaussian_count"] == N, "gaussian_count")
ok(s["corner_slots"] == N * 8, "corner_slots")
ok(s["fully_bound"] == 5, "fully bound count == 5")
ok(s["unbound"] == N - 15, "unbound count")
ok(s["partial_bound"] == 10, "partial bound count")
ok(abs(s["corner_rate"] - (ids >= 0).sum() / (N * 8)) < 1e-12, "corner rate")

# all-invalid edge case
empty = UniMGSBindingData(np.zeros((3, 3)), np.full((3, 8), -1, dtype=np.int32),
                          np.zeros((3, 8, 3)), np.full((3, 8), np.inf),
                          np.zeros((3, 8, 3)))
se = empty.stats()
ok(se["fully_bound"] == 0 and se["partial_bound"] == 0 and se["unbound"] == 3
   and se["corner_rate"] == 0.0, "all-invalid stats")

# npz roundtrip
path = os.path.join(tempfile.mkdtemp(), "bind.npz")
data.save_npz(path)
loaded = UniMGSBindingData.load_npz(path)
ok(np.array_equal(loaded.triangle_ids, data.triangle_ids), "triangle_ids roundtrip")
ok(np.allclose(loaded.barycentric, data.barycentric), "barycentric roundtrip")
ok(np.array_equal(loaded.valid_mask, data.valid_mask), "valid_mask roundtrip")
ok(loaded.summary_line().startswith("Gaussians="), "summary line")

print("\n%d checks passed" % PASS)
