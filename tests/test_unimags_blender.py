"""Headless integration tests for the UniMGS pipeline in Blender.

Run from Blender 5.2:
  blender.exe -b --python test_unimags_blender.py

Covers:
  1. Build UniMGS binding (BBX corners -> camera rays -> face ids).
  2. Apply deformation: a rigid translation of the proxy moves every
     bound Gaussian by the same vector and preserves covariance.
  3. A second apply follows a further deformation (snapshot updates).
  4. Topology changes raise a clear ValueError.
  5. Clear binding invalidates Apply.
"""

import os
import sys

import bmesh
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import gaussian_mesh_editor as gme

gme.register()

import bpy

from gaussian_mesh_editor.unimags import blender as unimags_blender

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print("  ok -", msg)


def make_cube(name, half, loc=(0, 0, 0)):
    v = [(dx * half, dy * half, dz * half) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    faces = [
        (0, 2, 3, 1),  # x = -1
        (4, 5, 7, 6),  # x = +1
        (0, 1, 5, 4),  # y = -1
        (2, 6, 7, 3),  # y = +1
        (0, 4, 6, 2),  # z = -1
        (1, 3, 7, 5),  # z = +1
    ]
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], faces)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    return obj


def make_gaussian_empty(name, centers, scales):
    """KIRI-style EMPTY proxy: is_gaussian_splat + gaussian_data bytes."""
    n = len(centers)
    arr = np.zeros((n, 59), dtype=np.float32)
    arr[:, :3] = centers
    arr[:, 3] = 1.0  # w = 1 (identity quaternion)
    arr[:, 7:10] = scales
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.empty_display_type = "SPHERE"
    obj["is_gaussian_splat"] = True
    obj["gaussian_count"] = n
    obj["sh_degree"] = 0
    obj["gaussian_data"] = arr.tobytes()
    return obj


def add_camera(name, loc):
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    return cam


def make_cloud(name, pts):
    """KIRI MESH-representation Gaussian: point-cloud mesh with f_dc_0."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in pts], [], [])
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    me.attributes.new(name="f_dc_0", type="FLOAT_COLOR", domain="POINT")
    return obj


def gauss_data(obj):
    cache = getattr(bpy, "gaussian_object_cache", None)
    if isinstance(cache, dict) and obj.name in cache:
        return np.asarray(cache[obj.name]["gaussian_data"], dtype=np.float32)
    return np.frombuffer(
        bytearray(obj["gaussian_data"]), dtype=np.float32
    ).reshape(int(obj["gaussian_count"]), 59)


def main():
    scene = bpy.context.scene
    props = scene.gme_props

    # Clean slate
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    cube = make_cube("ProxyCube", 1.0)
    g1 = make_gaussian_empty(
        "GaussA",
        np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.0]], dtype=np.float32),
        np.array([[0.1, 0.1, 0.1], [0.15, 0.1, 0.12]], dtype=np.float32),
    )
    add_camera("Cam", (0.0, 0.0, 6.0))

    props.proxy_obj = cube
    props.gaussian_obj = g1

    # --- 1. build binding ------------------------------------------------
    data = unimags_blender.build_binding(scene, k=1.0)
    ok(data is not None, "build_binding returns data")
    s = data.stats()
    ok(s["gaussian_count"] == 2, "binding covers 2 Gaussians")
    ok(s["valid_corners"] > 0, "some corners bound: {}/{}".format(
        s["valid_corners"], s["corner_slots"]))
    ok(unimags_blender.get_binding(scene) is data, "binding stored for the scene")

    before = gauss_data(g1)
    mu_before = before[:, :3].copy()

    # --- 2. rigid translation of the proxy -------------------------------
    for v in cube.data.vertices:
        v.co.x += 0.5
    cube.data.update()

    data2, moved, max_disp = unimags_blender.apply_deformation(scene, persist=False)
    ok(moved >= 1, "some Gaussians moved ({})".format(moved))
    after = gauss_data(g1)
    mu_after = after[:, :3]
    bound = data.valid_mask.any(axis=1)
    exp = mu_before[bound] + np.array([0.5, 0.0, 0.0])
    ok(np.allclose(mu_after[bound], exp, atol=1e-5),
       "bound Gaussians translate with the mesh (max err {:.2e})".format(
           np.abs(mu_after[bound] - exp).max()))
    ok(np.allclose(after[:, 3:10], before[:, 3:10], atol=1e-6),
       "translation preserves quaternion and scale exactly")

    # --- 3. second apply follows a further deformation --------------------
    for v in cube.data.vertices:
        v.co.z -= 0.25
    cube.data.update()
    _, moved2, _ = unimags_blender.apply_deformation(scene, persist=False)
    after2 = gauss_data(g1)
    exp2 = mu_before[bound] + np.array([0.5, 0.0, -0.25])
    ok(np.allclose(after2[bound, :3], exp2, atol=1e-5),
       "second apply accumulates the new displacement")
    ok(moved2 >= 1, "second apply still moves Gaussians")

    # --- 4. topology change is rejected -----------------------------------
    me = cube.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[0]], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.update()
    try:
        unimags_blender.apply_deformation(scene, persist=False)
        ok(False, "topology change should raise")
    except ValueError as exc:
        ok("topology" in str(exc).lower(), "topology change raises ValueError")

    # --- 5. clear binding invalidates apply -------------------------------
    unimags_blender.clear_binding(scene)
    ok(unimags_blender.get_binding(scene) is None, "clear_binding removes the cache")
    try:
        unimags_blender.apply_deformation(scene, persist=False)
        ok(False, "apply without binding should raise")
    except ValueError as exc:
        ok("binding" in str(exc).lower(), "apply without binding raises ValueError")

    # --- 6. rebuild works after clearing ----------------------------------
    data3 = unimags_blender.build_binding(scene, k=1.0)
    ok(data3 is not None and len(data3.gaussian_centers) == 2,
       "binding can be rebuilt after clear")

    # --- 7. MESH-representation Gaussian (point-cloud, like 'scene.001') ---
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    cube2 = make_cube("ProxyCube2", 1.0)
    cloud = make_cloud("scene.001", np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.0]], dtype=np.float64))
    add_camera("Cam2", (0.0, 0.0, 6.0))
    props.proxy_obj = cube2
    props.gaussian_obj = cloud

    data4 = unimags_blender.build_binding(scene, k=1.0)
    ok(data4 is not None and len(data4.gaussian_centers) == 2,
       "MESH-representation Gaussian binds without gaussian_count")
    ok(np.allclose(data4.gaussian_centers,
                   np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.0]]), atol=1e-6),
       "centers read from mesh vertices")

    for v in cube2.data.vertices:
        v.co.x += 0.5
    cube2.data.update()
    _, moved4, _ = unimags_blender.apply_deformation(scene, persist=False)
    ok(moved4 >= 1, "MESH Gaussian deforms")
    ok(np.allclose(
        np.array([list(v.co) for v in cloud.data.vertices]),
        np.array([[0.5, 0.0, 0.0], [0.8, 0.2, 0.0]]), atol=1e-5),
       "mesh vertices follow the proxy")
    cached = gauss_data(cloud)
    ok(np.allclose(cached[:, :3],
                   np.array([[0.5, 0.0, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32),
                   atol=1e-5),
       "KIRI runtime cache updated for MESH Gaussian (viewport sees the move)")
    ok(getattr(bpy, "gaussian_global_needs_update", False) is True,
       "global texture rebuild flagged for MESH Gaussian")

    # --- 8. persist=True also rewrites gaussian_data on MESH objects -------
    _, _, _ = unimags_blender.apply_deformation(scene, persist=True)
    ok("gaussian_data" in cloud and int(cloud["gaussian_count"]) == 2,
       "persist writes gaussian_data + gaussian_count on the MESH object")

    # --- 9. auto-follow drives the UniMGS deformation live -----------------
    from gaussian_mesh_editor.unimags import auto_follow

    props.auto_follow_enabled = True
    for v in cube2.data.vertices:
        v.co.y += 0.3
    cube2.data.update()
    auto_follow._run_if_changed(scene)
    af = gauss_data(cloud)
    ok(np.allclose(af[:, :3],
                   np.array([[0.5, 0.3, 0.0], [0.8, 0.5, 0.0]], dtype=np.float32),
                   atol=1e-4),
       "auto-follow propagates mesh edits through the UniMGS binding")

    # disabled -> edits are ignored
    props.auto_follow_enabled = False
    for v in cube2.data.vertices:
        v.co.z += 0.5
    cube2.data.update()
    auto_follow._run_if_changed(scene)
    af2 = gauss_data(cloud)
    ok(np.allclose(af2[:, :3], af[:, :3], atol=1e-6),
       "auto-follow disabled leaves Gaussians untouched")

    # --- 10. source mesh + GPU proxy: write-back lands on the render target --
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    cube3 = make_cube("ProxyCube3", 1.0)
    src = make_cloud("scene.001",
                     np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.0]], dtype=np.float64))
    src["gaussian_source_uuid"] = "uuid-src-001"
    proxy = bpy.data.objects.new("scene.001Splat_Proxy", None)
    bpy.context.scene.collection.objects.link(proxy)
    proxy["is_gaussian_splat"] = True
    proxy["source_mesh_uuid"] = "uuid-src-001"
    proxy["source_mesh_name"] = "scene.001"
    add_camera("Cam3", (0.0, 0.0, 6.0))
    props.proxy_obj = cube3
    props.gaussian_obj = src

    data5 = unimags_blender.build_binding(scene, k=1.0)
    ok(data5 is not None, "binding builds from the source mesh")

    for v in cube3.data.vertices:
        v.co.x += 0.5
    cube3.data.update()
    _, moved5, _ = unimags_blender.apply_deformation(scene, persist=False)
    ok(moved5 >= 1, "deformation runs on source+proxy pair")
    ok(np.allclose(np.array([list(v.co) for v in src.data.vertices]),
                   np.array([[0.5, 0.0, 0.0], [0.8, 0.2, 0.0]]), atol=1e-5),
       "source mesh vertices follow")
    proxy_cache = gauss_data(proxy)
    ok(np.allclose(proxy_cache[:, :3],
                   np.array([[0.5, 0.0, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32),
                   atol=1e-4),
       "GPU proxy cache updated - KIRI viewport sees the move")
    _, _, _ = unimags_blender.apply_deformation(scene, persist=True)
    ok("gaussian_data" in proxy and int(proxy["gaussian_count"]) == 2,
       "persist writes gaussian_data on the GPU proxy")

    print("\nPASS: {} assertions".format(PASS))


main()
