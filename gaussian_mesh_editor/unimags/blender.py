"""Blender integration: scene -> UniMGS binding -> deformation -> KIRI.

The pure-numpy modules in this package stay importable outside Blender;
this module is the only bpy-dependent glue:

  - read KIRI gaussian_data (positions / quats / scales) from the object
    or the runtime cache, in world space
  - BlenderCameraSource feeds the camera-ray binder from scene cameras
  - build_binding / apply_deformation drive the full pipeline:
    BBX -> ray binding -> Eq.12 corner deformation -> Eq.13 update ->
    write-back to the KIRI cache / IDProperty (local space).

Conventions (match KIRI's gaussian_data columns):
  0:3 positions (local), 3:7 quaternion (w, x, y, z, local rotation),
  7:10 LINEAR scale (local). The object matrix_world maps to world.
"""

import os
import tempfile
import time

import numpy as np

import bpy

from .. import kiri_bridge
from .bbx import get_gaussian_bbx
from .binding_data import UniMGSBindingData
from .covariance import propagate_covariance
from .deformation import (
    corner_deformations,
    gaussian_updates,
    vertex_deformations,
)
from .ray_binding import BVHCaster, CameraSource, bind_corners
from .rotation import matrix_to_quat, quat_to_matrix

_bindings = {}  # scene key -> UniMGSBindingData (in-memory cache)
_disk_missing = set()  # scene keys confirmed absent on disk this session

IDENT_TOL = 1e-9


def _scene_key(scene):
    return scene.get("gme_unimags_key")


def _ensure_scene_key(scene):
    key = scene.get("gme_unimags_key")
    if key is None:
        import uuid

        key = uuid.uuid4().hex
        scene["gme_unimags_key"] = key
    return key


def _binding_path(scene):
    """(path, key) of the persisted binding npz for this scene, or (None, key).

    Saved next to the .blend (needs a saved file); temp dir as a fallback.
    """
    key = _scene_key(scene)
    if key is None:
        return None, None
    base = bpy.path.abspath("//")
    if not base:
        base = tempfile.gettempdir()
    return os.path.join(base, "gme_unimags_binding_{}.npz".format(key)), key


def get_binding(scene):
    """In-memory binding, falling back to the npz persisted by set_binding.

    The npz survives Blender restarts; a loaded binding is cached in memory
    and re-validated against the current mesh topology at apply time.
    """
    key = _scene_key(scene)
    if key is None:
        return None
    data = _bindings.get(key)
    if data is not None:
        return data
    if key in _disk_missing:
        return None
    path, _ = _binding_path(scene)
    if path and os.path.exists(path):
        try:
            data = UniMGSBindingData.load_npz(path)
        except Exception as exc:
            print("[GME-debug] binding load failed ({}): {}".format(path, exc))
        else:
            _bindings[key] = data
            print("[GME-debug] binding loaded from disk: {}".format(path))
            return data
    _disk_missing.add(key)
    return None


def set_binding(scene, data):
    key = _ensure_scene_key(scene)
    _bindings[key] = data
    _disk_missing.discard(key)
    try:
        path, _ = _binding_path(scene)
        if path:
            data.save_npz(path)
    except Exception as exc:
        print("[GME-debug] binding save failed: {}".format(exc))


def clear_binding(scene):
    key = _scene_key(scene)
    if key is not None:
        _bindings.pop(key, None)
        _disk_missing.add(key)
        path, _ = _binding_path(scene)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def _resolve_gaussian(scene):
    """A scene object that actually carries readable Gaussian data.

    KIRI renders is_gaussian_splat GPU proxies that are REBUILT from an
    EVALUATED source MESH on every refresh. Only that source mesh is an
    authoritative gaussian object - a deformation written anywhere else
    is silently reverted by KIRI's next refresh. The proxy's source MESH
    is therefore preferred over any user-picked gaussian object; the
    existing preference order is kept as fallback when no rendered chain
    exists (e.g. pure PLY imports).
    """
    props = scene.gme_props
    for obj in scene.objects:
        if obj.type != "EMPTY" or not obj.get("is_gaussian_splat", False):
            continue
        src = _resolve_source_mesh(obj)
        if src is None:
            continue
        try:
            gaussian_table(src)
        except ValueError:
            continue
        props.gaussian_obj = src
        return src
    if props.gaussian_obj is not None and kiri_bridge.is_gaussian_object(props.gaussian_obj):
        try:
            gaussian_table(props.gaussian_obj)
        except ValueError:
            pass
        else:
            return props.gaussian_obj
    for g in kiri_bridge.find_gaussian_objects(scene):
        try:
            gaussian_table(g)
        except ValueError:
            continue
        props.gaussian_obj = g
        return g
    gaussians = kiri_bridge.find_gaussian_objects(scene)
    if gaussians:
        props.gaussian_obj = gaussians[0]
        return gaussians[0]
    return None


def _resolve_render_target(g_obj):
    """The object KIRI actually renders (may differ from the binding source).

    Blender-object sources exist as two objects: the source MESH (carries
    gaussian_source_uuid + f_dc_* attributes) and a GPU proxy EMPTY
    (is_gaussian_splat, source_mesh_uuid = the source's gaussian_source_uuid)
    that KIRI renders from. Deformations must land on the proxy, or the
    viewport keeps showing the pre-deformation splats.
    """
    if g_obj.type == "MESH":
        src_uuid = g_obj.get("gaussian_source_uuid", "")
        for obj in bpy.data.objects:
            if obj is g_obj or obj.type != "EMPTY":
                continue
            if not obj.get("is_gaussian_splat", False):
                continue
            if (obj.get("source_mesh_uuid") == src_uuid or
                    obj.get("source_mesh_name") == g_obj.name):
                return obj
    return g_obj


def _resolve_source_mesh(g_obj):
    """The source MESH whose vertices carry the splat centers.

    KIRI GPU proxies (EMPTY, is_gaussian_splat) are DERIVED from a source
    MESH: on every refresh KIRI rebuilds the proxy from the EVALUATED
    source mesh, so the deformation MUST also land on that mesh, or any
    KIRI refresh silently reverts the splats to the rest pose.
    """
    if g_obj.type == "MESH":
        return g_obj
    src_uuid = g_obj.get("source_mesh_uuid", "")
    src_name = g_obj.get("source_mesh_name", "")
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if src_uuid and obj.get("gaussian_source_uuid") == src_uuid:
            return obj
        if src_name and obj.name == src_name:
            return obj
    return None


def _matrices_close(m1, m2, atol=1e-6):
    a = np.asarray(m1, dtype=np.float64)
    b = np.asarray(m2, dtype=np.float64)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=atol))


# ------------------------------------------------------------ gaussian IO
def gaussian_table(g_obj):
    """(N, 59) float32 KIRI gaussian_data.

    Sources, in order of preference:
      1. runtime cache (bpy.gaussian_object_cache[g_obj.name])
      2. the object's 'gaussian_data' IDProperty (with 'gaussian_count')
      3. MESH representation: the vertices are the splat centers, with
         rot_0..3 / scale_0..2 float attributes when KIRI wrote them
         (fallback: identity quaternion, unit scale)

    Raises ValueError with a user-facing message when nothing is readable.
    """
    cache = getattr(bpy, "gaussian_object_cache", None)
    if isinstance(cache, dict) and g_obj.name in cache:
        entry = cache[g_obj.name]
        c = np.asarray(entry.get("gaussian_data"), dtype=np.float32)
        cnt = int(entry.get("gaussian_count", 0))
        if c.shape == (cnt, 59):
            return c
    data = g_obj.get("gaussian_data")
    count = int(g_obj.get("gaussian_count", 0))
    if count > 0 and isinstance(data, (bytes, bytearray)) and len(data) == count * 59 * 4:
        return np.frombuffer(bytearray(data), dtype=np.float32).reshape(count, 59)
    if g_obj.type == "MESH":
        me = g_obj.data
        n = len(me.vertices)
        if n > 0:
            arr = np.zeros((n, 59), dtype=np.float32)
            co = np.empty(n * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            arr[:, :3] = co.reshape(n, 3)
            arr[:, 3] = 1.0      # identity quaternion (w, x, y, z)
            arr[:, 7:10] = 1.0   # unit linear scale
            attrs = getattr(me, "attributes", None)
            if attrs is not None:
                for i, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
                    try:
                        attr = attrs[name]
                    except Exception:
                        continue
                    if attr.data_type == "FLOAT" and len(attr.data) == n:
                        vals = np.empty(n, dtype=np.float32)
                        attr.data.foreach_get("value", vals)
                        arr[:, 3 + i] = vals
                # KIRI persists quaternions in rot_0..3; renormalize in case
                # a deformed covariance was written back with drift (norm 0
                # rows become the identity quaternion, matching KIRI).
                rn = np.linalg.norm(arr[:, 3:7], axis=1)
                zero = rn == 0.0
                rn[zero] = 1.0
                arr[:, 3:7] /= rn[:, None]
                arr[zero, 3] = 1.0
                for i, name in enumerate(("scale_0", "scale_1", "scale_2")):
                    try:
                        attr = attrs[name]
                    except Exception:
                        continue
                    if attr.data_type == "FLOAT" and len(attr.data) == n:
                        vals = np.empty(n, dtype=np.float32)
                        attr.data.foreach_get("value", vals)
                        # KIRI persists LOG scale in these attributes and
                        # applies exp() when rebuilding gaussian_data.
                        arr[:, 7 + i] = np.exp(vals)
            return arr
    raise ValueError(
        "Gaussian object '{}' has no readable gaussian_data "
        "(no runtime cache, no gaussian_data property, no mesh vertices). "
        "Run a KIRI refresh, or check that the object is a Gaussian "
        "imported by KIRI.".format(g_obj.name)
    )


def gaussian_world_state(g_obj, table=None):
    """World-space Gaussian state.

    Returns (mu_w (N, 3), R_w (N, 3, 3), scale (N, 3), A_inv (3, 3)).
    R_w = A @ R_local and scale are a valid factorization of the world
    covariance Sigma_w = R_w diag(scale^2) R_w^T, where A is the object's
    matrix_world 3x3 (rotation, optionally including scale). A_inv is
    used to factor propagated results back into local space.
    """
    if table is None:
        table = gaussian_table(g_obj)
    arr = table
    local_mu = arr[:, :3].astype(np.float64)
    quats = arr[:, 3:7].astype(np.float64)
    scale = arr[:, 7:10].astype(np.float64)

    mat = g_obj.matrix_world
    A = np.array(mat.to_3x3(), dtype=np.float64)
    trans = np.array(mat.to_translation(), dtype=np.float64)
    A_inv = np.linalg.inv(A)

    mu_w = local_mu @ A.T + trans
    R_local = quat_to_matrix(quats)
    R_w = np.einsum("ij,njk->nik", A, R_local)
    return mu_w, R_w, scale, A_inv


def _world_to_local(world, matrix):
    inv = matrix.inverted()
    rot = np.array(inv.to_3x3(), dtype=np.float64)
    trans = np.array(inv.to_translation(), dtype=np.float64)
    return world @ rot.T + trans


def write_back(g_obj, mu_local, quats, scales, persist=True, arr=None):
    """Write local-space (mu, quat wxyz, linear scale) into a KIRI object.

    Mirrors binding.deform.write_back_positions: cache always updated,
    the large IDProperty only rewritten when persist=True (delete-then-set
    for the Blender 5.2 overwrite quirk). MESH-representation Gaussians
    get their vertex positions plus rot_*/scale_* attributes when present.

    arr is the base (N, 59) table to patch. Pass it in when the binding
    source differs from the write target (e.g. a source MESH deforming a
    GPU proxy EMPTY whose own gaussian_data is corrupt/missing); when
    omitted it is read from g_obj itself.
    """
    count = int(g_obj.get("gaussian_count", 0))
    if arr is None:
        try:
            arr = gaussian_table(g_obj)
        except ValueError:
            n = len(np.asarray(mu_local))
            arr = np.zeros((n, 59), dtype=np.float32)
            arr[:, 3] = 1.0
            arr[:, 7:10] = 1.0
    if not arr.flags.writeable:
        arr = arr.copy()
    arr[:, 0:3] = np.asarray(mu_local, dtype=np.float32)
    arr[:, 3:7] = np.asarray(quats, dtype=np.float32)
    arr[:, 7:10] = np.asarray(scales, dtype=np.float32)
    count = count or len(arr)

    if g_obj.type == "MESH":
        g_obj.data.vertices.foreach_set("co", np.asarray(mu_local, dtype=np.float32).ravel())
        _write_mesh_attributes(g_obj, quats, scales, count)
        g_obj.data.update()
        # KIRI refreshes BlenderObj proxies from the EVALUATED source mesh;
        # without a re-evaluation it still sees the pre-deformation copy.
        try:
            bpy.context.evaluated_depsgraph_get().update()
        except Exception:
            pass
        # KIRI renders MESH-representation Gaussians from the runtime
        # cache (built by refresh_data_from_evaluated_sources), so the
        # rebuilt 59-column table must land there too, or the viewport
        # keeps showing the pre-deformation splats.
        if persist:
            if "gaussian_data" in g_obj:
                del g_obj["gaussian_data"]
            g_obj["gaussian_data"] = arr.tobytes()
            g_obj["gaussian_count"] = count
        cache = getattr(bpy, "gaussian_object_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            bpy.gaussian_object_cache = cache
        if g_obj.name in cache:
            cache[g_obj.name]["gaussian_data"] = arr
            cache[g_obj.name]["gaussian_count"] = count
        else:
            cache[g_obj.name] = {
                "gaussian_data": arr,
                "gaussian_count": count,
                "sh_degree": int(g_obj.get("sh_degree", 48)),
                "object": g_obj,
                "ply_filepath": g_obj.get("ply_filepath", ""),
                "source_info": "",
            }
        bpy.gaussian_global_needs_update = True
        bpy.gaussian_needs_depth_sort = True
        return

    if g_obj.type != "MESH" and g_obj.get("source_mesh_uuid"):
        # Derived proxy: its gaussian_data IDProperty is redundant (KIRI
        # rebuilds it from the evaluated source mesh) and a stale copy may
        # be corrupted by the Blender 5.2 large-property overwrite bug,
        # which breaks KIRI's cache auto-reconstruct. Drop it so KIRI uses
        # the source-mesh fallback path.
        if "gaussian_data" in g_obj:
            del g_obj["gaussian_data"]
            print("[GME-debug] dropped stale gaussian_data from proxy '{}'".format(g_obj.name))

    if persist and not g_obj.get("source_mesh_uuid"):
        # Source-backed proxies are DERIVED data: KIRI rebuilds them from the
        # evaluated source mesh on refresh, so their 160MB+ gaussian_data
        # IDProperty is redundant, and rewriting it on every apply corrupts
        # the property in Blender 5.2 ('buffer size must be a multiple of
        # element size'). Only PLY-source proxies own their data here.
        if "gaussian_data" in g_obj:
            del g_obj["gaussian_data"]
        g_obj["gaussian_data"] = arr.tobytes()
        g_obj["gaussian_count"] = count
    elif persist:
        print("[GME-debug] skip proxy '{}' gaussian_data IDProperty write "
              "(derived from source mesh)".format(g_obj.name))

    cache = getattr(bpy, "gaussian_object_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        bpy.gaussian_object_cache = cache
    if g_obj.name in cache:
        cache[g_obj.name]["gaussian_data"] = arr
        cache[g_obj.name]["gaussian_count"] = count
    else:
        entry = {
            "gaussian_data": arr,
            "gaussian_count": count,
            "sh_degree": int(g_obj.get("sh_degree", 48)),
            "object": g_obj,
            "ply_filepath": g_obj.get("ply_filepath", ""),
            "source_info": "",
        }
        if g_obj.get("source_mesh_uuid"):
            entry["source_mesh_uuid"] = g_obj.get("source_mesh_uuid")
            entry["source_mesh_name"] = g_obj.get("source_mesh_name", "")
            entry["source_info"] = "Mesh:{}".format(
                g_obj.get("source_mesh_name", "Unknown")
            )
        cache[g_obj.name] = entry
    bpy.gaussian_global_needs_update = True
    bpy.gaussian_needs_depth_sort = True


def _write_mesh_attributes(g_obj, quats, scales, count):
    me = g_obj.data
    attrs = getattr(me, "attributes", None)
    if attrs is None:
        return
    # KIRI persists LOG scale in scale_0..2 and applies exp() when rebuilding
    # gaussian_data; writing linear scale here explodes the splat size and
    # crashes the KIRI renderer (exp(linear) can overflow to inf).
    log_scales = np.log(np.maximum(np.asarray(scales, dtype=np.float64), 1e-12))
    pairs = [
        ("rot_0", quats[:, 0]), ("rot_1", quats[:, 1]),
        ("rot_2", quats[:, 2]), ("rot_3", quats[:, 3]),
        ("scale_0", log_scales[:, 0]), ("scale_1", log_scales[:, 1]),
        ("scale_2", log_scales[:, 2]),
    ]
    for name, vals in pairs:
        try:
            attr = attrs[name]
        except Exception:
            continue
        if attr.data_type != "FLOAT" or len(attr.data) != count:
            continue
        attr.data.foreach_set("value", np.asarray(vals, dtype=np.float32))


# -------------------------------------------------------------- mesh IO
def mesh_world_geometry(obj):
    """(verts (V, 3), tris (M, 3) int32, tri_verts (M, 3, 3)) world space.

    Every polygon is fan-triangulated; triangle index order matches the
    face order used by bind_corners. Edit Mode reads the live bmesh.
    """
    mat = obj.matrix_world
    if obj.mode == "EDIT":
        import bmesh

        bm = bmesh.from_edit_mesh(obj.data)
        verts = [v.co for v in bm.verts]
        polys = [[v.index for v in f.verts] for f in bm.faces]
    else:
        verts = [v.co for v in obj.data.vertices]
        polys = [list(p.vertices) for p in obj.data.polygons]
    local = np.array([tuple(v) for v in verts], dtype=np.float64)
    rot = np.array(mat.to_3x3(), dtype=np.float64)
    trans = np.array(mat.to_translation(), dtype=np.float64)
    world = local @ rot.T + trans

    tris = []
    for pv in polys:
        if len(pv) < 3:
            continue
        for i in range(1, len(pv) - 1):
            tris.append([pv[0], pv[i], pv[i + 1]])
    tris = np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)
    return world, tris, world[tris]


# ------------------------------------------------------------ camera
class BlenderCameraSource(CameraSource):
    """CameraSource backed by the scene's camera objects (world origins)."""

    name = "Blender Scene Cameras"

    def origins(self, scene):
        cams = [
            obj.matrix_world.translation
            for obj in scene.objects
            if obj.type == "CAMERA"
        ]
        if not cams:
            raise ValueError(
                "No cameras in the scene. Add a camera for UniMGS ray binding."
            )
        return np.array([(c.x, c.y, c.z) for c in cams], dtype=np.float64)


class SurroundCameraSource(CameraSource):
    """Scene cameras plus virtual origins on a bounding sphere of the proxy.

    A single viewport camera leaves the proxy's occluded hemisphere with
    no corner bindings (rays always hit front faces first), so dragging
    back-facing vertices propagates nothing. Real UniMGS relies on dense
    training camera rigs; the virtual origins restore that coverage
    without forcing the user to place dozens of cameras.
    """

    name = "Scene Cameras + Surround"

    def __init__(self, count=6, radius_factor=1.5, base=None):
        self.count = int(count)
        self.radius_factor = radius_factor
        self.base = base if base is not None else BlenderCameraSource()

    def origins(self, scene):
        parts = []
        try:
            base = np.asarray(self.base.origins(scene), dtype=np.float64)
            if len(base):
                parts.append(base)
        except ValueError:
            pass
        if self.count > 0:
            m_obj = scene.gme_props.proxy_obj
            verts, _tris, _tt = mesh_world_geometry(m_obj)
            center = verts.mean(axis=0)
            radius = float(np.linalg.norm(verts - center, axis=1).max())
            radius = max(radius, 1e-3) * self.radius_factor
            i = np.arange(self.count, dtype=np.float64)
            phi = np.pi * (3.0 - np.sqrt(5.0))  # golden-angle (fibonacci) sphere
            y = 1.0 - 2.0 * (i + 0.5) / self.count
            r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
            th = phi * i
            extra = np.stack([r * np.cos(th), y, r * np.sin(th)], axis=1)
            parts.append(extra * radius + center)
        if not parts:
            raise ValueError(
                "No cameras in the scene. Add a camera for UniMGS ray binding."
            )
        return np.vstack(parts)


# ------------------------------------------------------------ pipeline
def build_binding(scene, k=1.0, camera_source=None):
    """Full UniMGS binding: BBX -> camera rays -> best face per corner.

    Returns the UniMGSBindingData (also stored for the scene). Raises
    ValueError with a user-facing message on failure.
    """
    g_obj = _resolve_gaussian(scene)
    if g_obj is None:
        raise ValueError("No Gaussian object found in the scene.")
    m_obj = scene.gme_props.proxy_obj
    if m_obj is None or m_obj.type != "MESH":
        raise ValueError("No proxy mesh set (use 'Set Active as Proxy Mesh').")
    if m_obj.mode == "EDIT":
        raise ValueError(
            "Proxy mesh is in Edit Mode. Switch to Object Mode before "
            "building the UniMGS binding - Edit Mode vertex indices can "
            "differ from the mesh data, and the binding would break as "
            "soon as you leave Edit Mode."
        )
    # A rebuild invalidates the old binding outright: never let a stale
    # binding from the previous topology answer Apply/auto-follow calls.
    clear_binding(scene)

    mu_w, R_w, scale, _ = gaussian_world_state(g_obj)
    if len(mu_w) == 0:
        raise ValueError("The Gaussian object holds no splats.")
    verts, tris, tri_verts = mesh_world_geometry(m_obj)
    if len(tris) == 0:
        raise ValueError("The proxy mesh has no faces (binding needs a mesh, "
                         "not a bare point cloud).")
    print("[GME-debug] build: proxy={} mode={} verts={} tris={} gaussians={}".format(
        m_obj.name, m_obj.mode, len(verts), len(tris), len(mu_w)))

    corners = get_gaussian_bbx(mu_w, R_w, scale, k=k)
    src = camera_source if camera_source is not None else SurroundCameraSource(
        count=scene.gme_props.unimags_surround)
    origins = src.origins(scene)
    print("[GME-debug] cameras: {}".format(len(origins)))
    wm = getattr(bpy.context, "window_manager", None)
    ws = getattr(bpy.context, "workspace", None)

    def _progress(done, total):
        if total:
            pct = 5.0 + 93.0 * done / total
            if wm is not None:
                wm.progress_update(pct)
            if ws is not None:
                ws.status_text_set("GME: binding {:.0f}%".format(pct))

    try:
        if wm is not None:
            wm.progress_begin(0, 100)
            wm.progress_update(3.0)
        res = bind_corners(corners, origins, tri_verts,
                           caster=BVHCaster(tri_verts), centers=mu_w,
                           progress=_progress)
        data = UniMGSBindingData(
            mu_w, res["face_ids"], res["barycentric"],
            res["distances"], res["hit_points"],
            rest_verts=verts, rest_tris=tris,
        )
        set_binding(scene, data)
        if wm is not None:
            wm.progress_update(100.0)
    finally:
        if wm is not None:
            wm.progress_end()
        if ws is not None:
            ws.status_text_set(None)
    print("[GME-debug] binding REBUILT: rest_verts={} rest_tris={} "
          "gaussians={}".format(len(verts), len(tris), len(mu_w)))
    return data


def _vertex_deformations(scene, rest_v, cur_v, rest_t):
    """Eq.12 vertex stage; GPU compute when scene.gme_props.gpu_accel is set.

    The GPU kernel returns the same (delta, rot, shear) contract as the
    CPU vertex_deformations. Any GPU failure (no GL context, shader
    compile error, texture limit, mesh too dense) logs once and falls
    back to the CPU loop.
    """
    if scene.gme_props.gpu_accel:
        try:
            from . import gpu_pipeline

            t0 = time.perf_counter()
            delta, rot, shear = gpu_pipeline.vertex_deformations_gpu(
                rest_v, cur_v, rest_t
            )
            print("[GME-debug] vertex deformations: GPU ({:.1f} ms)".format(
                (time.perf_counter() - t0) * 1e3))
            return delta, rot, shear
        except Exception as exc:
            print("[GME-debug] GPU vertex deformations failed ({}); "
                  "falling back to CPU".format(exc))
    t0 = time.perf_counter()
    delta, rot, shear = vertex_deformations(rest_v, cur_v, rest_t)
    print("[GME-debug] vertex deformations: CPU ({:.1f} ms)".format(
        (time.perf_counter() - t0) * 1e3))
    return delta, rot, shear


def apply_deformation(scene, persist=True):
    """Run Eq.12/13 on the current mesh and write the result to KIRI.

    Returns (data, moved, max_displacement). Raises ValueError with a
    user-facing message on failure. Unbound Gaussians and Gaussians whose
    corners see no deformation keep their exact rest parameters.
    """
    data = get_binding(scene)
    if data is None:
        raise ValueError("No UniMGS binding. Run 'Build UniMGS Binding' first.")
    m_obj = scene.gme_props.proxy_obj
    if m_obj is None or m_obj.type != "MESH":
        raise ValueError("No proxy mesh set (use 'Set Active as Proxy Mesh').")
    g_obj = _resolve_gaussian(scene)
    if g_obj is None:
        raise ValueError("No Gaussian object found in the scene.")
    render_obj = _resolve_render_target(g_obj)
    source_mesh = _resolve_source_mesh(g_obj)
    # The proxy EMPTY renders in the source mesh's frame: keep its
    # transform in lockstep so whole-object moves of the source carry the
    # splats immediately (KIRI also syncs this on refresh).
    if (source_mesh is not None and render_obj is not source_mesh and
            not _matrices_close(render_obj.matrix_world, source_mesh.matrix_world)):
        render_obj.matrix_world = source_mesh.matrix_world.copy()
        print("[GME-debug] proxy '{}' transform synced to source '{}'".format(
            render_obj.name, source_mesh.name))
    print("[GME-debug] g_obj={} render={} source={}".format(
        g_obj.name, render_obj.name,
        source_mesh.name if source_mesh is not None else None))
    table = gaussian_table(g_obj)

    rest_v = data.rest_verts
    rest_t = data.rest_tris
    if len(rest_t) == 0:
        raise ValueError("Binding holds no rest mesh - rebuild the binding.")
    cur_v, cur_t, _ = mesh_world_geometry(m_obj)
    if len(cur_v) != len(rest_v) or len(cur_t) != len(rest_t) or \
            not np.array_equal(cur_t, rest_t):
        raise ValueError(
            "Proxy mesh topology changed ({} vertices / {} triangles now, "
            "binding expects {} / {}). Switch to Object Mode and click "
            "'Build UniMGS Binding' to rebind the new mesh.".format(
                len(cur_v), len(cur_t), len(rest_v), len(rest_t)
            )
        )
    print("[GME-debug] mode={} cur min/max {:.5f}/{:.5f} rest min/max {:.5f}/{:.5f} max|cur-rest| {:.6f}".format(
        m_obj.mode, cur_v.min(), cur_v.max(), rest_v.min(), rest_v.max(),
        float(np.abs(cur_v - rest_v).max())))

    delta, rot, shear = _vertex_deformations(scene, rest_v, cur_v, rest_t)
    print("[GME-debug] delta max {:.6f}".format(float(np.abs(delta).max())))
    nz = np.flatnonzero(np.abs(delta).max(axis=1) > 1e-9)
    ref = np.unique(rest_t[data.triangle_ids[data.valid_mask]]) \
        if data.valid_mask.any() else np.array([], dtype=np.int32)
    bv = data.barycentric[data.valid_mask]
    print("[GME-debug] moved_verts={}/{} referenced_by_bound_tris={}".format(
        len(nz), len(delta), int(np.isin(nz, ref).sum())))
    print("[GME-debug] tri_ids range {}..{} n_tris={} n_valid={} | bary zero_rows={}/{} mean={:.4f} min={:.6f} max={:.6f}".format(
        int(data.triangle_ids.min()), int(data.triangle_ids.max()), len(rest_t),
        int(data.valid_mask.sum()),
        int((np.abs(bv).sum(axis=1) == 0).sum()), len(bv),
        float(bv.mean()), float(bv.min()), float(bv.max())))
    c_d, c_lr, c_s = corner_deformations(
        data.triangle_ids, data.barycentric, rest_t, delta, rot, shear
    )
    print("[GME-debug] corner_delta max {:.6f}".format(float(np.abs(c_d).max())))
    if float(np.abs(c_d).max()) < 1e-9 and float(np.abs(delta).max()) > 1e-9:
        print("[GME-debug] WARNING: dragged vertices have no bound faces "
              "(referenced_by_bound_tris={}) - raise Surround Cameras and "
              "rebuild the binding, or drag camera-visible faces.".format(
                  int(np.isin(nz, ref).sum())))

    mu_w, R_w, scale, A_inv = gaussian_world_state(g_obj, table=table)
    if len(mu_w) != len(data.gaussian_centers):
        raise ValueError(
            "Gaussian count changed since binding ({} now, {} at bind time). "
            "Rebuild the binding.".format(len(mu_w), len(data.gaussian_centers))
        )

    mu_new_w, R_inc, S_inc = gaussian_updates(
        mu_w, R_w, scale, c_d, c_lr, c_s, data.valid_mask
    )
    print("[GME-debug] mu shift max {:.6f}".format(
        float(np.abs(mu_new_w - mu_w).max())))
    R_out_w, s_out = propagate_covariance(R_w, scale, R_inc, S_inc)

    # Factor the propagated world covariance back into local space so the
    # object transform is not baked twice. Untouched Gaussians keep their
    # exact local quaternion / scale (no eigendecomposition noise).
    quats_out = np.array(table[:, 3:7], dtype=np.float64).copy()
    scales_out = np.array(scale).copy()
    dR = np.max(np.abs(R_inc - np.eye(3)), axis=(-2, -1))
    dS = np.max(np.abs(S_inc - np.eye(3)), axis=(-2, -1))
    changed = (dR >= IDENT_TOL) | (dS >= IDENT_TOL)
    if changed.any():
        I3 = np.eye(3)
        idx = np.flatnonzero(changed)
        C = np.einsum("ij,njk->nik", A_inv, R_out_w[idx])
        R_l, s_l = propagate_covariance(
            np.tile(I3, (len(idx), 1, 1)), s_out[idx], C, np.tile(I3, (len(idx), 1, 1))
        )
        quats_out[idx] = matrix_to_quat(R_l)
        scales_out[idx] = s_l

    # KIRI renders the GPU proxy (if any), not the source mesh: write the
    # full 59-column table there so the viewport actually moves, and keep
    # the source mesh in sync so a later KIRI refresh does not overwrite
    # the proxy back to the stale rest pose. Each target gets local
    # coordinates under its own matrix_world.
    mu_local_target = _world_to_local(mu_new_w, render_obj.matrix_world)
    write_back(render_obj, mu_local_target, quats_out, scales_out,
               persist=persist, arr=table)
    if g_obj is not render_obj:
        mu_local = _world_to_local(mu_new_w, g_obj.matrix_world)
        # Copy so the proxy's cache entry (same arr object, set above) is not
        # silently re-patched to a different local space.
        write_back(g_obj, mu_local, quats_out, scales_out,
                   persist=False, arr=table.copy())
    if (source_mesh is not None and source_mesh is not g_obj and
            source_mesh is not render_obj):
        mu_local_src = _world_to_local(mu_new_w, source_mesh.matrix_world)
        write_back(source_mesh, mu_local_src, quats_out, scales_out,
                   persist=False, arr=table.copy())

    bound = data.valid_mask.any(axis=1)
    disp = np.linalg.norm(mu_new_w[bound] - mu_w[bound], axis=1)
    moved = int(bound.sum())
    max_disp = float(disp.max()) if moved else 0.0
    if moved:
        p50, p90, p99 = np.percentile(disp, [50, 90, 99])
        print("[GME-debug] disp n={} mean={:.4f} p50={:.4f} p90={:.4f} "
              "p99={:.4f} max={:.4f} | >0.5:{} >0.1:{} >0.01:{}".format(
                  moved, float(disp.mean()), float(p50), float(p90),
                  float(p99), float(disp.max()),
                  int((disp > 0.5).sum()), int((disp > 0.1).sum()),
                  int((disp > 0.01).sum())))

    # Keep the snapshot consistent with the deformed state so a second
    # apply (or a later save) reflects the current mesh.
    data.gaussian_centers = np.asarray(mu_new_w, dtype=np.float64)
    data.rest_verts = np.asarray(cur_v, dtype=np.float64)
    return data, moved, max_disp
