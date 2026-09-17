"""Auto-follow: drag Gaussians live while the proxy mesh is edited.

Two triggers share one vertex snapshot (so no double work):

  1. depsgraph_update_post  - fires on object-level G/R/S moves and any
     depsgraph-triggering data change (latency ~0).
  2. bpy.app.timers polling - fires ~20x/s. In Edit Mode the mesh
     vertices live in the bmesh (not obj.data.vertices) until the mode
     is exited, so only a timer can see them mid-drag. This is what
     makes the Gaussians follow while the user is still dragging.

Both channels run the same _run_if_changed(): compare the current
world-space vertices against the last snapshot, and only then run the
UniMGS deformation (Eq.12/13). The snapshot comparison also breaks the
update -> write-back -> update loop, since writing the Gaussian object
never touches the mesh.

Opt-in only: the handlers are always registered but return immediately
unless scene.gme_props.auto_follow_enabled is set.
"""

import time
import traceback

import bmesh
import numpy as np

import bpy

from . import blender as unimags_blender

POLL_INTERVAL = 0.05  # seconds between Edit-Mode polls
_PRINT_GAP = 0.3  # seconds between console progress prints
_DIAG_GAP = 2.0  # seconds between diagnostic prints

_snapshot = {}
_last_print = {"t": 0.0}
_last_diag = {"t": 0.0}
_timer_alive = {"done": False}
_last_beat = {"t": 0.0}


def _diag(msg):
    """Throttled diagnostic line so silent early-returns are visible."""
    now = time.monotonic()
    if now - _last_diag["t"] >= _DIAG_GAP:
        _last_diag["t"] = now
        print("[GME Auto-Follow] diag: {}".format(msg))


def _read_vertices(mesh_obj):
    """(V, 3) float64 copy of world-space vertex coordinates.

    In Edit Mode the live positions live in the bmesh, not in
    obj.data.vertices; read them from there so a mid-drag state is seen.
    """
    if mesh_obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(mesh_obj.data)
        local = np.array([v.co for v in bm.verts], dtype=np.float64)
    else:
        v = mesh_obj.data.vertices
        local = np.empty((len(v), 3), dtype=np.float64)
        v.foreach_get("co", local.ravel())
    mat = mesh_obj.matrix_world
    rot = np.array(mat.to_3x3(), dtype=np.float64)
    trans = np.array(mat.to_translation(), dtype=np.float64)
    return local @ rot.T + trans


def _tag_redraw():
    screen = getattr(bpy.context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


def _run_if_changed(scene):
    """Deform the Gaussians if the proxy's world vertices moved."""
    props = scene.gme_props
    if not props.auto_follow_enabled:
        _diag("'Auto Follow (live)' is OFF - tick the checkbox in the panel")
        return
    m_obj = props.proxy_obj
    if m_obj is None or m_obj.type != "MESH":
        _diag("no proxy mesh set (use 'Set Active as Proxy Mesh')")
        return
    if unimags_blender.get_binding(scene) is None:
        _diag("no UniMGS binding found - run 'Build UniMGS Binding' first "
              "(bindings are now saved to disk and auto-reload after restart)")
        return

    verts = _read_vertices(m_obj)
    key = (scene.name, m_obj.name)
    last = _snapshot.get(key)
    if last is not None and last.shape == verts.shape and np.array_equal(last, verts):
        return
    mesh_disp = 0.0
    if last is not None and last.shape == verts.shape:
        mesh_disp = float(np.linalg.norm(verts - last, axis=1).max())
    _snapshot[key] = verts

    try:
        data, moved, max_disp = unimags_blender.apply_deformation(
            scene, persist=False
        )
    except (ValueError, BufferError) as exc:
        msg = str(exc)
        if "topology changed" in msg:
            # Topology edits invalidate the binding; hint once per topology
            # instead of spamming every poll tick.
            print("[GME Auto-Follow] proxy topology changed - switch to "
                  "Object Mode and click 'Build UniMGS Binding' to rebind, "
                  "then Auto-Follow resumes.")
        else:
            now = time.monotonic()
            if now - _last_print["t"] >= _PRINT_GAP:
                _last_print["t"] = now
                print("[GME Auto-Follow] skipped: {}".format(msg))
        return
    except Exception:
        print("[GME Auto-Follow] unexpected error in apply_deformation "
              "(Auto-Follow paused, see traceback):")
        traceback.print_exc()
        return
    now = time.monotonic()
    if now - _last_print["t"] >= _PRINT_GAP:
        _last_print["t"] = now
        print(
            "[GME Auto-Follow] scene='{}' mesh={} moved {:,} Gaussians "
            "(mesh max {:.4f}, gaussian max {:.4f})".format(
                scene.name, m_obj.name, moved, mesh_disp, max_disp
            )
        )
    _tag_redraw()


def _ensure_timer():
    """Re-register the poll timer if it was lost (file load, long ops).

    depsgraph handlers fire on every scene change, so checking here heals
    the loop without waiting for a manual restart.
    """
    if not bpy.app.timers.is_registered(_poll_timer):
        bpy.app.timers.register(_poll_timer)
        print("[GME Auto-Follow] poll timer re-registered")


def auto_follow_handler(scene, depsgraph):
    _ensure_timer()
    try:
        _run_if_changed(scene)
    except Exception:
        print("[GME Auto-Follow] handler error (traceback below):")
        traceback.print_exc()


def _poll_timer():
    # Any exception here would silently kill the timer (Blender removes
    # timers that raise), so swallow and log everything to keep the
    # Auto-Follow loop alive and diagnosable.
    try:
        scene = getattr(bpy.context, "scene", None)
        if scene is not None:
            _run_if_changed(scene)
            if not _timer_alive["done"]:
                _timer_alive["done"] = True
                print("[GME Auto-Follow] poll timer active ({} s)".format(POLL_INTERVAL))
            now = time.monotonic()
            if now - _last_beat["t"] >= 10.0:
                _last_beat["t"] = now
                props = scene.gme_props
                m_obj = props.proxy_obj
                uni = unimags_blender.get_binding(scene) is not None
                print("[GME Auto-Follow] heartbeat: enabled={} proxy={} "
                      "binding={}".format(
                          props.auto_follow_enabled,
                          m_obj.name if m_obj is not None else None,
                          "uni" if uni else "none"))
    except Exception:
        print("[GME Auto-Follow] timer error (timer kept alive):")
        traceback.print_exc()
    return POLL_INTERVAL


def register():
    handlers = bpy.app.handlers.depsgraph_update_post
    if auto_follow_handler not in handlers:
        handlers.append(auto_follow_handler)
    if not bpy.app.timers.is_registered(_poll_timer):
        bpy.app.timers.register(_poll_timer)


def unregister():
    handlers = bpy.app.handlers.depsgraph_update_post
    if auto_follow_handler in handlers:
        handlers.remove(auto_follow_handler)
    if bpy.app.timers.is_registered(_poll_timer):
        bpy.app.timers.unregister(_poll_timer)
