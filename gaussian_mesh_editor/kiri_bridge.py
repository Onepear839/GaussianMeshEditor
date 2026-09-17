"""Interoperability with the KIRI 3DGS Render add-on.

Detection is data-level (custom properties / mesh attributes) so this
add-on works whether or not KIRI is installed. KIRI helper modules are
only returned when already loaded; they are never force-imported here,
because importing the KIRI package would execute its register().

KIRI may be loaded either as a legacy add-on (module
'dgs_render_by_kiri_engine') or as a Blender extension (module
'bl_ext.user_default.dgs_render_by_kiri_engine'); both are handled.
"""

import sys

import bpy

KIRI_ADDON_ID = "dgs_render_by_kiri_engine"


def _find_kiri_prefix():
    """Return the module prefix KIRI is loaded under, or None.

    '' for a legacy add-on, 'bl_ext.user_default.' for an extension.
    """
    for name in tuple(sys.modules):
        if name == KIRI_ADDON_ID:
            return ""
        if name.startswith("bl_ext.") and name.endswith(f".{KIRI_ADDON_ID}"):
            return name[: -len(KIRI_ADDON_ID)]
    try:
        for key in bpy.context.preferences.addons.keys():
            if key == KIRI_ADDON_ID:
                return ""
            if key.endswith(f".{KIRI_ADDON_ID}"):
                return key[: -len(KIRI_ADDON_ID)]
    except Exception:
        pass
    return None


def kiri_enabled():
    """Return True when the KIRI 3DGS Render add-on is present and enabled."""
    return _find_kiri_prefix() is not None


def kiri_important():
    """Return the already-loaded KIRI src.important module, or None.

    Useful for later phases that want KIRI helpers (e.g.
    kiri_sync_gaussian_object_cache) without re-importing the package.
    """
    prefix = _find_kiri_prefix()
    if prefix is None:
        return None
    return sys.modules.get(f"{prefix}{KIRI_ADDON_ID}.src.important")


def is_gaussian_object(obj):
    """True for KIRI GPU proxies (EMPTY) and PLY-imported 3DGS meshes."""
    if obj is None or obj.type not in {"MESH", "EMPTY"}:
        return False
    if obj.get("is_gaussian_splat", False):
        return True
    if obj.type == "MESH":
        attrs = getattr(obj.data, "attributes", None)
        if attrs is not None:
            try:
                return "f_dc_0" in attrs
            except TypeError:
                return any(a.name == "f_dc_0" for a in attrs)
    return False


def find_gaussian_objects(scene=None):
    """Return all Gaussian objects in the scene, in scene order."""
    scene = scene if scene is not None else bpy.context.scene
    return [obj for obj in scene.objects if is_gaussian_object(obj)]
