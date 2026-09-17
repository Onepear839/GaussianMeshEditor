# Gaussian Mesh Editor — independent Proxy Mesh alignment module for
# KIRI 3DGS Render. The KIRI add-on itself is never modified.
#
# Phase 2: designate an existing scene mesh as proxy + green wireframe display.
#
# Flat layout (no subpackage) so the add-on installs cleanly as both an
# extension and a legacy add-on.

import bpy

bl_info = {
    "name": "Gaussian Mesh Editor",
    "author": "GME",
    "description": "Proxy Mesh alignment module for KIRI 3DGS Render",
    "blender": (5, 1, 0),
    "version": (0, 19, 15),
    "category": "3D View",
}

from .props import GME_Props
from .operators import (
    GME_OT_set_proxy,
    GME_OT_print_bounds,
    GME_OT_rough_align,
    GME_OT_reset_proxy,
    GME_OT_export_mesh,
    GME_OT_subdivide,
    GME_OT_fit_gaussian,
    GME_OT_refine,
    GME_OT_unimags_build_binding,
    GME_OT_unimags_apply_deformation,
    GME_OT_unimags_clear_binding,
)
from .ui import GME_PT_alignment, GME_PT_unimags
from .unimags import auto_follow


def _safe_register(cls):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)


def _safe_unregister(cls):
    try:
        bpy.utils.unregister_class(cls)
    except Exception:
        pass


def register():
    _safe_register(GME_Props)
    bpy.types.Scene.gme_props = bpy.props.PointerProperty(type=GME_Props)
    _safe_register(GME_OT_set_proxy)
    _safe_register(GME_OT_print_bounds)
    _safe_register(GME_OT_rough_align)
    _safe_register(GME_OT_reset_proxy)
    _safe_register(GME_OT_export_mesh)
    _safe_register(GME_OT_subdivide)
    _safe_register(GME_OT_fit_gaussian)
    _safe_register(GME_OT_refine)
    _safe_register(GME_OT_unimags_build_binding)
    _safe_register(GME_OT_unimags_apply_deformation)
    _safe_register(GME_OT_unimags_clear_binding)
    _safe_register(GME_PT_alignment)
    _safe_register(GME_PT_unimags)
    auto_follow.register()


def unregister():
    auto_follow.unregister()
    _safe_unregister(GME_PT_unimags)
    _safe_unregister(GME_PT_alignment)
    _safe_unregister(GME_OT_unimags_clear_binding)
    _safe_unregister(GME_OT_unimags_apply_deformation)
    _safe_unregister(GME_OT_unimags_build_binding)
    _safe_unregister(GME_OT_refine)
    _safe_unregister(GME_OT_fit_gaussian)
    _safe_unregister(GME_OT_subdivide)
    _safe_unregister(GME_OT_export_mesh)
    _safe_unregister(GME_OT_reset_proxy)
    _safe_unregister(GME_OT_rough_align)
    _safe_unregister(GME_OT_print_bounds)
    _safe_unregister(GME_OT_set_proxy)
    try:
        del bpy.types.Scene.gme_props
    except AttributeError:
        pass
    _safe_unregister(GME_Props)
