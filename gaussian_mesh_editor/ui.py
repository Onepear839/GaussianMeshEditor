"""Phase 2/3: "Gaussian Mesh Editor" side panel."""

import bpy

from . import kiri_bridge
from .unimags import blender as unimags_blender


def _fmt(v):
    return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"


class GME_PT_alignment(bpy.types.Panel):
    bl_label = "Gaussian Mesh Editor"
    bl_idname = "GME_PT_alignment"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Gaussian Mesh Editor"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        props = context.scene.gme_props
        try:
            self._draw_panel(layout, context, props)
        except Exception:
            import traceback
            traceback.print_exc()
            layout.label(text="Panel error - see system console", icon="ERROR")

    def _draw_panel(self, layout, context, props):
        gaussians = kiri_bridge.find_gaussian_objects(context.scene)
        # UI drawing must not write to ID properties (Blender forbids it);
        # the Gaussian target pointer is only assigned by operators.
        target = props.gaussian_obj
        if target is None and gaussians:
            target = gaussians[0]

        box = layout.box()
        box.label(text="Proxy Mesh", icon="MESH_DATA")
        box.label(text="Select a mesh object in the scene, then:")
        box.operator("gme.set_proxy_mesh", text="Set Active as Proxy Mesh", icon="CHECKMARK")
        row = box.row()
        row.label(text="Proxy:")
        if props.proxy_obj is not None:
            row.label(text=props.proxy_obj.name, icon="OBJECT_DATA")
        else:
            row.label(text="None")
        box.operator("gme.reset_proxy", text="Reset Transform", icon="LOOP_BACK")

        box = layout.box()
        row = box.row()
        row.prop(props, "enable_target", text="Alignment Target", icon="OBJECT_DATA")
        row = box.row()
        row.label(text="Gaussian:")
        row.prop(props, "gaussian_obj", text="")
        if target is None:
            box.label(text="(no Gaussian object found)", icon="INFO")
        elif props.gaussian_obj is None:
            box.label(text=f"(auto: {target.name})", icon="INFO")

        box = layout.box()
        box.label(text="Rough Alignment", icon="CHECKMARK")
        box.label(text="Center + uniform scale on the Gaussian bounds.")
        box.label(text="Then fine-tune with G / R / S.")
        box.operator("gme.rough_align", text="Rough Align", icon="CHECKMARK")
        if props.align_scale > 0.0:
            row = box.row()
            row.label(text="Last align:")
            row.label(text=f"scale {props.align_scale:.3f}x")

        box = layout.box()
        box.label(text="Bounds (World Space)", icon="EMPTY_AXIS")
        if props.has_bounds:
            row = box.row()
            row.label(text="Gaussian Center:")
            row.label(text=_fmt(props.gaussian_center))
            row = box.row()
            row.label(text="Gaussian Size:")
            row.label(text=_fmt(props.gaussian_size))
            row = box.row()
            row.label(text="Mesh Center:")
            row.label(text=_fmt(props.mesh_center))
            row = box.row()
            row.label(text="Mesh Size:")
            row.label(text=_fmt(props.mesh_size))
        else:
            box.label(text="Not computed yet")
        box.operator("gme.print_bounds", text="Print Bounds", icon="CONSOLE")

        box = layout.box()
        box.label(text="Subdivide & Fit", icon="MOD_SUBSURF")
        box.label(text="Linear subdivision keeps the shape; fitting pulls it")
        box.label(text="onto the Gaussian cloud. Run them repeatedly.")
        row = box.row()
        row.prop(props, "subdiv_levels", text="Levels")
        row = box.row()
        row.label(text="Faces:")
        if props.proxy_obj is not None and props.proxy_obj.type == "MESH":
            row.label(text="{:,}".format(len(props.proxy_obj.data.polygons)))
        else:
            row.label(text="-")
        box.operator("gme.subdivide", text="Subdivide", icon="MOD_SUBSURF")
        box.separator()
        row = box.row()
        row.prop(props, "fit_mode", text="Mode")
        row.prop(props, "fit_k", text="k")
        row = box.row()
        row.prop(props, "fit_iterations", text="Iterations")
        row.prop(props, "fit_strength", text="Strength")
        row = box.row()
        row.prop(props, "fit_smooth", text="Smoothing")
        box.operator("gme.fit_gaussian", text="Fit to Gaussian", icon="SNAP_ON")
        box.operator("gme.refine", text="Subdivide + Fit", icon="AUTO")

        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        box.label(text="Bakes the proxy's world transform into the file.")
        box.operator("gme.export_mesh", text="Export Aligned Mesh", icon="EXPORT")

        box = layout.box()
        kiri_status = "enabled" if kiri_bridge.kiri_enabled() else "not found"
        box.label(text=f"KIRI 3DGS Render: {kiri_status}")
        box.label(text=f"Gaussian objects: {len(gaussians)}")


class GME_PT_unimags(bpy.types.Panel):
    bl_label = "UniMGS Deform"
    bl_idname = "GME_PT_unimags"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Gaussian Mesh Editor"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        props = context.scene.gme_props
        try:
            self._draw(layout, context, props)
        except Exception:
            import traceback
            traceback.print_exc()
            layout.label(text="Panel error - see system console", icon="ERROR")

    def _draw(self, layout, context, props):
        box = layout.box()
        box.label(text="Build Binding (Paper)", icon="LINKED")
        box.label(text="UniMGS: camera rays hit each Gaussian BBX corner")
        box.label(text="and bind every corner to a proxy triangle. A single")
        box.label(text="camera only covers the facing hemisphere, so virtual")
        box.label(text="surround cameras are added by default.")
        row = box.row()
        row.prop(props, "unimags_k", text="BBX k")
        row.prop(props, "unimags_surround", text="Surround Cams")
        box.operator("gme.unimags_build_binding",
                     text="Build UniMGS Binding", icon="LINKED")

        data = unimags_blender.get_binding(context.scene)
        if data is None:
            box.label(text="No UniMGS binding yet.", icon="INFO")
        else:
            s = data.stats()
            box = layout.box()
            box.label(text="Binding Stats", icon="FILE_TICK")

            def row(label, value):
                r = box.row()
                r.label(text=label)
                r.label(text=value)

            row("Gaussians", "{:,}".format(s["gaussian_count"]))
            row("Corner Rate", "{:.2f}%".format(s["corner_rate"] * 100.0))
            row("Fully Bound", "{:,}".format(s["fully_bound"]))
            row("Partial Bound", "{:,}".format(s["partial_bound"]))
            row("Unbound", "{:,}".format(s["unbound"]))
            box.operator("gme.unimags_clear_binding",
                         text="Clear Binding", icon="X")

        box = layout.box()
        box.label(text="Deformation (Eq.12/13)", icon="MOD_MESHDEFORM")
        box.label(text="Edit the proxy mesh, then apply: each corner's")
        box.label(text="deformation (one-ring gradient + barycentric")
        box.label(text="interpolation) is averaged over the 8 BBX corners,")
        box.label(text="updating position and covariance, then written")
        box.label(text="back to the KIRI object.")
        row = box.row()
        row.prop(props, "auto_follow_enabled", text="Auto Follow (live)")
        row.label(text="ON" if props.auto_follow_enabled else "OFF",
                  icon="CHECKBOX_HLT" if props.auto_follow_enabled else "CHECKBOX_DEHLT")
        row = box.row()
        row.prop(props, "gpu_accel", text="GPU Acceleration")
        row.label(text="GPU" if props.gpu_accel else "CPU")
        box.prop(props, "unimags_persist", text="Persist to Object (slow)")
        box.operator("gme.unimags_apply_deformation",
                     text="Apply UniMGS Deformation", icon="FILE_TICK")
