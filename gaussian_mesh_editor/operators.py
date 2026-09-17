"""Phase 2/3/4/5/6/7: proxy mesh, bounds, alignment, reset, export,
subdivide and fit, plus the UniMGS paper reproduction (BBX ray binding
and Eq.12/13 deformation).

Phase 2: designate an existing scene mesh as the alignment proxy; the
object keeps its name and is moved with the native G / R / S tools.
Phase 3: compute and print Gaussian / Mesh world-space bounds, center
and size.
Phase 4: rough align - center the proxy on the Gaussian bounds and
apply a uniform scale, then fine-tune manually.
Phase 5: reset - restore the transform snapshot taken when the proxy
was set.
Phase 6: export the aligned proxy mesh (world transform baked) as
OBJ / PLY.
Phase 7: linear subdivision + iterative vertex fitting onto the
Gaussian cloud, plus a one-click refine operator.
UniMGS: build the camera-ray BBX-corner binding and apply the
barycentric-corner deformation (Eq.12/13).
"""

import bmesh
import bpy

from . import bbox, export, fit, kiri_bridge, metrics, unimags
from .unimags import blender as unimags_blender


def _build_proxy_material():
    """Green semi-transparent emission material, owned by this add-on."""
    mat = bpy.data.materials.get("GME_ProxyMaterial")
    if mat is None:
        mat = bpy.data.materials.new("GME_ProxyMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        em = nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (0.15, 1.0, 0.3, 1.0)
        em.inputs["Strength"].default_value = 1.0
        tr = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = 0.35
        links.new(em.outputs["Emission"], mix.inputs[1])
        links.new(tr.outputs["BSDF"], mix.inputs[2])
        links.new(mix.outputs["Shader"], out.inputs["Surface"])
        mat.blend_method = "BLEND"
    return mat


def _prepare_proxy_object(obj):
    """Apply independent wireframe display, snapshot initial transform.

    The object is not renamed; only its viewport display and custom
    properties are touched. The snapshot feeds the Reset operator
    (Phase 5).
    """
    obj.display_type = "WIRE"
    obj.visible_shadow = False
    mat = _build_proxy_material()
    if mat.name not in obj.data.materials:
        obj.data.materials.append(mat)
    obj["gme_initial_location"] = [round(v, 6) for v in obj.location]
    obj["gme_initial_rotation"] = [round(v, 6) for v in obj.rotation_euler]
    obj["gme_initial_scale"] = [round(v, 6) for v in obj.scale]
    return obj


def _resolve_gaussian(context):
    """Return the alignment-target Gaussian object, auto-detecting if needed."""
    props = context.scene.gme_props
    if props.gaussian_obj is not None and kiri_bridge.is_gaussian_object(props.gaussian_obj):
        return props.gaussian_obj
    gaussians = kiri_bridge.find_gaussian_objects(context.scene)
    if gaussians:
        props.gaussian_obj = gaussians[0]
        return gaussians[0]
    return None


def _compute_and_store_bounds(context):
    """Compute world bounds for Gaussian + proxy, store into scene props.

    Returns (g_center, g_size, m_center, m_size) or None on failure.
    """
    props = context.scene.gme_props
    g_obj = _resolve_gaussian(context)
    m_obj = props.proxy_obj
    if g_obj is None or m_obj is None:
        return None
    g_bounds = bbox.gaussian_world_bounds(g_obj)
    m_bounds = bbox.mesh_world_bounds(m_obj)
    if g_bounds is None or m_bounds is None:
        return None
    g_center, g_size = bbox.bounds_center_size(*g_bounds)
    m_center, m_size = bbox.bounds_center_size(*m_bounds)
    props.gaussian_center = g_center
    props.gaussian_size = g_size
    props.mesh_center = m_center
    props.mesh_size = m_size
    props.has_bounds = True
    return g_center, g_size, m_center, m_size


class GME_OT_set_proxy(bpy.types.Operator):
    bl_idname = "gme.set_proxy_mesh"
    bl_label = "Set Active as Proxy Mesh"
    bl_description = "Use the active mesh object as the alignment proxy"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object first.")
            return {"CANCELLED"}
        if kiri_bridge.is_gaussian_object(obj):
            self.report(
                {"ERROR"},
                "This object is a Gaussian object. Pick a regular proxy mesh.",
            )
            return {"CANCELLED"}
        _prepare_proxy_object(obj)
        context.scene.gme_props.proxy_obj = obj
        _compute_and_store_bounds(context)
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        self.report({"INFO"}, f"Proxy mesh set: {obj.name}")
        return {"FINISHED"}


class GME_OT_print_bounds(bpy.types.Operator):
    bl_idname = "gme.print_bounds"
    bl_label = "Print Bounds"
    bl_description = "Compute world-space bounds and print them to the console"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.scene.gme_props.proxy_obj is not None

    def execute(self, context):
        result = _compute_and_store_bounds(context)
        if result is None:
            self.report({"ERROR"}, "Need both a Gaussian object and a proxy mesh.")
            return {"CANCELLED"}
        g_center, g_size, m_center, m_size = result
        print("[GME] Gaussian Center:", tuple(round(v, 6) for v in g_center))
        print("[GME] Gaussian Size:", tuple(round(v, 6) for v in g_size))
        print("[GME] Mesh Center:", tuple(round(v, 6) for v in m_center))
        print("[GME] Mesh Size:", tuple(round(v, 6) for v in m_size))
        self.report({"INFO"}, "Bounds printed to the console.")
        return {"FINISHED"}


class GME_OT_rough_align(bpy.types.Operator):
    bl_idname = "gme.rough_align"
    bl_label = "Rough Align"
    bl_description = "Center the proxy mesh on the Gaussian bounds and apply a uniform scale"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.gme_props.proxy_obj is not None

    def execute(self, context):
        props = context.scene.gme_props
        if not props.enable_target:
            self.report({"ERROR"}, "Enable the Alignment Target checkbox first.")
            return {"CANCELLED"}
        g_obj = _resolve_gaussian(context)
        m_obj = props.proxy_obj
        if g_obj is None or m_obj is None:
            self.report({"ERROR"}, "Need both a Gaussian object and a proxy mesh.")
            return {"CANCELLED"}
        g_bounds = bbox.gaussian_world_bounds(g_obj)
        m_bounds = bbox.mesh_world_bounds(m_obj)
        if g_bounds is None or m_bounds is None:
            self.report({"ERROR"}, "Could not compute world bounds.")
            return {"CANCELLED"}
        g_center, g_size = bbox.bounds_center_size(*g_bounds)
        m_center, m_size = bbox.bounds_center_size(*m_bounds)

        # Uniform scale: map the proxy's longest world axis onto the Gaussian's.
        factor = metrics.size_ratio(g_size, m_size)
        if factor > 0.0 and abs(factor - 1.0) > 1e-6:
            s = m_obj.scale
            m_obj.scale = (s[0] * factor, s[1] * factor, s[2] * factor)

        # Uniform scaling moves the world center along the line
        # (object origin -> old center); re-derive it analytically so no
        # depsgraph update is needed.
        origin = m_obj.matrix_world.translation
        m_center2 = origin + factor * (m_center - origin)
        offset = g_center - m_center2
        loc = m_obj.location
        m_obj.location = (loc[0] + offset[0], loc[1] + offset[1], loc[2] + offset[2])

        props.align_scale = factor
        props.align_offset = offset

        # Refresh the transform before re-measuring bounds (matrix_world
        # is only re-evaluated on a depsgraph update).
        bpy.context.view_layer.update()

        result = _compute_and_store_bounds(context)
        if result is not None:
            _, _, m_center_final, _ = result
            dist = metrics.center_distance(g_center, m_center_final)
            print("[GME] Rough align done.")
            print("[GME] Uniform scale factor:", round(factor, 6))
            print("[GME] Center offset applied:", tuple(round(v, 6) for v in offset))
            print("[GME] Residual center distance:", round(dist, 6))
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        self.report({"INFO"}, f"Rough aligned (scale {factor:.3f}x).")
        return {"FINISHED"}


class GME_OT_reset_proxy(bpy.types.Operator):
    bl_idname = "gme.reset_proxy"
    bl_label = "Reset Transform"
    bl_description = "Restore the transform the proxy had when it was set"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.gme_props.proxy_obj is not None

    def execute(self, context):
        props = context.scene.gme_props
        obj = props.proxy_obj
        if obj is None:
            self.report({"ERROR"}, "No proxy mesh set.")
            return {"CANCELLED"}
        keys = ("gme_initial_location", "gme_initial_rotation", "gme_initial_scale")
        if not all(k in obj for k in keys):
            self.report({"ERROR"}, "No initial transform snapshot. Set the proxy mesh again.")
            return {"CANCELLED"}
        obj.location = tuple(obj["gme_initial_location"])
        obj.rotation_euler = tuple(obj["gme_initial_rotation"])
        obj.scale = tuple(obj["gme_initial_scale"])

        props.align_scale = 0.0
        props.align_offset = (0.0, 0.0, 0.0)

        bpy.context.view_layer.update()
        result = _compute_and_store_bounds(context)
        if result is not None:
            print("[GME] Proxy reset to initial transform:", obj.name)
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        self.report({"INFO"}, f"Proxy reset: {obj.name}")
        return {"FINISHED"}


class GME_OT_export_mesh(bpy.types.Operator):
    bl_idname = "gme.export_mesh"
    bl_label = "Export Aligned Mesh"
    bl_description = "Export the proxy mesh with its world transform baked (OBJ or PLY)"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    format: bpy.props.EnumProperty(
        name="Format",
        items=[
            ("OBJ", "OBJ", "Wavefront OBJ"),
            ("PLY", "PLY", "Stanford PLY (ASCII)"),
        ],
        default="OBJ",
    )

    @classmethod
    def poll(cls, context):
        return context.scene.gme_props.proxy_obj is not None

    def invoke(self, context, event):
        obj = context.scene.gme_props.proxy_obj
        if obj is None:
            self.report({"ERROR"}, "No proxy mesh set.")
            return {"CANCELLED"}
        name = obj.name.replace("/", "_").replace("\\", "_")
        self.filepath = "//{}_aligned.{}".format(name, self.format.lower())
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        obj = context.scene.gme_props.proxy_obj
        if obj is None:
            self.report({"ERROR"}, "No proxy mesh set.")
            return {"CANCELLED"}
        filepath = bpy.path.abspath(self.filepath)
        if not filepath.lower().endswith("." + self.format.lower()):
            filepath += "." + self.format.lower()
        if self.format == "OBJ":
            export.write_obj(filepath, obj)
        else:
            export.write_ply(filepath, obj)
        print("[GME] Exported aligned mesh:", filepath)
        self.report({"INFO"}, "Exported: {}".format(filepath))
        return {"FINISHED"}


class GME_OT_subdivide(bpy.types.Operator):
    bl_idname = "gme.subdivide"
    bl_label = "Subdivide Proxy"
    bl_description = "Linear midpoint subdivision of the proxy mesh (faces x 4 per level)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.gme_props.proxy_obj is not None

    def execute(self, context):
        props = context.scene.gme_props
        obj = props.proxy_obj
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "No proxy mesh set.")
            return {"CANCELLED"}
        levels = props.subdiv_levels
        if levels <= 0:
            self.report({"ERROR"}, "Set Subdivide Levels to at least 1.")
            return {"CANCELLED"}
        me = obj.data
        faces_before = len(me.polygons)
        estimated = faces_before * (4 ** levels)
        if estimated > 1_000_000:
            self.report(
                {"ERROR"},
                "Estimated {:,} faces exceeds the 1,000,000 guard. Lower the levels.".format(
                    estimated
                ),
            )
            return {"CANCELLED"}
        bm = bmesh.new()
        bm.from_mesh(me)
        for _ in range(levels):
            # grid_fill splits each quad into 4 quads (edge-only subdivision
            # would turn them into n-gons).
            bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=1, use_grid_fill=True)
        bm.to_mesh(me)
        bm.free()
        me.update()
        print(
            "[GME] Subdivided {} -> {} faces ({} level(s)).".format(
                faces_before, len(me.polygons), levels
            )
        )
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        self.report(
            {"INFO"},
            "Subdivided: {} -> {} faces.".format(faces_before, len(me.polygons)),
        )
        return {"FINISHED"}


class GME_OT_fit_gaussian(bpy.types.Operator):
    bl_idname = "gme.fit_gaussian"
    bl_label = "Fit to Gaussian"
    bl_description = "Iteratively pull proxy vertices onto the Gaussian cloud"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.gme_props.proxy_obj is not None

    def execute(self, context):
        props = context.scene.gme_props
        obj = props.proxy_obj
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "No proxy mesh set.")
            return {"CANCELLED"}
        g_obj = _resolve_gaussian(context)
        if g_obj is None:
            self.report({"ERROR"}, "No Gaussian object found.")
            return {"CANCELLED"}
        bpy.context.view_layer.update()
        try:
            before, after = fit.fit_mesh(
                obj,
                g_obj,
                iterations=props.fit_iterations,
                strength=props.fit_strength,
                mode=props.fit_mode,
                k=props.fit_k,
                smooth=props.fit_smooth,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        print(
            "[GME] Fit done. Mean NN distance {:.4f} -> {:.4f} (world units).".format(
                before, after
            )
        )
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        self.report(
            {"INFO"},
            "Fit: NN distance {:.4f} -> {:.4f}.".format(before, after),
        )
        return {"FINISHED"}


class GME_OT_refine(bpy.types.Operator):
    bl_idname = "gme.refine"
    bl_label = "Subdivide + Fit"
    bl_description = "One click: subdivide per the levels, then run the fit loop"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.gme_props.proxy_obj is not None

    def execute(self, context):
        props = context.scene.gme_props
        if props.subdiv_levels > 0:
            res = GME_OT_subdivide.execute(self, context)
            if res != {"FINISHED"}:
                return res
        res = GME_OT_fit_gaussian.execute(self, context)
        if res != {"FINISHED"}:
            self.report({"WARNING"}, "Subdivided, but the fit failed.")
            return {"FINISHED"}
        return {"FINISHED"}


class GME_OT_unimags_build_binding(bpy.types.Operator):
    bl_idname = "gme.unimags_build_binding"
    bl_label = "Build UniMGS Binding"
    bl_description = "UniMGS: cast camera rays onto each Gaussian BBX corner and bind every corner to a proxy triangle"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.scene.gme_props.proxy_obj is not None

    def execute(self, context):
        props = context.scene.gme_props
        ws = getattr(context, "workspace", None)
        try:
            if ws is not None:
                ws.status_text_set("GME: building UniMGS binding, please wait ...")
            data = unimags_blender.build_binding(context.scene, k=props.unimags_k)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            if ws is not None:
                ws.status_text_set(None)
        s = data.stats()
        print("=== UniMGS Binding ===")
        print("Gaussians:        {:,}".format(s["gaussian_count"]))
        print("Corner slots:     {:,}".format(s["corner_slots"]))
        print("Valid corners:    {:,} ({:.2f}%)".format(
            s["valid_corners"], s["corner_rate"] * 100.0))
        print("Fully bound:      {:,}".format(s["fully_bound"]))
        print("Partially bound:  {:,}".format(s["partial_bound"]))
        print("Unbound:          {:,}".format(s["unbound"]))
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        self.report(
            {"INFO"},
            "UniMGS binding: {:.1f}% corners bound".format(s["corner_rate"] * 100.0),
        )
        return {"FINISHED"}


class GME_OT_unimags_apply_deformation(bpy.types.Operator):
    bl_idname = "gme.unimags_apply_deformation"
    bl_label = "Apply UniMGS Deformation"
    bl_description = "Propagate the proxy deformation to every Gaussian through its BBX corners (paper Eq.12/13)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        props = context.scene.gme_props
        return (props.proxy_obj is not None and
                unimags_blender.get_binding(context.scene) is not None)

    def execute(self, context):
        props = context.scene.gme_props
        try:
            data, moved, max_disp = unimags_blender.apply_deformation(
                context.scene, persist=props.unimags_persist
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        print("=== UniMGS Deformation ===")
        print("Moved Gaussians:  {:,}".format(moved))
        print("Max displacement: {:.6f}".format(max_disp))
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        self.report(
            {"INFO"},
            "UniMGS: deformed {:,} Gaussians (max {:.4f})".format(moved, max_disp),
        )
        return {"FINISHED"}


class GME_OT_unimags_clear_binding(bpy.types.Operator):
    bl_idname = "gme.unimags_clear_binding"
    bl_label = "Clear UniMGS Binding"
    bl_description = "Discard the UniMGS binding cache (do this after retopologizing the proxy mesh)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return unimags_blender.get_binding(context.scene) is not None

    def execute(self, context):
        unimags_blender.clear_binding(context.scene)
        self.report({"INFO"}, "UniMGS binding cleared.")
        return {"FINISHED"}
