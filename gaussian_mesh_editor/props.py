"""Scene-level property group for the alignment workflow."""

import bpy


def _auto_follow_updated(self, context):
    print("[GME Auto-Follow] auto_follow_enabled -> {}".format(
        self.auto_follow_enabled))
    # Toggling is a reliable place to heal a lost poll timer (Edit-Mode
    # vertex drags never fire depsgraph handlers, so the checkbox click is
    # the only guaranteed wake-up).
    try:
        from .unimags import auto_follow
        auto_follow._ensure_timer()
    except Exception as exc:
        print("[GME Auto-Follow] timer heal failed: {}".format(exc))


class GME_Props(bpy.types.PropertyGroup):
    gaussian_obj: bpy.props.PointerProperty(
        name="Gaussian Scene",
        description="Gaussian object used as the alignment target",
        type=bpy.types.Object,
    )
    enable_target: bpy.props.BoolProperty(
        name="Alignment Target",
        description="Enable the Gaussian object as alignment target",
        default=False,
    )
    proxy_obj: bpy.props.PointerProperty(
        name="Proxy Mesh",
        description="Proxy mesh being aligned",
        type=bpy.types.Object,
    )
    # World-space bounds computed by the print/refresh operator.
    gaussian_center: bpy.props.FloatVectorProperty(size=3, precision=3)
    gaussian_size: bpy.props.FloatVectorProperty(size=3, precision=3)
    mesh_center: bpy.props.FloatVectorProperty(size=3, precision=3)
    mesh_size: bpy.props.FloatVectorProperty(size=3, precision=3)
    has_bounds: bpy.props.BoolProperty(default=False)
    # Last rough-alignment result (Phase 4).
    align_scale: bpy.props.FloatProperty(default=0.0, precision=3)
    align_offset: bpy.props.FloatVectorProperty(size=3, precision=3)
    # Phase 7: subdivision + vertex fitting.
    subdiv_levels: bpy.props.IntProperty(
        name="Subdivide Levels",
        description="Linear subdivision passes per click (faces x 4 per pass)",
        default=1,
        min=0,
        max=4,
    )
    fit_iterations: bpy.props.IntProperty(
        name="Fit Iterations",
        description="Snap + smooth rounds applied per fit run",
        default=5,
        min=1,
        max=50,
    )
    fit_strength: bpy.props.FloatProperty(
        name="Fit Strength",
        description="Fraction of the way each vertex moves toward its target per round",
        default=0.8,
        min=0.05,
        max=1.0,
        precision=2,
    )
    fit_mode: bpy.props.EnumProperty(
        name="Fit Mode",
        description="How each vertex's target position is chosen",
        items=[
            (
                "PLANE",
                "Plane Projection",
                "Fit a tangent plane through the k nearest splats and slide the vertex onto it",
            ),
            (
                "NEAREST",
                "Nearest Splat",
                "Pull the vertex toward the nearest splat center",
            ),
        ],
        default="PLANE",
    )
    fit_k: bpy.props.IntProperty(
        name="Neighbors k",
        description="Nearest splats used for the local tangent plane (PLANE mode)",
        default=3,
        min=1,
        max=16,
    )
    fit_smooth: bpy.props.FloatProperty(
        name="Smoothing",
        description="Laplacian smoothing weight applied between fit rounds",
        default=0.3,
        min=0.0,
        max=1.0,
        precision=2,
    )
    # Auto-follow (Phase 3 live update).
    auto_follow_enabled: bpy.props.BoolProperty(
        name="Auto Follow",
        description="Automatically drag bound Gaussians along while the proxy mesh is edited (live)",
        default=False,
        update=_auto_follow_updated,
    )
    # UniMGS (paper reproduction): BBX half-edge multiplier k.
    unimags_k: bpy.props.FloatProperty(
        name="BBX k",
        description="Half-edge multiplier k of the Gaussian bounding box (paper: k = 1.0)",
        default=1.0,
        min=0.1,
        max=10.0,
        precision=3,
    )
    # UniMGS: virtual surround cameras for binding coverage. A single
    # viewport camera only binds the hemisphere facing it (rays always hit
    # front faces first), so back-facing proxy faces can never drive
    # deformation. Extra virtual origins around the proxy restore coverage.
    unimags_surround: bpy.props.IntProperty(
        name="Surround Cameras",
        description=(
            "Virtual camera origins evenly spread on a sphere around the proxy, "
            "added to the scene cameras for ray binding. 0 = scene cameras only. "
            "Higher counts cover more faces but scale binding time linearly."
        ),
        default=6,
        min=0,
        max=32,
    )
    # Persist deformed data to the object IDProperty vs runtime cache only.
    unimags_persist: bpy.props.BoolProperty(
        name="Persist to Object",
        description=(
            "Also rewrite the large gaussian_data IDProperty (slow; Blender 5.2 "
            "can corrupt large overwrites). Off = runtime cache only, which the "
            "viewport renders from."
        ),
        default=False,
    )
    # UniMGS: GPU compute for the per-vertex deformation-gradient stage
    # (the Python loop that is the slowest part of Eq.12).
    gpu_accel: bpy.props.BoolProperty(
        name="GPU Acceleration",
        description=(
            "Compute the per-vertex one-ring deformation gradient on the GPU "
            "with a GLSL compute shader instead of the Python loop. Falls back "
            "to the CPU path automatically if the GPU path fails. Requires a "
            "GPU-context (viewport open); headless renders ignore this."
        ),
        default=False,
    )
