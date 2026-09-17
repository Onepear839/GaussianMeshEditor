import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gaussian_mesh_editor as gme

gme.register()

import bpy

from gaussian_mesh_editor import export

# EXPORT icon validity
fn = bpy.types.UILayout.bl_rna.functions["operator"]
icons = {e.identifier for e in next(p for p in fn.parameters if p.identifier == "icon").enum_items}
print("EXPORT icon valid:", "EXPORT" in icons)

sc = bpy.context.scene


def make_cube(name, half, loc, scale=(1, 1, 1)):
    v = [(dx * half, dy * half, dz * half) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], faces)
    obj = bpy.data.objects.new(name, me)
    sc.collection.objects.link(obj)
    obj.location = loc
    obj.scale = scale
    return obj


gobj = make_cube("GaussianMesh", 1.0, (10, 0, 0))
gobj.data.attributes.new(name="f_dc_0", type="FLOAT_COLOR", domain="POINT")
pobj = make_cube("ProxyMesh", 0.25, (0, 5, -3), scale=(2, 1, 1))

props = sc.gme_props
bpy.context.view_layer.objects.active = pobj
bpy.ops.gme.set_proxy_mesh()
props.enable_target = True
bpy.ops.gme.rough_align()
print("aligned: loc", tuple(round(v, 3) for v in pobj.location), "scale", tuple(round(v, 3) for v in pobj.scale))

out_obj = os.path.join(tempfile.gettempdir(), "gme_t_out.obj")
out_ply = os.path.join(tempfile.gettempdir(), "gme_t_out.ply")
export.write_obj(out_obj, pobj)
export.write_ply(out_ply, pobj)

verts = []
with open(out_obj) as f:
    for line in f:
        if line.startswith("v "):
            verts.append(tuple(float(x) for x in line.split()[1:4]))
xs = [v[0] for v in verts]
ys = [v[1] for v in verts]
zs = [v[2] for v in verts]
print("obj verts:", len(verts), "| x:", round(min(xs), 4), round(max(xs), 4),
      "y:", round(min(ys), 4), round(max(ys), 4), "z:", round(min(zs), 4), round(max(zs), 4))
ok_obj = (
    len(verts) == 8
    and abs(min(xs) - 9) < 1e-3 and abs(max(xs) - 11) < 1e-3
    and abs(min(ys) + 0.5) < 1e-3 and abs(max(ys) - 0.5) < 1e-3
    and abs(min(zs) + 0.5) < 1e-3 and abs(max(zs) - 0.5) < 1e-3
)

pverts = fcount = 0
with open(out_ply) as f:
    for line in f:
        if line.startswith("element vertex"):
            pverts = int(line.split()[2])
        if line.startswith("element face"):
            fcount = int(line.split()[2])
print("ply verts:", pverts, "faces:", fcount)
# Phase 7: export triangulates, so 6 quads become 12 triangles.
ok_ply = pverts == 8 and fcount == 12

# operator path via EXEC_DEFAULT (bypasses the file dialog)
out2 = os.path.join(tempfile.gettempdir(), "gme_t_op.ply")
bpy.ops.gme.export_mesh('EXEC_DEFAULT', filepath=out2, format='PLY')
ok_op = os.path.exists(out2) and os.path.getsize(out2) > 0
print("PASS obj:", ok_obj, "| PASS ply:", ok_ply, "| PASS operator:", ok_op)

gme.unregister()
