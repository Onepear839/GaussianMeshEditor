import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gaussian_mesh_editor as gme

gme.register()

import bpy
from mathutils import Vector

from gaussian_mesh_editor import export, fit


def make_cube(name, half, loc, scale=(1, 1, 1)):
    v = [(dx * half, dy * half, dz * half) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], faces)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    obj.scale = scale
    return obj


def make_sphere(name, radius, count):
    pts = []
    for i in range(count):
        y = 1.0 - (i / (count - 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        phi = i * 2.399963229728653
        pts.append((r * math.cos(phi) * radius, y * radius, r * math.sin(phi) * radius))
    me = bpy.data.meshes.new(name)
    me.from_pydata(pts, [], [])
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (10, 0, 0)
    me.attributes.new(name="f_dc_0", type="FLOAT_COLOR", domain="POINT")
    return obj


def mean_nn(obj, kd):
    total = 0.0
    for v in obj.data.vertices:
        _co, _idx, d = kd.find(obj.matrix_world @ Vector(v.co))
        total += d
    return total / len(obj.data.vertices)


sc = bpy.context.scene
gobj = make_sphere("GaussianSphere", 2.0, 300)
pobj = make_cube("ProxyCube", 0.25, (0, 5, -3), scale=(2, 1, 1))

props = sc.gme_props
bpy.context.view_layer.objects.active = pobj
bpy.ops.gme.set_proxy_mesh()
props.enable_target = True
bpy.ops.gme.rough_align()
print("aligned loc:", tuple(round(v, 3) for v in pobj.location),
      "scale:", tuple(round(v, 3) for v in pobj.scale))

kd = fit.gaussian_kdtree(gobj)

# --- subdivision: 6 faces -> 6 * 4^2 = 96 ---
props.subdiv_levels = 2
bpy.ops.gme.subdivide()
f = len(pobj.data.polygons)
print("faces after subdiv x2:", f)
ok_sub = f == 96

# --- PLANE fit (defaults): mean NN distance must drop ---
props.fit_mode = "PLANE"
props.fit_k = 3
props.fit_iterations = 5
props.fit_strength = 0.8
props.fit_smooth = 0.3
before = mean_nn(pobj, kd)
bpy.ops.gme.fit_gaussian()
after = mean_nn(pobj, kd)
print("PLANE fit nn dist: {:.4f} -> {:.4f}".format(before, after))
ok_plane = after < before * 0.7

# --- NEAREST, strength 1, 1 iteration: exact snap onto splats ---
props.fit_mode = "NEAREST"
props.fit_iterations = 1
props.fit_strength = 1.0
props.fit_smooth = 0.0
bpy.ops.gme.fit_gaussian()
dist = mean_nn(pobj, kd)
print("NEAREST snap residual:", dist)
ok_snap = dist < 1e-3

# --- one-click refine: 96 faces -> 384, and fit runs ---
props.subdiv_levels = 1
props.fit_mode = "PLANE"
props.fit_iterations = 3
props.fit_strength = 0.8
props.fit_smooth = 0.3
f_before = len(pobj.data.polygons)
bpy.ops.gme.refine()
f_after = len(pobj.data.polygons)
print("refine faces:", f_before, "->", f_after)
ok_refine = f_after == f_before * 4

# --- export: everything triangulated (384 quads -> 768 triangles) ---
out_obj = os.path.join(tempfile.gettempdir(), "gme_t7.obj")
out_ply = os.path.join(tempfile.gettempdir(), "gme_t7.ply")
export.write_obj(out_obj, pobj)
export.write_ply(out_ply, pobj)

n_verts = n_faces = 0
all_tri = True
with open(out_obj) as f:
    for line in f:
        if line.startswith("v "):
            n_verts += 1
        elif line.startswith("f "):
            n_faces += 1
            if len(line.split()[1:]) != 3:
                all_tri = False
print("obj:", n_verts, "verts,", n_faces, "faces, all triangles:", all_tri)

ply_verts = ply_faces = 0
all_tri_ply = True
body = False
seen_verts = 0
seen_faces = 0
with open(out_ply) as f:
    for line in f:
        line = line.strip()
        if not body:
            if line == "end_header":
                body = True
            elif line.startswith("element vertex"):
                ply_verts = int(line.split()[2])
            elif line.startswith("element face"):
                ply_faces = int(line.split()[2])
            continue
        parts = line.split()
        if seen_verts < ply_verts:
            seen_verts += 1
            continue
        if seen_faces < ply_faces:
            seen_faces += 1
            if parts[0] != "3":
                all_tri_ply = False
print("ply:", ply_faces, "faces, all triangles:", all_tri_ply)
ok_export = all_tri and all_tri_ply and n_faces == 768 and ply_faces == 768

# --- reset still restores the transform after topology changes ---
bpy.ops.gme.reset_proxy()
loc_ok = all(abs(a - b) < 1e-4 for a, b in zip(pobj.location, (0, 5, -3)))
scale_ok = all(abs(a - b) < 1e-4 for a, b in zip(pobj.scale, (2, 1, 1)))
print("reset loc:", tuple(round(v, 3) for v in pobj.location),
      "scale:", tuple(round(v, 3) for v in pobj.scale))
ok_reset = loc_ok and scale_ok

print("PASS subdivide:", ok_sub, "| PASS plane_fit:", ok_plane, "| PASS snap:", ok_snap,
      "| PASS refine:", ok_refine, "| PASS export_tri:", ok_export, "| PASS reset:", ok_reset)

gme.unregister()
