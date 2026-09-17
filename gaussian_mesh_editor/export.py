"""Phase 6/7: export the aligned proxy mesh with its world transform baked.

Writes OBJ / PLY directly (no Blender exporter dependencies) so the
output is deterministic and vertices are in world space. Every polygon
is fan-triangulated on export (Phase 7), so downstream engines receive
triangle-only files.
"""

from mathutils import Vector


def _world_vertices(obj):
    return [tuple(obj.matrix_world @ Vector(v.co)) for v in obj.data.vertices]


def _polygons(obj):
    return [list(p.vertices) for p in obj.data.polygons]


def _triangulate(faces):
    """Fan-triangulate every polygon (quads become two triangles)."""
    out = []
    for face in faces:
        n = len(face)
        if n < 3:
            continue
        if n == 3:
            out.append(face)
        else:
            out.extend([(face[0], face[i], face[i + 1]) for i in range(1, n - 1)])
    return out


def _face_normal(verts, face):
    a = Vector(verts[face[0]])
    b = Vector(verts[face[1]])
    c = Vector(verts[face[2]])
    n = (b - a).cross(c - a)
    if n.length_squared > 1e-12:
        n.normalize()
    else:
        n = Vector((0.0, 0.0, 1.0))
    return tuple(n)


def write_obj(filepath, obj):
    verts = _world_vertices(obj)
    faces = _triangulate(_polygons(obj))
    normals = [_face_normal(verts, face) for face in faces]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Exported by Gaussian Mesh Editor (world space, triangulated)\n")
        for v in verts:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*v))
        for n in normals:
            f.write("vn {:.6f} {:.6f} {:.6f}\n".format(*n))
        for i, face in enumerate(faces):
            f.write("f " + " ".join("{}//{}".format(idx + 1, i + 1) for idx in face) + "\n")


def write_ply(filepath, obj):
    verts = _world_vertices(obj)
    faces = _triangulate(_polygons(obj))
    with open(filepath, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {}\n".format(len(verts)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("element face {}\n".format(len(faces)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(*v))
        for face in faces:
            f.write("3 {} {} {}\n".format(*face))
