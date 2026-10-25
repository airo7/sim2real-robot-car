"""
Line-art still of an exported run: the car in grey with Freestyle outlines and an orange
wireframe twin AHEAD frames later, through an orthographic isometric camera.

Usage (inside Blender):
    blender --background --python mjsim/blender_feature.py -- runs/L_rl.json out.png FRAME [AHEAD]
"""

import json
import sys

import bpy
from mathutils import Quaternion, Vector

DARK = (0.012, 0.012, 0.013, 1)
ORANGE = (1.0, 0.42, 0.08, 1)


def material(name, rgba, emission=0.0):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.55
    if emission:
        bsdf.inputs["Emission Color"].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def grey(rgba):
    v = 0.04 + 0.22 * (0.3 * rgba[0] + 0.6 * rgba[1] + 0.1 * rgba[2])
    return (v, v, v * 1.02, 1)


def build_car(run, frame, mat_for, wire=False):
    objs = []
    for g in run["geoms"]:
        b = g["body"]
        if g["type"] == "plane" or run["bodies"][b].startswith("obst"):
            continue
        size = g["size"]
        if g["type"] == "box":
            bpy.ops.mesh.primitive_cube_add(size=2)
            obj = bpy.context.object
            obj.scale = size
        elif g["type"] in ("cylinder", "capsule"):
            bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2 * size[1], vertices=40)
            obj = bpy.context.object
        else:
            continue
        bpy.ops.object.transform_apply(scale=True)
        body_pos, body_quat = Vector(frame["xpos"][b]), frame["xquat"][b]
        qb, qg = Quaternion(body_quat), Quaternion(g["quat"])
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = qb @ qg
        obj.location = body_pos + qb @ Vector(g["pos"])
        obj.data.materials.append(mat_for(g))
        if wire:
            mod = obj.modifiers.new("wire", "WIREFRAME")
            mod.thickness = 0.0016
        objs.append(obj)
    return objs


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    src, dst, k = argv[0], argv[1], int(argv[2])
    ahead = int(argv[3]) if len(argv) > 3 else 14
    run = json.load(open(src))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    bpy.ops.mesh.primitive_plane_add(size=40)
    bpy.context.object.data.materials.append(material("floor", DARK))

    tape_mat = material("tape", (0.30, 0.30, 0.31, 1))
    verts, faces = [], []
    half = run["tape_width"] / 2
    for p, q in run["tape_segments"]:
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / n * half, dx / n * half
        i = len(verts)
        verts += [(p[0] + nx, p[1] + ny, 0.0006), (p[0] - nx, p[1] - ny, 0.0006),
                  (q[0] - nx, q[1] - ny, 0.0006), (q[0] + nx, q[1] + ny, 0.0006)]
        faces.append((i, i + 1, i + 2, i + 3))
    mesh = bpy.data.meshes.new("tape")
    mesh.from_pydata(verts, [], faces)
    tape = bpy.data.objects.new("tape", mesh)
    scene.collection.objects.link(tape)
    tape.data.materials.append(tape_mat)

    chassis = run["bodies"].index("chassis")
    path = [f["xpos"][chassis] for f in run["frames"][max(0, k - 120):k + ahead]]
    curve = bpy.data.curves.new("path", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.0022
    spline = curve.splines.new("POLY")
    spline.points.add(len(path) - 1)
    for pt, xyz in zip(spline.points, path):
        pt.co = (xyz[0], xyz[1], 0.002, 1)
    path_obj = bpy.data.objects.new("path", curve)
    scene.collection.objects.link(path_obj)
    path_obj.data.materials.append(material("trace", ORANGE, emission=0.9))

    cache = {}

    def solid(g):
        key = tuple(round(c, 2) for c in grey(g["rgba"]))
        if key not in cache:
            cache[key] = material(f"car_{len(cache)}", grey(g["rgba"]))
        return cache[key]

    ghost_mat = material("ghost", ORANGE, emission=1.1)      # higher strengths tone-map to white
    ink = bpy.data.collections.new("ink")                     # only these get Freestyle outlines
    scene.collection.children.link(ink)
    for obj in build_car(run, run["frames"][k], solid) + [tape]:
        for c in obj.users_collection:
            c.objects.unlink(obj)
        ink.objects.link(obj)
    build_car(run, run["frames"][min(k + ahead, len(run["frames"]) - 1)], lambda g: ghost_mat, wire=True)

    centre = Vector(run["frames"][k]["xpos"][chassis])
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 0.95
    cam.rotation_euler = (0.9553, 0.0, 0.7854 + 3.1416)           # isometric: 54.7° tilt, 45° around z
    rot = cam.rotation_euler.to_matrix()
    # car in the right third, leaving room for a title on the left
    cam.location = centre + rot @ Vector((-0.20 * cam.data.ortho_scale, 0.0, 2.0))
    scene.camera = cam

    key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
    key.data.energy, key.data.size = 60, 1.5
    key.location = centre + Vector((-0.8, -0.6, 1.4))
    key.rotation_euler = (0.8, -0.5, 0.0)
    scene.collection.objects.link(key)
    rim = bpy.data.objects.new("rim", bpy.data.lights.new("rim", "AREA"))
    rim.data.energy, rim.data.size = 25, 1.0
    rim.location = centre + Vector((0.9, 0.9, 0.6))
    rim.rotation_euler = (-1.0, 0.7, 0.0)
    scene.collection.objects.link(rim)
    world = bpy.data.worlds.new("world")
    world.node_tree.nodes["Background"].inputs["Color"].default_value = DARK
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.4
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 2400, 1260
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.use_freestyle = True
    scene.render.line_thickness = 1.4
    layer = scene.view_layers[0]
    layer.use_freestyle = True
    sets = layer.freestyle_settings.linesets
    lineset = sets[0] if len(sets) else sets.new("ink")
    if lineset.linestyle is None:                 # the empty factory scene's line set has no style
        lineset.linestyle = bpy.data.linestyles.new("ink")
    lineset.linestyle.color = (0.62, 0.62, 0.64)
    lineset.select_by_collection = True
    lineset.collection = ink
    scene.render.filepath = dst
    bpy.ops.render.render(write_still=True)


main()
