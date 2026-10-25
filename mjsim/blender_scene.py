"""
Rebuild a run exported by mjsim/export_blender.py in Blender and render it to video.
MuJoCo and Blender both use z-up, metres and (w, x, y, z) quaternions, so poses copy across unchanged.

Usage (inside Blender, not the project's Python):
    blender --background --python mjsim/blender_scene.py -- runs/figure8_rl.json runs/figure8.mp4 [--cycles]
"""

import json
import sys

import bpy

CAMERA_OFFSET = (-0.8, -0.8, 0.7)      # m, world frame, from the chassis


def material(name, rgba, roughness=0.6):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def wood_material():
    mat = bpy.data.materials.new("laminate")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (0.6, 12.0, 1.0)        # long planks along x
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.45, 0.28, 0.14, 1)
    ramp.color_ramp.elements[1].color = (0.68, 0.48, 0.28, 1)
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def add_geom(g, parent):
    kind, size = g["type"], g["size"]
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=2)
        obj = bpy.context.object
        obj.scale = size
    elif kind in ("cylinder", "capsule"):
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2 * size[1], vertices=32)
        obj = bpy.context.object
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size[0])
        obj = bpy.context.object
    else:
        return None
    obj.name = g["name"] or f"geom_{kind}"
    obj.parent = parent
    obj.location = g["pos"]
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = g["quat"]
    key = "mat_" + "_".join(f"{c:.2f}" for c in g["rgba"])
    obj.data.materials.append(material(key, g["rgba"], 0.35 if g["rgba"][0] > 0.8 else 0.6))
    if kind != "box":
        if hasattr(bpy.ops.object, "shade_smooth_by_angle"):     # keep cylinder caps flat
            bpy.ops.object.shade_smooth_by_angle(angle=0.52)
        else:
            bpy.ops.object.shade_smooth()
    return obj


def add_tape(segments, width):
    verts, faces = [], []
    half = width / 2
    for (p, q) in segments:
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / n * half, dx / n * half
        k = len(verts)
        verts += [(p[0] + nx, p[1] + ny, 0.0008), (p[0] - nx, p[1] - ny, 0.0008),
                  (q[0] - nx, q[1] - ny, 0.0008), (q[0] + nx, q[1] + ny, 0.0008)]
        faces.append((k, k + 1, k + 2, k + 3))
    mesh = bpy.data.meshes.new("tape")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("tape", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material("tape", (0.02, 0.02, 0.02, 1), 0.8))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    src, dst = argv[0], argv[1]
    cycles = "--cycles" in argv
    run = json.load(open(src))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = run["fps"]
    scene.frame_start, scene.frame_end = 1, len(run["frames"])

    bpy.ops.mesh.primitive_plane_add(size=30)
    bpy.context.object.data.materials.append(wood_material())
    add_tape(run["tape_segments"], run["tape_width"])

    empties = {}
    for g in run["geoms"]:
        if g["type"] == "plane":
            continue
        b = g["body"]
        if b not in empties:
            empty = bpy.data.objects.new(run["bodies"][b], None)
            empty.rotation_mode = "QUATERNION"
            bpy.context.collection.objects.link(empty)
            empties[b] = empty
        add_geom(g, empties[b])

    for f, frame in enumerate(run["frames"], start=1):
        for b, empty in empties.items():
            empty.location = frame["xpos"][b]
            empty.rotation_quaternion = frame["xquat"][b]
            empty.keyframe_insert("location", frame=f)
            empty.keyframe_insert("rotation_quaternion", frame=f)

    chassis = empties[run["bodies"].index("chassis")]
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    track = cam.constraints.new("TRACK_TO")
    track.target = chassis
    for f, frame in enumerate(run["frames"], start=1):
        x, y, z = frame["xpos"][run["bodies"].index("chassis")]
        cam.location = (x + CAMERA_OFFSET[0], y + CAMERA_OFFSET[1], z + CAMERA_OFFSET[2])
        cam.keyframe_insert("location", frame=f)

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (0.6, 0.2, 0.8)
    bpy.context.collection.objects.link(sun)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world

    if cycles:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 64
    else:
        engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    if hasattr(scene.render.image_settings, "media_type"):     # Blender 5: must be VIDEO before the FFMPEG format
        scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.filepath = dst
    bpy.ops.render.render(animation=True)


main()
