import json
import math
import os
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector

addons_root = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if addons_root not in sys.path:
    sys.path.insert(0, addons_root)

import ssbl

try:
    ssbl.unregister()
except Exception:
    pass
ssbl.register()

output_dir = Path(tempfile.gettempdir()) / "ssbl_mcp_monkey_recording"
frames_dir = output_dir / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

for path in frames_dir.glob("*.png"):
    path.unlink()

for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

scene = bpy.context.scene
scene.frame_set(1)
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.fps = 12
scene.world.color = (0.025, 0.03, 0.035)

if not hasattr(bpy.ops.mesh, "primitive_monkey_add"):
    raise RuntimeError("This Blender build does not expose bpy.ops.mesh.primitive_monkey_add")

bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0.0, 0.0, 1.1), rotation=(0.0, 0.0, 0.0))
obj = bpy.context.object
obj.name = "SSBL_Suzanne_Monkey_Cloth"
obj.data.name = "SSBL_Suzanne_Monkey_Mesh"

mat = bpy.data.materials.new("Suzanne Test Orange")
mat.diffuse_color = (1.0, 0.5, 0.12, 1.0)
obj.data.materials.append(mat)

z_values = [vert.co.z for vert in obj.data.vertices]
z_min = min(z_values)
z_max = max(z_values)
pin_threshold = z_max - (z_max - z_min) * 0.18
pin_indices = [vert.index for vert in obj.data.vertices if vert.co.z >= pin_threshold]
if not pin_indices:
    pin_indices = [max(obj.data.vertices, key=lambda vert: vert.co.z).index]
pin = obj.vertex_groups.new(name="ssbl_pin")
pin.add(pin_indices, 1.0, "ADD")

# Old preview behavior advanced the frame and could trigger visibility animation.
# Keep this in the test so the recording catches regressions.
obj.hide_viewport = False
obj.hide_render = False
obj.keyframe_insert(data_path="hide_viewport", frame=1)
obj.keyframe_insert(data_path="hide_render", frame=1)
obj.hide_viewport = True
obj.hide_render = True
obj.keyframe_insert(data_path="hide_viewport", frame=2)
obj.keyframe_insert(data_path="hide_render", frame=2)
scene.frame_set(1)

camera_data = bpy.data.cameras.new("Monkey_Record_Camera")
camera = bpy.data.objects.new("Monkey_Record_Camera", camera_data)
bpy.context.collection.objects.link(camera)
camera.location = (0.0, -6.0, 2.3)
camera.rotation_euler = (math.radians(72), 0.0, 0.0)
camera_data.lens = 40
scene.camera = camera

light_data = bpy.data.lights.new("Monkey_Record_Key", "AREA")
light = bpy.data.objects.new("Monkey_Record_Key", light_data)
bpy.context.collection.objects.link(light)
light.location = (0.0, -3.2, 5.0)
light_data.energy = 450
light_data.size = 4

settings = scene.ssbl_preview
settings.pin_vertex_group = "ssbl_pin"
settings.use_ground = False
settings.use_wall = False
settings.use_sphere = False
settings.static_collider_collection = None
settings.self_collision = False
settings.frame_count = 24
settings.substeps = 4
settings.iterations = 14
settings.dt = 1.0 / 30.0
settings.damping = 0.99
settings.stretch_compliance = 1e-6
settings.bend_compliance = 1e-4

diagnostics = []


def snapshot(label):
    bbox = [tuple(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    local_z = [vert.co.z for vert in obj.data.vertices]
    diagnostics.append(
        {
            "label": label,
            "frame": int(scene.frame_current),
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
            "visible_get": bool(obj.visible_get(view_layer=bpy.context.view_layer)),
            "mesh": obj.data.name,
            "vertex_count": len(obj.data.vertices),
            "pin_count": len(pin_indices),
            "min_local_z": min(local_z),
            "max_local_z": max(local_z),
            "bbox_min_z": min(point[2] for point in bbox),
            "bbox_max_z": max(point[2] for point in bbox),
        }
    )


def render_frame(label, index):
    snapshot(label)
    scene.render.filepath = str(frames_dir / f"{index:03d}_{label}.png")
    bpy.ops.render.opengl(write_still=True, view_context=False)


render_frame("before", 0)
session = ssbl.solver.start_preview(bpy.context, obj)
render_frame("start", 1)

for i in range(1, 9):
    ssbl.solver.step_preview(bpy.context, obj.name)
    render_frame(f"step{i}", i + 1)

ssbl.solver.request_stop(obj)
render_frame("stopped", 10)

summary = {
    "output_dir": str(output_dir),
    "frames_dir": str(frames_dir),
    "diagnostics": diagnostics,
    "frame_paths": [str(path) for path in sorted(frames_dir.glob("*.png"))],
}
summary_path = output_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("SSBL_MCP_MONKEY_SUMMARY", str(summary_path))
print(json.dumps(summary, ensure_ascii=False))
