import math
import os
import json
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
SSBL_ROOT = Path(ADDONS_ROOT) / "ssbl"
TOOLS_DIR = SSBL_ROOT / "tools"
for path in (ADDONS_ROOT, str(TOOLS_DIR)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ssbl
from ssbl.xpbd_core import to_local
from wring_towel_smoke import (
    RADIAL_SEGMENTS,
    TWIST_RADIANS,
    WRING_FRAME_COUNT,
    _make_towel,
    _step_wring_frame,
)


OBJECT_NAME = "SSBL_Wring_Towel_CurrentScene"
CACHE_MODIFIER_NAME = "SSBL Wring Towel H0 Cache"
TEXT_NAME = "SSBL_Wring_Towel_Instructions"
PIN_GROUP = "ssbl_pin"


def _ensure_ssbl_registered():
    if not hasattr(bpy.context.scene, "ssbl_preview"):
        ssbl.register()


def _remove_old_test_objects():
    prefixes = (
        OBJECT_NAME,
        "SSBL_Wring_Towel",
        TEXT_NAME,
        "SSBL_Wring_Camera",
        "SSBL_Wring_Key",
    )
    for obj in list(bpy.context.scene.objects):
        if any(obj.name.startswith(prefix) for prefix in prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.name.startswith("SSBL_Wring_Towel"):
            bpy.data.meshes.remove(mesh)


def _material(name, color):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def _stripe_towel(obj):
    obj.data.materials.append(_material("SSBL_Towel_Blue", (0.22, 0.42, 0.9, 1.0)))
    obj.data.materials.append(_material("SSBL_Towel_Light_Stripe", (0.76, 0.84, 1.0, 1.0)))
    obj.data.materials.append(_material("SSBL_Towel_Orange_Stripe", (1.0, 0.52, 0.18, 1.0)))
    for poly in obj.data.polygons:
        radial_index = int(poly.index) % RADIAL_SEGMENTS
        if radial_index in (0, 1, 2):
            poly.material_index = 2
        elif radial_index % 6 == 0:
            poly.material_index = 1
        else:
            poly.material_index = 0
    obj.show_wire = True


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_camera_and_light():
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("SSBL_Wring_Camera_Data")
    camera = bpy.data.objects.new("SSBL_Wring_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (3.2, -5.2, 2.35)
    _look_at(camera, (0.0, 0.0, 0.8))
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 4.6
    scene.camera = camera

    light_data = bpy.data.lights.new("SSBL_Wring_Key_Data", "AREA")
    light = bpy.data.objects.new("SSBL_Wring_Key", light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, -3.0, 4.5)
    light_data.energy = 550
    light_data.size = 5.0


def _add_instruction_text():
    font_curve = bpy.data.curves.new(TEXT_NAME, "FONT")
    font_curve.body = (
        "SSBL Wring Towel Test\n"
        "Hardness = 0.0, Volume/Pressure = Off, Hidden tether count = 0\n"
        "Playback uses a generated PC2 Mesh Cache from frame 1 to 81."
    )
    font_curve.align_x = "LEFT"
    font_curve.align_y = "CENTER"
    font_curve.size = 0.12
    obj = bpy.data.objects.new(TEXT_NAME, font_curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (-2.1, -1.1, 1.8)
    obj.rotation_euler = (math.radians(70.0), 0.0, 0.0)


def _configure_settings():
    scene = bpy.context.scene
    settings = scene.ssbl_preview
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = PIN_GROUP
    settings.hardness = 0.0
    settings.hardness_initialized = True
    settings.use_volume_pressure = False
    settings.self_collision = False
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 2
    settings.max_self_collision_neighbors = 32
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.static_collider_collection = None
    settings.collision_margin = 0.01
    settings.cloth_thickness = 0.02
    settings.substeps = 16
    settings.iterations = 4
    settings.damping = 0.995
    settings.frame_count = WRING_FRAME_COUNT + 1
    scene.frame_start = 1
    scene.frame_end = max(int(scene.frame_end), WRING_FRAME_COUNT + 1)
    scene.frame_set(1)
    return settings


def _cache_path():
    if bpy.data.filepath:
        root = Path(bpy.path.abspath("//"))
    else:
        root = Path(bpy.app.tempdir)
    path = root / "ssbl_cache" / "SSBL_Wring_Towel_CurrentScene_h0_no_pressure.pc2"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_pc2_header(handle, vertex_count, start_frame, sample_count):
    handle.write(
        struct.pack(
            "<12siiffi",
            b"POINTCACHE2\0",
            1,
            int(vertex_count),
            float(start_frame),
            1.0,
            int(sample_count),
        )
    )


def _write_pc2_sample(handle, world_positions, matrix_world_inv):
    local = to_local(np.asarray(world_positions, dtype=np.float32), matrix_world_inv)
    handle.write(np.ascontiguousarray(local, dtype="<f4").tobytes())


def _bind_cache(obj, path):
    for modifier in list(obj.modifiers):
        if modifier.name == CACHE_MODIFIER_NAME:
            obj.modifiers.remove(modifier)
    modifier = obj.modifiers.new(CACHE_MODIFIER_NAME, "MESH_CACHE")
    modifier.cache_format = "PC2"
    modifier.filepath = str(path)
    modifier.frame_start = 1.0
    modifier.frame_scale = 1.0
    obj["ssbl_wring_towel_cache"] = str(path)


def _build_and_cache_wring_test(obj):
    scene = bpy.context.scene
    context = bpy.context
    source_mesh = obj.data
    session = ssbl.solver.start_preview(context, obj)
    slot = session.slots[obj.name]
    path = _cache_path()
    finite = True
    min_z = float("inf")
    max_radius = 0.0
    try:
        with open(path, "wb") as handle:
            _write_pc2_header(handle, len(slot.cloth.positions_world), 1, WRING_FRAME_COUNT + 1)
            _write_pc2_sample(handle, slot.current_positions_world, slot.cloth.matrix_world_inv)
            for frame in range(1, WRING_FRAME_COUNT + 1):
                scene.frame_set(frame + 1)
                positions = _step_wring_frame(session, obj, frame, WRING_FRAME_COUNT)
                finite = finite and bool(np.isfinite(positions).all())
                min_z = min(min_z, float(np.min(positions[:, 2])))
                radius = np.sqrt(np.square(positions[:, 1]) + np.square(positions[:, 2] - 1.25))
                max_radius = max(max_radius, float(np.max(radius)))
                _write_pc2_sample(handle, positions, slot.cloth.matrix_world_inv)
    finally:
        ssbl.solver.request_stop(obj)
        obj.data = source_mesh
        scene.frame_set(1)
    _bind_cache(obj, path)
    return {
        "cache": str(path),
        "finite": finite,
        "min_z": min_z,
        "max_radius": max_radius,
        "tethers": len(slot.cloth.lra_edges),
        "frames": WRING_FRAME_COUNT + 1,
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "twist_degrees": math.degrees(TWIST_RADIANS),
    }


def main():
    _ensure_ssbl_registered()
    _remove_old_test_objects()
    _configure_settings()
    towel = _make_towel()
    towel.name = OBJECT_NAME
    towel.data.name = f"{OBJECT_NAME}_Mesh"
    _stripe_towel(towel)
    result = _build_and_cache_wring_test(towel)
    _add_camera_and_light()
    _add_instruction_text()

    bpy.ops.object.select_all(action="DESELECT")
    towel.select_set(True)
    bpy.context.view_layer.objects.active = towel
    bpy.context.scene.frame_set(1)

    result["object"] = towel.name
    result["hardness"] = float(bpy.context.scene.ssbl_preview.hardness)
    result["use_volume_pressure"] = bool(bpy.context.scene.ssbl_preview.use_volume_pressure)
    summary_dir = SSBL_ROOT / "recordings" / "wring_towel_current_scene"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    result["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SSBL_CURRENT_SCENE_WRING_TOWEL_TEST", result)
    return result


main()
