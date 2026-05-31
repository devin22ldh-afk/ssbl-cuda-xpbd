import json
import math
import os
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _find_monkey():
    active = bpy.context.active_object
    if active is not None and active.type == "MESH":
        return active
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and "suzanne" in obj.name.lower():
            return obj
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh object found in the scene")
    return meshes[0]


def _ensure_pin_group(obj, settings):
    group_name = str(getattr(settings, "pin_vertex_group", "") or "").strip() or "ssbl_pin"
    group = obj.vertex_groups.get(group_name)
    if group is not None:
        return group_name, False

    z_values = [vert.co.z for vert in obj.data.vertices]
    threshold = max(z_values) - (max(z_values) - min(z_values)) * 0.18
    pin_indices = [vert.index for vert in obj.data.vertices if vert.co.z >= threshold]
    if not pin_indices:
        pin_indices = [max(obj.data.vertices, key=lambda vert: vert.co.z).index]
    group = obj.vertex_groups.new(name=group_name)
    group.add(pin_indices, 1.0, "ADD")
    settings.pin_vertex_group = group_name
    return group_name, True


def _setup_render(scene, obj):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 12
    if scene.world is not None:
        scene.world.color = (0.025, 0.03, 0.035)

    if scene.camera is None:
        camera_data = bpy.data.cameras.new("SSBL_Record_Camera")
        camera = bpy.data.objects.new("SSBL_Record_Camera", camera_data)
        bpy.context.collection.objects.link(camera)
        center = obj.location
        camera.location = (center.x, center.y - 6.0, center.z + 1.5)
        camera.rotation_euler = (math.radians(75.0), 0.0, 0.0)
        camera_data.lens = 45
        scene.camera = camera

    if not any(light.type == "LIGHT" for light in scene.objects):
        light_data = bpy.data.lights.new("SSBL_Record_Key", "AREA")
        light = bpy.data.objects.new("SSBL_Record_Key", light_data)
        bpy.context.collection.objects.link(light)
        light.location = (obj.location.x, obj.location.y - 3.5, obj.location.z + 4.0)
        light_data.energy = 500
        light_data.size = 4


def _bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _apply_env_overrides(settings):
    if hasattr(settings, "frame_count"):
        settings.frame_count = int(_float_env("SSBL_RECORD_FRAME_COUNT", float(settings.frame_count)))
    if hasattr(settings, "hardness") and "SSBL_RECORD_HARDNESS" in os.environ:
        settings.hardness = max(0.0, min(1.0, _float_env("SSBL_RECORD_HARDNESS", float(settings.hardness))))
        if hasattr(settings, "hardness_initialized"):
            settings.hardness_initialized = True
    if hasattr(settings, "substeps"):
        settings.substeps = int(_float_env("SSBL_RECORD_SUBSTEPS", float(settings.substeps)))
    if hasattr(settings, "iterations"):
        settings.iterations = int(_float_env("SSBL_RECORD_ITERATIONS", float(settings.iterations)))
    if hasattr(settings, "use_ground"):
        settings.use_ground = _bool_env("SSBL_RECORD_USE_GROUND", False)
    if hasattr(settings, "ground_height"):
        settings.ground_height = _float_env("SSBL_RECORD_GROUND_HEIGHT", float(settings.ground_height))
    if hasattr(settings, "collision_margin"):
        settings.collision_margin = _float_env("SSBL_RECORD_COLLISION_MARGIN", float(settings.collision_margin))
    if hasattr(settings, "stretch_compliance") and "SSBL_RECORD_STRETCH_COMPLIANCE" in os.environ:
        settings.stretch_compliance = _float_env("SSBL_RECORD_STRETCH_COMPLIANCE", float(settings.stretch_compliance))
    if hasattr(settings, "bend_compliance") and "SSBL_RECORD_BEND_COMPLIANCE" in os.environ:
        settings.bend_compliance = _float_env("SSBL_RECORD_BEND_COMPLIANCE", float(settings.bend_compliance))
    if hasattr(settings, "self_collision_mode"):
        mode = os.environ.get("SSBL_RECORD_SELF_COLLISION_MODE", str(settings.self_collision_mode))
        if mode == "quality":
            mode = "fast"
        settings.self_collision_mode = mode
        if mode == "fast":
            settings.self_collision_interval = int(_float_env("SSBL_RECORD_SELF_COLLISION_INTERVAL", 2))
            settings.max_self_collision_neighbors = int(_float_env("SSBL_RECORD_MAX_SELF_NEIGHBORS", 32))
    if hasattr(settings, "use_volume_pressure"):
        settings.use_volume_pressure = _bool_env("SSBL_RECORD_VOLUME_PRESSURE", bool(settings.use_volume_pressure))
    if hasattr(settings, "volume_compliance"):
        settings.volume_compliance = _float_env("SSBL_RECORD_VOLUME_COMPLIANCE", float(settings.volume_compliance))
    if hasattr(settings, "pressure_strength"):
        settings.pressure_strength = _float_env("SSBL_RECORD_PRESSURE_STRENGTH", float(settings.pressure_strength))
    if hasattr(settings, "volume_target_scale"):
        settings.volume_target_scale = _float_env("SSBL_RECORD_VOLUME_TARGET_SCALE", float(settings.volume_target_scale))


def _snapshot(obj, label, diagnostics):
    bbox = [tuple(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    coords = [component for vert in obj.data.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    z_coords = [vert.co.z for vert in obj.data.vertices]
    session_diag = ssbl.solver.session_diagnostics(obj)
    diagnostics.append(
        {
            "label": label,
            "frame": int(bpy.context.scene.frame_current),
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
            "visible_get": bool(obj.visible_get(view_layer=bpy.context.view_layer)),
            "mesh": obj.data.name,
            "vertex_count": len(obj.data.vertices),
            "all_vertex_coords_finite": all(math.isfinite(float(value)) for value in coords),
            "bbox_min_z": min(point[2] for point in bbox),
            "bbox_max_z": max(point[2] for point in bbox),
            "data_min_z": min(z_coords) if z_coords else None,
            "data_max_z": max(z_coords) if z_coords else None,
            "step_ms": float(session_diag.step_ms),
            "hash_build_ms": float(session_diag.hash_build_ms),
            "candidate_count": int(session_diag.candidate_count),
            "resolved_contacts": int(session_diag.resolved_contacts),
            "min_gap": None if session_diag.min_gap is None else float(session_diag.min_gap),
            "penetration_depth": float(session_diag.penetration_depth),
            "ccd_clamp_count": int(session_diag.ccd_clamp_count),
            "recovery_passes": int(session_diag.recovery_passes),
            "local_retry_count": int(session_diag.local_retry_count),
            "finite_flag": int(session_diag.finite),
        }
    )


def _render_frame(scene, obj, frames_dir, index, label, diagnostics):
    _snapshot(obj, label, diagnostics)
    scene.render.filepath = str(frames_dir / f"{index:03d}_{label}.png")
    try:
        bpy.ops.render.opengl(write_still=True, view_context=False)
    except RuntimeError:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        bpy.ops.render.render(write_still=True)


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    output_override = os.environ.get("SSBL_RECORD_OUTPUT_DIR")
    output_dir = Path(output_override) if output_override else Path(tempfile.gettempdir()) / "ssbl_current_scene_monkey_recording"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("*.*"):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            path.unlink()

    scene = bpy.context.scene
    scene.frame_set(1)
    obj = _find_monkey()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    source_mesh = obj.data
    source_coords_before = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    settings = scene.ssbl_preview
    _apply_env_overrides(settings)
    pin_group, created_pin = _ensure_pin_group(obj, settings)
    _setup_render(scene, obj)
    scene.render.image_settings.file_format = "PNG"

    diagnostics = []
    _render_frame(scene, obj, frames_dir, 0, "before", diagnostics)
    session = ssbl.solver.start_preview(bpy.context, obj)
    _render_frame(scene, obj, frames_dir, 1, "start", diagnostics)

    frame_count = min(max(int(getattr(settings, "frame_count", 24)), 1), 60)
    for i in range(1, frame_count + 1):
        ssbl.solver.step_preview(bpy.context, obj.name)
        if i <= 12 or i == frame_count or i % 5 == 0:
            _render_frame(scene, obj, frames_dir, i + 1, f"step{i}", diagnostics)

    ssbl.solver.request_stop(obj)
    _render_frame(scene, obj, frames_dir, 1000, "stopped", diagnostics)
    source_coords_after = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    original_mesh_max_abs_delta = max(
        (abs(float(after) - float(before)) for before, after in zip(source_coords_before, source_coords_after)),
        default=0.0,
    )

    summary = {
        "blend_file": bpy.data.filepath,
        "object": obj.name,
        "source_mesh": source_mesh.name,
        "restored_mesh": obj.data.name,
        "original_mesh_max_abs_delta": original_mesh_max_abs_delta,
        "pin_group": pin_group,
        "created_pin_group_for_test": created_pin,
        "output_dir": str(output_dir),
        "frames_dir": str(frames_dir),
        "diagnostics": diagnostics,
        "frame_paths": [str(path) for path in sorted(frames_dir.glob("*.*")) if path.suffix.lower() in {".png", ".jpg", ".jpeg"}],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SSBL_SCENE_MONKEY_SUMMARY", str(summary_path))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
