import json
import os
from pathlib import Path
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
TOOLS_DIR = str(Path(ADDONS_ROOT) / "ssbl" / "tools")
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import ssbl
from repair_wring_hook_scene import WRING_PREVIEW_STEPS, repair_current_scene
from mathutils import Vector


OBJECT_NAME = "SSBL_Wring_Towel_CurrentScene"
OUT_DIR = Path(ADDONS_ROOT) / "ssbl" / "recordings" / "wring_towel_hook_driven_preview"
FRAME_START = 1
FRAME_END = FRAME_START + WRING_PREVIEW_STEPS


def _ensure_registered():
    if not hasattr(bpy.context.scene, "ssbl_preview"):
        ssbl.register()


def _configure_scene(obj):
    scene = bpy.context.scene
    settings = scene.ssbl_preview
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.frame_set(FRAME_START)

    settings.runtime_mode = "preview"
    # Leave this disabled on purpose: the code path must auto-enable evaluated
    # input when it sees Hook modifiers, otherwise Hook-driven previews are easy
    # to misconfigure from the UI.
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = "ssbl_pin"
    settings.hardness = 0.0
    settings.hardness_initialized = True
    settings.use_volume_pressure = False
    settings.self_collision = True
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 1
    settings.max_self_collision_neighbors = 128
    settings.collision_margin = 0.015
    settings.cloth_thickness = 0.045
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.static_collider_collection = None
    settings.substeps = 16
    settings.iterations = 4
    settings.damping = 0.995
    settings.frame_count = WRING_PREVIEW_STEPS

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _configure_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.light = "STUDIO"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 24
    scene.render.image_settings.file_format = "JPEG"
    if scene.world is not None:
        scene.world.color = (0.025, 0.03, 0.035)
    camera_data = bpy.data.cameras.get("SSBL_Wring_Hook_Record_Camera")
    if camera_data is None:
        camera_data = bpy.data.cameras.new("SSBL_Wring_Hook_Record_Camera")
    camera = bpy.data.objects.get("SSBL_Wring_Hook_Record_Camera")
    if camera is None:
        camera = bpy.data.objects.new("SSBL_Wring_Hook_Record_Camera", camera_data)
        scene.collection.objects.link(camera)
    camera.location = (3.2, -5.2, 2.35)
    direction = Vector((0.0, 0.0, 0.8)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 4.6
    scene.camera = camera


def _hide_scene_text_for_recording():
    hidden = []
    for obj in bpy.context.scene.objects:
        if obj.type != "FONT":
            continue
        hidden.append((obj, bool(obj.hide_render), bool(obj.hide_viewport)))
        obj.hide_render = True
        obj.hide_viewport = True
    return hidden


def _restore_hidden_objects(hidden):
    for obj, hide_render, hide_viewport in hidden:
        if obj.name not in bpy.data.objects:
            continue
        obj.hide_render = hide_render
        obj.hide_viewport = hide_viewport


def _render_frame(frame):
    scene = bpy.context.scene
    scene.render.filepath = str(OUT_DIR / "sequence" / f"frame_{frame:04d}.jpg")
    bpy.ops.render.render(write_still=True)


def _diag_snapshot(obj):
    diag = ssbl.solver.session_diagnostics(obj)
    return {
        "frame": int(bpy.context.scene.frame_current),
        "step_ms": float(diag.step_ms),
        "hash_build_ms": float(diag.hash_build_ms),
        "candidate_count": int(diag.candidate_count),
        "resolved_contacts": int(diag.resolved_contacts),
        "min_gap": None if diag.min_gap is None else float(diag.min_gap),
        "penetration_depth": float(diag.penetration_depth),
        "ccd_clamp_count": int(diag.ccd_clamp_count),
        "recovery_passes": int(diag.recovery_passes),
        "local_retry_count": int(diag.local_retry_count),
        "finite_flag": int(diag.finite),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sequence").mkdir(parents=True, exist_ok=True)
    _ensure_registered()
    repair_result = None
    if os.environ.get("SSBL_REPAIR_WRING_HOOKS") == "1":
        repair_result = repair_current_scene()

    scene = bpy.context.scene
    obj = bpy.data.objects.get(OBJECT_NAME)
    if obj is None:
        raise RuntimeError(f"Missing object: {OBJECT_NAME}")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    _configure_scene(obj)
    _configure_render()
    hidden_text = _hide_scene_text_for_recording()

    hook_modifiers = [
        {
            "name": modifier.name,
            "show_viewport": bool(modifier.show_viewport),
            "object": modifier.object.name if getattr(modifier, "object", None) else "",
        }
        for modifier in obj.modifiers
        if modifier.type == "HOOK"
    ]

    session = ssbl.solver.start_preview(bpy.context, obj)
    slot = session.slots[obj.name]
    result = {
        "blend": bpy.data.filepath,
        "object": obj.name,
        "hook_modifiers": hook_modifiers,
        "ui_use_evaluated_mesh": False,
        "slot_use_evaluated_mesh": bool(slot.use_evaluated_mesh),
        "hardness": float(scene.ssbl_preview.hardness),
        "use_volume_pressure": bool(scene.ssbl_preview.use_volume_pressure),
        "self_collision_mode": str(scene.ssbl_preview.self_collision_mode),
        "self_collision_interval": int(scene.ssbl_preview.self_collision_interval),
        "cloth_thickness": float(scene.ssbl_preview.cloth_thickness),
        "preview_steps": WRING_PREVIEW_STEPS,
        "tethers": int(len(slot.cloth.lra_edges)),
        "frames": FRAME_END - FRAME_START + 1,
        "sequence_dir": str(OUT_DIR / "sequence"),
        "expected_video": str(OUT_DIR / "hook_driven_xpbd_preview.mp4"),
        "diagnostics": [],
    }
    if repair_result is not None:
        result["repair"] = repair_result

    try:
        _render_frame(FRAME_START)
        result["diagnostics"].append(_diag_snapshot(obj))
        for frame in range(FRAME_START + 1, FRAME_END + 1):
            finished = ssbl.solver.step_preview(bpy.context, obj.name)
            result["diagnostics"].append(_diag_snapshot(obj))
            if finished:
                result["finished_early_at"] = frame
                break
            _render_frame(frame)
    finally:
        ssbl.solver.request_stop(obj)
        _restore_hidden_objects(hidden_text)

    summary_path = OUT_DIR / "summary.json"
    result["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SSBL_HOOK_DRIVEN_WRING_PREVIEW", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
