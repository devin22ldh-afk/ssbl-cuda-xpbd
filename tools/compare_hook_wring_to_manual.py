import json
import sys
from pathlib import Path

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
SSBL_ROOT = Path(ADDONS_ROOT) / "ssbl"
TOOLS_DIR = SSBL_ROOT / "tools"
for path in (ADDONS_ROOT, str(TOOLS_DIR)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ssbl
from repair_wring_hook_scene import OBJECT_NAME, repair_current_scene
from wring_towel_smoke import WRING_FRAME_COUNT, _step_wring_frame


OUT_PATH = SSBL_ROOT / "recordings" / "wring_towel_hook_driven_preview" / "hook_vs_manual_compare.json"
SAMPLES = (0, 20, 40, 60, 80)


def _ensure_registered():
    if not hasattr(bpy.context.scene, "ssbl_preview"):
        ssbl.register()


def _configure_scene(obj):
    scene = bpy.context.scene
    settings = scene.ssbl_preview
    scene.frame_start = 1
    scene.frame_end = WRING_FRAME_COUNT + 1
    scene.frame_set(1)
    settings.runtime_mode = "preview"
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = "ssbl_pin"
    settings.hardness = 0.0
    settings.hardness_initialized = True
    settings.use_volume_pressure = False
    settings.self_collision = False
    settings.self_collision_mode = "fast"
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.static_collider_collection = None
    settings.substeps = 16
    settings.iterations = 4
    settings.damping = 0.995
    settings.frame_count = WRING_FRAME_COUNT + 1
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _run_hook(obj):
    _configure_scene(obj)
    session = ssbl.solver.start_preview(bpy.context, obj)
    slot = session.slots[obj.name]
    frames = {0: np.array(slot.current_positions_world, dtype=np.float32, copy=True)}
    try:
        for step in range(1, WRING_FRAME_COUNT + 1):
            finished = ssbl.solver.step_preview(bpy.context, obj.name)
            slot = session.slots[obj.name]
            if step in SAMPLES:
                frames[step] = np.array(slot.current_positions_world, dtype=np.float32, copy=True)
            if finished:
                break
    finally:
        ssbl.solver.request_stop(obj)
    return frames


def _run_manual(obj):
    hidden_hooks = []
    for modifier in obj.modifiers:
        if modifier.type != "HOOK":
            continue
        hidden_hooks.append((modifier, bool(modifier.show_viewport), bool(modifier.show_render)))
        modifier.show_viewport = False
        modifier.show_render = False
    try:
        _configure_scene(obj)
        session = ssbl.solver.start_preview(bpy.context, obj)
        slot = session.slots[obj.name]
        frames = {0: np.array(slot.current_positions_world, dtype=np.float32, copy=True)}
        try:
            for step in range(1, WRING_FRAME_COUNT + 1):
                positions = _step_wring_frame(session, obj, step, WRING_FRAME_COUNT)
                if step in SAMPLES:
                    frames[step] = np.array(positions, dtype=np.float32, copy=True)
        finally:
            ssbl.solver.request_stop(obj)
    finally:
        for modifier, show_viewport, show_render in hidden_hooks:
            if obj.modifiers.get(modifier.name) is not None:
                modifier.show_viewport = show_viewport
                modifier.show_render = show_render
    return frames


def main():
    _ensure_registered()
    repair_current_scene()
    obj = bpy.data.objects.get(OBJECT_NAME)
    if obj is None:
        raise RuntimeError(f"Missing object: {OBJECT_NAME}")
    hook_frames = _run_hook(obj)
    manual_frames = _run_manual(obj)
    comparisons = {}
    for frame in SAMPLES:
        hook = hook_frames[frame]
        manual = manual_frames[frame]
        delta = hook - manual
        comparisons[str(frame)] = {
            "max_abs_delta": float(np.max(np.abs(delta))),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "rms_delta": float(np.sqrt(np.mean(delta * delta))),
            "finite": bool(np.isfinite(hook).all() and np.isfinite(manual).all()),
        }
    result = {
        "blend": bpy.data.filepath,
        "object": obj.name,
        "samples": list(SAMPLES),
        "comparisons": comparisons,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SSBL_HOOK_VS_MANUAL_WRING_COMPARE", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
