import math
import sys
from pathlib import Path

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
SSBL_ROOT = Path(ADDONS_ROOT) / "ssbl"
TOOLS_DIR = SSBL_ROOT / "tools"
for path in (ADDONS_ROOT, str(TOOLS_DIR)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wring_towel_smoke import LENGTH, PIN_BAND, TWIST_RADIANS


OBJECT_NAME = "SSBL_Wring_Towel_CurrentScene"
PIN_GROUP = "ssbl_pin"
LEFT_HOOK_GROUP = "ssbl_hook_left"
RIGHT_HOOK_GROUP = "ssbl_hook_right"
LEFT_HOOK_NAME = "SSBL Hook Left"
RIGHT_HOOK_NAME = "SSBL Hook Right"
LEFT_EMPTY_NAME = "SSBL_Wring_Hook_Left"
RIGHT_EMPTY_NAME = "SSBL_Wring_Hook_Right"
WRING_PREVIEW_STEPS = 200
WRING_END_FRAME = WRING_PREVIEW_STEPS + 1
WRING_REFERENCE_STEPS = 80
WRING_TOTAL_TWIST_RADIANS = TWIST_RADIANS * (WRING_PREVIEW_STEPS / WRING_REFERENCE_STEPS)


def _remove_existing_hook_setup(obj):
    hook_objects = []
    for modifier in list(obj.modifiers):
        if modifier.type not in {"HOOK", "MESH_CACHE"}:
            continue
        hook_object = getattr(modifier, "object", None)
        if hook_object is not None:
            hook_objects.append(hook_object)
        obj.modifiers.remove(modifier)
    for name in (LEFT_EMPTY_NAME, RIGHT_EMPTY_NAME, "Empty", "Empty.001"):
        existing = bpy.data.objects.get(name)
        if existing is not None:
            hook_objects.append(existing)
    for hook_object in set(hook_objects):
        if hook_object.name in bpy.data.objects and hook_object.type == "EMPTY":
            bpy.data.objects.remove(hook_object, do_unlink=True)


def _replace_group(obj, name, indices):
    old = obj.vertex_groups.get(name)
    if old is not None:
        obj.vertex_groups.remove(old)
    group = obj.vertex_groups.new(name=name)
    if indices:
        group.add(indices, 1.0, "ADD")
    return group


def _band_indices(obj):
    left_limit = -LENGTH * 0.5 + PIN_BAND + 1.0e-5
    right_limit = LENGTH * 0.5 - PIN_BAND - 1.0e-5
    left = [vert.index for vert in obj.data.vertices if float(vert.co.x) <= left_limit]
    right = [vert.index for vert in obj.data.vertices if float(vert.co.x) >= right_limit]
    return left, right


def _create_hook_with_operator(obj, indices, modifier_name, empty_name):
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    index_set = set(indices)
    for vert in obj.data.vertices:
        vert.select = vert.index in index_set
    bpy.ops.object.mode_set(mode="EDIT")
    before_modifiers = {modifier.name for modifier in obj.modifiers}
    before_objects = set(bpy.data.objects.keys())
    bpy.ops.object.hook_add_newob()
    bpy.ops.object.mode_set(mode="OBJECT")

    modifier = next(
        modifier
        for modifier in obj.modifiers
        if modifier.type == "HOOK" and modifier.name not in before_modifiers
    )
    created_objects = [bpy.data.objects[name] for name in set(bpy.data.objects.keys()) - before_objects]
    empty = modifier.object
    if empty is None and created_objects:
        empty = created_objects[0]
        modifier.object = empty
    if empty is None:
        raise RuntimeError(f"Failed to create hook empty for {modifier_name}")

    modifier.name = modifier_name
    empty.name = empty_name
    empty.empty_display_type = "ARROWS"
    empty.empty_display_size = 0.35
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    return modifier, empty


def _iter_action_fcurves(action):
    if action is None:
        return
    for curve in getattr(action, "fcurves", []) or []:
        yield curve
    for layer in getattr(action, "layers", []) or []:
        for strip in getattr(layer, "strips", []) or []:
            for bag in getattr(strip, "channelbags", []) or []:
                for curve in getattr(bag, "fcurves", []) or []:
                    yield curve


def _force_linear_action(action):
    for curve in _iter_action_fcurves(action):
        for key in curve.keyframe_points:
            key.interpolation = "LINEAR"


def _animate_hook(empty, twist):
    scene = bpy.context.scene
    scene.frame_set(1)
    empty.rotation_euler = (0.0, 0.0, 0.0)
    empty.keyframe_insert(data_path="rotation_euler", frame=1)
    scene.frame_set(WRING_END_FRAME)
    empty.rotation_euler = (float(twist), 0.0, 0.0)
    empty.keyframe_insert(data_path="rotation_euler", frame=WRING_END_FRAME)
    action = empty.animation_data.action if empty.animation_data else None
    _force_linear_action(action)


def repair_current_scene():
    obj = bpy.data.objects.get(OBJECT_NAME)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"Missing mesh object: {OBJECT_NAME}")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    _remove_existing_hook_setup(obj)
    left, right = _band_indices(obj)
    if not left or not right:
        raise RuntimeError("Could not find left/right end bands for the towel mesh")
    _replace_group(obj, PIN_GROUP, sorted(left + right))
    _replace_group(obj, LEFT_HOOK_GROUP, left)
    _replace_group(obj, RIGHT_HOOK_GROUP, right)

    _left_modifier, left_empty = _create_hook_with_operator(obj, left, LEFT_HOOK_NAME, LEFT_EMPTY_NAME)
    _right_modifier, right_empty = _create_hook_with_operator(obj, right, RIGHT_HOOK_NAME, RIGHT_EMPTY_NAME)
    _animate_hook(left_empty, WRING_TOTAL_TWIST_RADIANS)
    _animate_hook(right_empty, -WRING_TOTAL_TWIST_RADIANS)

    scene = bpy.context.scene
    if hasattr(scene, "ssbl_preview"):
        settings = scene.ssbl_preview
        settings.runtime_mode = "preview"
        settings.use_evaluated_mesh = False
        settings.pin_vertex_group = PIN_GROUP
        settings.hardness = 0.0
        settings.hardness_initialized = True
        settings.use_volume_pressure = False
        settings.self_collision = True
        settings.self_collision_mode = "fast"
        settings.self_collision_interval = 1
        settings.max_self_collision_neighbors = 64
        settings.collision_margin = 0.015
        settings.cloth_thickness = 0.035
        settings.use_ground = False
        settings.use_wall = False
        settings.use_sphere = False
        settings.static_collider_collection = None
        settings.substeps = 16
        settings.iterations = 4
        settings.damping = 0.995
        settings.frame_count = WRING_PREVIEW_STEPS
    scene.frame_start = 1
    scene.frame_end = WRING_END_FRAME
    scene.frame_set(1)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return {
        "object": obj.name,
        "left_hook_vertices": len(left),
        "right_hook_vertices": len(right),
        "pin_vertices": len(left) + len(right),
        "twist_degrees": math.degrees(WRING_TOTAL_TWIST_RADIANS),
        "preview_steps": WRING_PREVIEW_STEPS,
    }


def main():
    result = repair_current_scene()
    print("SSBL_REPAIRED_WRING_HOOK_SCENE", result)


if __name__ == "__main__":
    main()
