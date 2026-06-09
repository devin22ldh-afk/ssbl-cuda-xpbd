from __future__ import annotations

import json
import math
import sys

import bpy
import numpy as np
from mathutils import Vector


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _group_weight(group: bpy.types.VertexGroup, vertex_index: int) -> float | None:
    try:
        return float(group.weight(int(vertex_index)))
    except RuntimeError:
        return None


def _mesh_group_weight(mesh: bpy.types.Mesh, group_index: int, vertex_index: int) -> float | None:
    for assignment in mesh.vertices[int(vertex_index)].groups:
        if int(assignment.group) == int(group_index):
            return float(assignment.weight)
    return None


def _make_cloth() -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=9, y_subdivisions=9, size=1.2, location=(0.0, 0.0, 1.0))
    obj = bpy.context.object
    obj.name = "SSBL_Interactive_Pin_Cloth"
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([0], 0.375, "ADD")
    settings = obj.ssbl_cloth
    settings.enabled = True
    settings.pin_vertex_group = "ssbl_pin"
    settings.use_evaluated_mesh = False
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.use_volume_pressure = False
    settings.substeps = 2
    settings.iterations = 1
    settings.frame_count = 12
    settings.preview_writeback_interval = 1
    return obj


def _run_unpinned_vertex_case(scene: bpy.types.Scene, cloth: bpy.types.Object) -> dict[str, object]:
    scene.frame_current = 1
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    vertex_index = 10
    start_world = cloth.matrix_world @ cloth.data.vertices[vertex_index].co
    handle = ssbl.solver.begin_interactive_pin(bpy.context, cloth.name, vertex_index, start_world)
    if handle is None:
        raise RuntimeError("begin_interactive_pin returned None")
    if bpy.context.view_layer.objects.active != handle or not handle.select_get():
        raise RuntimeError("Interactive pin handle was not selected")
    if handle.empty_display_type != "SPHERE":
        raise RuntimeError(f"Handle empty display type mismatch: {handle.empty_display_type}")

    slot = session.slots[cloth.name]
    pin_indices_after_begin = [int(value) for value in slot.cloth.pin_indices.tolist()]
    if vertex_index not in pin_indices_after_begin:
        raise RuntimeError(f"Interactive vertex missing from pin indices: {pin_indices_after_begin}")
    if not bool(slot.use_evaluated_mesh):
        raise RuntimeError("Interactive hook should force evaluated mesh input")

    state = slot.interactive_pin
    if state is None:
        raise RuntimeError("Slot did not record interactive pin state")
    handle_name = handle.name
    modifier = cloth.modifiers.get(state.hook_modifier_name)
    if modifier is None or modifier.type != "HOOK" or modifier.object != handle:
        raise RuntimeError("Hook modifier was not created for the handle")
    preview_mesh = cloth.data
    cloth.data = slot.original_mesh
    hook_group = cloth.vertex_groups.get(state.hook_group_name)
    hook_group_index = -1 if hook_group is None else cloth.vertex_groups.find(state.hook_group_name)
    hook_weight = None if hook_group is None else _mesh_group_weight(slot.original_mesh, hook_group_index, vertex_index)
    cloth.data = preview_mesh
    if hook_group is None or not math.isclose(hook_weight or 0.0, 1.0, abs_tol=1.0e-6):
        preview_weight = None if hook_group is None else _mesh_group_weight(cloth.data, hook_group_index, vertex_index)
        source_groups = [
            (int(assignment.group), round(float(assignment.weight), 6))
            for assignment in slot.original_mesh.vertices[vertex_index].groups
        ]
        preview_groups = [
            (int(assignment.group), round(float(assignment.weight), 6))
            for assignment in cloth.data.vertices[vertex_index].groups
        ]
        raise RuntimeError(
            "Hook vertex group does not contain only the interactive vertex: "
            f"hook_weight={hook_weight}, preview_weight={preview_weight}, "
            f"hook_group_index={hook_group_index}, "
            f"source_groups={source_groups}, preview_groups={preview_groups}"
        )

    target_world = start_world + Vector((0.12, -0.04, 0.18))
    ssbl.solver.move_interactive_pin(cloth.name, target_world)
    scene.frame_current = 2
    ssbl.solver.step_timeline_preview(bpy.context, scene)
    pin_indices_after_move = [int(value) for value in slot.cloth.pin_indices.tolist()]
    moved_pin_offset = pin_indices_after_move.index(vertex_index)
    moved_target = np.asarray(slot.pin_targets_world[moved_pin_offset], dtype=np.float64)
    if float(np.linalg.norm(moved_target - np.asarray(target_world, dtype=np.float64))) > 1.0e-4:
        raise RuntimeError(
            f"Moved pin target mismatch: {moved_target.tolist()} vs {list(target_world)}"
        )

    ssbl.solver.end_interactive_pin(cloth.name)
    handle_removed = bpy.data.objects.get(handle_name) is None
    hook_removed = cloth.modifiers.get(state.hook_modifier_name) is None
    hook_group_removed = cloth.vertex_groups.get(state.hook_group_name) is None
    ssbl.solver.reset_preview_object(cloth)
    pin_group = cloth.vertex_groups.get("ssbl_pin")
    restored_unpinned = _group_weight(pin_group, vertex_index) is None
    return {
        "pin_indices_after_begin": pin_indices_after_begin,
        "pin_indices_after_move": pin_indices_after_move,
        "handle_removed": bool(handle_removed),
        "hook_removed": bool(hook_removed),
        "hook_group_removed": bool(hook_group_removed),
        "restored_unpinned": bool(restored_unpinned),
    }


def _run_existing_weight_case(scene: bpy.types.Scene, cloth: bpy.types.Object) -> dict[str, object]:
    scene.frame_current = 1
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    vertex_index = 0
    start_world = cloth.matrix_world @ cloth.data.vertices[vertex_index].co
    handle = ssbl.solver.begin_interactive_pin(bpy.context, cloth.name, vertex_index, start_world)
    if handle is None:
        raise RuntimeError("begin_interactive_pin returned None for existing pin")
    handle_name = handle.name
    slot = session.slots[cloth.name]
    if vertex_index not in [int(value) for value in slot.cloth.pin_indices.tolist()]:
        raise RuntimeError("Existing pin disappeared during interactive pin")
    ssbl.solver.end_interactive_pin(cloth.name)
    ssbl.solver.reset_preview_object(cloth)
    pin_group = cloth.vertex_groups.get("ssbl_pin")
    restored_weight = _group_weight(pin_group, vertex_index)
    if not math.isclose(restored_weight or 0.0, 0.375, abs_tol=1.0e-6):
        raise RuntimeError(f"Existing pin weight was not restored: {restored_weight}")
    return {
        "restored_weight": round(float(restored_weight or 0.0), 6),
        "handle_removed": bpy.data.objects.get(handle_name) is None,
    }


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    _clear_scene()

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 12
    cloth = _make_cloth()
    bpy.context.view_layer.objects.active = cloth
    cloth.select_set(True)

    unpinned = _run_unpinned_vertex_case(scene, cloth)
    existing = _run_existing_weight_case(scene, cloth)
    result = {
        "unpinned": unpinned,
        "existing": existing,
        "session_names_after_reset": ssbl.solver.session_object_names(scene),
    }
    print("SSBL_INTERACTIVE_PIN_SMOKE " + json.dumps(result, sort_keys=True))
    if not (
        unpinned["handle_removed"]
        and unpinned["hook_removed"]
        and unpinned["hook_group_removed"]
        and unpinned["restored_unpinned"]
        and existing["handle_removed"]
        and existing["restored_weight"] == 0.375
        and not result["session_names_after_reset"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
