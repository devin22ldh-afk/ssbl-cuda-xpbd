from __future__ import annotations

import importlib
import json
import sys
from typing import Any

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Quaternion, Vector


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
TEST_SCENE_NAME = "SSBL_MCP_ALT_TEST_SCENE"
TEST_CLOTH_NAME = "SSBL_MCP_ALT_TEST_Cloth"
TEST_MESH_NAME = "SSBL_MCP_ALT_TEST_Mesh"
TEST_TARGET_VERTEX = 10
TEST_STATE_KEY = "_ssbl_mcp_alt_modal_validation"


def _json_print(marker: str, payload: dict[str, Any]) -> None:
    print(f"{marker} " + json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _ensure_addons_root() -> None:
    if ADDONS_ROOT not in sys.path:
        sys.path.insert(0, ADDONS_ROOT)


def _reload_ssbl():
    _ensure_addons_root()
    try:
        bpy.ops.preferences.addon_enable(module="ssbl")
    except Exception:
        pass

    import ssbl

    try:
        ssbl.unregister()
    except Exception:
        pass

    for module_name in (
        "ssbl.operators",
        "ssbl.session_manager",
        "ssbl.solver",
        "ssbl.xpbd_core",
        "ssbl",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)

    import ssbl as reloaded

    reloaded.register()
    return reloaded


def _first_view3d_context():
    window = bpy.context.window
    if window is None or window.screen is None:
        raise RuntimeError("no active Blender window")
    for area in window.screen.areas:
        if area.type != "VIEW_3D":
            continue
        region = next((item for item in area.regions if item.type == "WINDOW"), None)
        space = next((item for item in area.spaces if item.type == "VIEW_3D"), None)
        rv3d = getattr(space, "region_3d", None) if space is not None else None
        if region is not None and rv3d is not None:
            return window, area, region, space, rv3d
    raise RuntimeError("no VIEW_3D area in current Blender window")


def _remove_scene(name: str) -> None:
    scene = bpy.data.scenes.get(name)
    if scene is None:
        return
    for obj in list(scene.objects):
        mesh_data = obj.data if getattr(obj, "type", None) == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh_data is not None and mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)
    bpy.data.scenes.remove(scene)


def _make_test_cloth(scene: bpy.types.Scene) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(TEST_MESH_NAME)
    size = 1.2
    steps = 4
    verts = [
        ((x / (steps - 1) - 0.5) * size, (y / (steps - 1) - 0.5) * size, 0.0)
        for y in range(steps)
        for x in range(steps)
    ]
    faces = []
    for y in range(steps - 1):
        for x in range(steps - 1):
            a = y * steps + x
            faces.append((a, a + 1, a + 1 + steps, a + steps))
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    cloth = bpy.data.objects.new(TEST_CLOTH_NAME, mesh)
    scene.collection.objects.link(cloth)

    group = cloth.vertex_groups.new(name="ssbl_pin")
    group.add([0], 0.375, "ADD")

    settings = cloth.ssbl_cloth
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
    return cloth


def _mesh_group_weight(mesh: bpy.types.Mesh, group_index: int, vertex_index: int) -> float | None:
    for assignment in mesh.vertices[int(vertex_index)].groups:
        if int(assignment.group) == int(group_index):
            return float(assignment.weight)
    return None


def _group_weight(obj: bpy.types.Object, group_name: str, vertex_index: int) -> float | None:
    group = obj.vertex_groups.get(group_name)
    if group is None:
        return None
    try:
        return float(group.weight(int(vertex_index)))
    except RuntimeError:
        return None


def _scene_state() -> dict[str, str]:
    window = bpy.context.window
    scene = window.scene if window is not None else bpy.context.scene
    active = bpy.context.view_layer.objects.active
    return {
        "scene_name": scene.name if scene is not None else "",
        "active_name": active.name if active is not None else "",
        "selection_names": json.dumps([obj.name for obj in bpy.context.selected_objects]),
    }


def setup() -> None:
    ssbl = _reload_ssbl()
    original = _scene_state()
    _remove_scene(TEST_SCENE_NAME)

    scene = bpy.data.scenes.new(TEST_SCENE_NAME)
    bpy.context.window.scene = scene
    scene.frame_start = 1
    scene.frame_end = 12
    scene.frame_current = 2

    cloth = _make_test_cloth(scene)
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth

    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None or cloth.name not in session.slots:
        raise RuntimeError("timeline preview session did not include test cloth")

    window, area, region, space, rv3d = _first_view3d_context()
    rv3d.view_perspective = "ORTHO"
    rv3d.view_location = Vector((0.0, 0.0, 0.0))
    rv3d.view_distance = 3.0
    rv3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    space.region_3d.view_location = Vector((0.0, 0.0, 0.0))

    target_world = cloth.matrix_world @ cloth.data.vertices[TEST_TARGET_VERTEX].co
    screen_xy = view3d_utils.location_3d_to_region_2d(region, rv3d, target_world)
    if screen_xy is None:
        raise RuntimeError("target vertex could not be projected")

    mouse_x = int(region.x + screen_xy.x)
    mouse_y = int(region.y + screen_xy.y)

    # Register the real modal operator through Blender, not by instantiating it.
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
        bpy.ops.ssbl.interactive_pin_monitor("INVOKE_DEFAULT")
    ssbl.operators._ensure_interactive_pin_monitor(bpy.context)

    state = {
        **original,
        "target_vertex": str(TEST_TARGET_VERTEX),
        "mouse_x": str(mouse_x),
        "mouse_y": str(mouse_y),
        "region_x": str(region.x),
        "region_y": str(region.y),
        "region_width": str(region.width),
        "region_height": str(region.height),
    }
    scene[TEST_STATE_KEY] = json.dumps(state)

    _json_print(
        "SSBL_MCP_ALT_SETUP_OK",
        {
            "active_monitor": bool(ssbl.operators.SSBL_OT_interactive_pin_monitor._active),
            "cloth": cloth.name,
            "mouse_x": mouse_x,
            "mouse_y": mouse_y,
            "region": [region.x, region.y, region.width, region.height],
            "scene": scene.name,
            "target_vertex": TEST_TARGET_VERTEX,
        },
    )


def probe() -> None:
    _ensure_addons_root()
    import ssbl

    scene = bpy.data.scenes.get(TEST_SCENE_NAME)
    cloth = bpy.data.objects.get(TEST_CLOTH_NAME)
    payload: dict[str, Any] = {
        "scene_exists": scene is not None,
        "cloth_exists": cloth is not None,
        "active_monitor": bool(ssbl.operators.SSBL_OT_interactive_pin_monitor._active),
        "active_object": bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else "",
        "selected": [obj.name for obj in bpy.context.selected_objects],
    }
    if scene is None or cloth is None:
        _json_print("SSBL_MCP_ALT_PROBE", payload)
        return

    session = ssbl.session_manager._SCENE_SESSIONS.get(scene.name)
    slot = session.slots.get(TEST_CLOTH_NAME) if session is not None else None
    interactive = slot.interactive_pin if slot is not None else None
    payload.update(
        {
            "session_exists": session is not None and not getattr(session, "closed", False),
            "session_objects": ssbl.solver.session_object_names(scene),
            "interactive_exists": interactive is not None,
            "frame": int(scene.frame_current),
            "target_weight": _group_weight(cloth, "ssbl_pin", TEST_TARGET_VERTEX),
        }
    )
    if slot is not None:
        payload["runtime_pin_indices"] = [int(value) for value in slot.cloth.pin_indices.tolist()]
        payload["use_evaluated_mesh"] = bool(slot.use_evaluated_mesh)
    if interactive is not None:
        handle = bpy.data.objects.get(interactive.empty_name)
        hook_group = cloth.vertex_groups.get(interactive.hook_group_name)
        payload.update(
            {
                "empty": interactive.empty_name,
                "empty_exists": handle is not None,
                "empty_type": handle.empty_display_type if handle is not None else "",
                "empty_selected": bool(handle.select_get()) if handle is not None else False,
                "empty_location": [float(v) for v in handle.location] if handle is not None else [],
                "hook_modifier": interactive.hook_modifier_name,
                "hook_modifier_exists": cloth.modifiers.get(interactive.hook_modifier_name) is not None,
                "hook_group": interactive.hook_group_name,
                "hook_group_exists": hook_group is not None,
                "hook_group_weight": _group_weight(cloth, interactive.hook_group_name, TEST_TARGET_VERTEX),
                "previous_weight": interactive.previous_weight,
            }
        )
    _json_print("SSBL_MCP_ALT_PROBE", payload)


def cleanup() -> None:
    _ensure_addons_root()
    import ssbl

    scene = bpy.data.scenes.get(TEST_SCENE_NAME)
    raw_state = scene.get(TEST_STATE_KEY, "{}") if scene is not None else "{}"
    try:
        state = json.loads(raw_state)
    except Exception:
        state = {}

    if scene is not None:
        try:
            ssbl.solver.cleanup_interactive_pins(scene)
            for obj in list(scene.objects):
                if ssbl.solver.has_session(obj):
                    ssbl.solver.reset_preview_object(obj)
        except Exception:
            pass

    original_scene = bpy.data.scenes.get(state.get("scene_name", ""))
    if bpy.context.window is not None and original_scene is not None:
        bpy.context.window.scene = original_scene
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for name in json.loads(state.get("selection_names", "[]")):
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.select_set(True)
        active = bpy.data.objects.get(state.get("active_name", ""))
        if active is not None:
            bpy.context.view_layer.objects.active = active
    except Exception:
        pass
    _remove_scene(TEST_SCENE_NAME)
    _json_print("SSBL_MCP_ALT_CLEANUP_OK", {"scene_removed": bpy.data.scenes.get(TEST_SCENE_NAME) is None})
