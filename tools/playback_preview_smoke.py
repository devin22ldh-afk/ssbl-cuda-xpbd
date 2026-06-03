from __future__ import annotations

import json
import math
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _snapshot(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in obj.data.vertices]


def _max_source_delta(obj: bpy.types.Object, before: list[tuple[float, float, float]]) -> float:
    if len(obj.data.vertices) != len(before):
        return float("inf")
    return max(
        (
            max(
                abs(float(vertex.co.x) - old[0]),
                abs(float(vertex.co.y) - old[1]),
                abs(float(vertex.co.z) - old[2]),
            )
            for vertex, old in zip(obj.data.vertices, before)
        ),
        default=0.0,
    )


def _finite_object(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _make_cloth(name: str, location: tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=13, y_subdivisions=13, size=1.2, location=location)
    obj = bpy.context.object
    obj.name = name
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([vertex.index for vertex in obj.data.vertices if vertex.co.y > 0.48], 1.0, "ADD")
    settings = obj.ssbl_cloth
    settings.enabled = True
    settings.pin_vertex_group = "ssbl_pin"
    settings.use_evaluated_mesh = True
    settings.self_collision_mode = "off"
    settings.use_volume_pressure = False
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.substeps = 2
    settings.iterations = 1
    settings.frame_count = 24
    settings.preview_writeback_interval = 1
    return obj


def _run_single(scene: bpy.types.Scene) -> dict[str, object]:
    obj = _make_cloth("SSBL_Playback_Single", (0.0, 0.0, 1.0))
    bpy.context.view_layer.objects.active = obj
    original_mesh = obj.data
    before = _snapshot(obj)
    scene.frame_start = 1
    scene.frame_end = 24
    scene.frame_current = 1
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    for frame in (2, 3, 4):
        scene.frame_current = frame
        ssbl.solver.step_timeline_preview(bpy.context, scene)
    ssbl.solver.pause_timeline_preview(scene)
    paused_keeps_preview_mesh = obj.data != original_mesh and ssbl.solver.has_session(obj)
    paused_status = ssbl.solver.session_status(obj)
    finite = _finite_object(obj) and bool(ssbl.solver.session_diagnostics(obj).finite)
    ssbl.solver.reset_preview_object(obj)
    return {
        "slots": len(session.slots) if session else 0,
        "paused_keeps_preview_mesh": bool(paused_keeps_preview_mesh),
        "paused_status": paused_status,
        "restored_mesh": obj.data == original_mesh,
        "restore_delta": _max_source_delta(obj, before),
        "finite": bool(finite),
    }


def _run_multi(scene: bpy.types.Scene) -> dict[str, object]:
    first = _make_cloth("SSBL_Playback_Multi_A", (-0.7, 0.0, 1.0))
    second = _make_cloth("SSBL_Playback_Multi_B", (0.7, 0.0, 1.0))
    before_first = _snapshot(first)
    before_second = _snapshot(second)
    scene.frame_current = 1
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    scene.frame_current = 2
    ssbl.solver.step_timeline_preview(bpy.context, scene)
    diagnostics = ssbl.solver.session_diagnostics(first)
    finite = _finite_object(first) and _finite_object(second) and bool(diagnostics.finite)
    cross_mode = str(session.cross_cloth_mode) if session else ""
    ssbl.solver.reset_preview_object(first)
    return {
        "slots": len(session.slots) if session else 0,
        "cross_mode": cross_mode,
        "dynamic_upload_ms": float(diagnostics.dynamic_upload_ms),
        "dynamic_collision_ms": float(diagnostics.dynamic_collision_ms),
        "finite": bool(finite),
        "restore_delta_first": _max_source_delta(first, before_first),
        "restore_delta_second": _max_source_delta(second, before_second),
    }


def _run_manual_multi(scene: bpy.types.Scene) -> dict[str, object]:
    first = _make_cloth("SSBL_Manual_Multi_A", (-0.7, 0.0, 1.0))
    second = _make_cloth("SSBL_Manual_Multi_B", (0.7, 0.0, 1.0))
    before_first = _snapshot(first)
    before_second = _snapshot(second)
    bpy.ops.object.select_all(action="DESELECT")
    first.select_set(True)
    second.select_set(True)
    bpy.context.view_layer.objects.active = first
    scene.frame_current = 1
    session = ssbl.solver.start_preview(bpy.context, first)
    ssbl.solver.step_preview(bpy.context, first.name)
    diagnostics = ssbl.solver.session_diagnostics(first)
    finite = _finite_object(first) and _finite_object(second) and bool(diagnostics.finite)
    cross_mode = str(session.cross_cloth_mode) if session else ""
    ssbl.solver.reset_preview_object(first)
    return {
        "slots": len(session.slots) if session else 0,
        "cross_mode": cross_mode,
        "dynamic_upload_ms": float(diagnostics.dynamic_upload_ms),
        "dynamic_collision_ms": float(diagnostics.dynamic_collision_ms),
        "finite": bool(finite),
        "restore_delta_first": _max_source_delta(first, before_first),
        "restore_delta_second": _max_source_delta(second, before_second),
    }


def _run_jump_restart(scene: bpy.types.Scene) -> dict[str, object]:
    obj = _make_cloth("SSBL_Playback_Jump", (0.0, 0.0, 1.0))
    before = _snapshot(obj)
    scene.frame_current = 1
    first = ssbl.solver.start_timeline_preview(bpy.context, scene)
    scene.frame_current = 2
    ssbl.solver.step_timeline_preview(bpy.context, scene)
    scene.frame_current = 10
    ssbl.solver.step_timeline_preview(bpy.context, scene)
    restarted = ssbl.solver.has_session(obj)
    second = ssbl.solver.start_timeline_preview(bpy.context, scene)
    scene.frame_current = 11
    ssbl.solver.step_timeline_preview(bpy.context, scene)
    finite = _finite_object(obj) and bool(ssbl.solver.session_diagnostics(obj).finite)
    ssbl.solver.reset_preview_object(obj)
    return {
        "first_slots": len(first.slots) if first else 0,
        "second_slots": len(second.slots) if second else 0,
        "restarted": bool(restarted),
        "finite": bool(finite),
        "restore_delta": _max_source_delta(obj, before),
    }


def _run_endpoint_reset(scene: bpy.types.Scene) -> dict[str, object]:
    obj = _make_cloth("SSBL_Playback_Endpoint", (0.0, 0.0, 1.0))
    bpy.context.view_layer.objects.active = obj
    original_mesh = obj.data
    before = _snapshot(obj)
    scene.frame_start = 1
    scene.frame_end = 8
    scene.frame_current = 1
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    scene.frame_current = 2
    ssbl.solver.step_timeline_preview(bpy.context, scene)
    active_before_endpoint = ssbl.solver.has_session(obj) and obj.data != original_mesh
    scene.frame_current = scene.frame_end
    ssbl.operators._ssbl_frame_change_post(scene)
    return {
        "slots": len(session.slots) if session else 0,
        "active_before_endpoint": bool(active_before_endpoint),
        "reset_at_endpoint": not ssbl.solver.has_session(obj),
        "restored_mesh": obj.data == original_mesh,
        "restore_delta": _max_source_delta(obj, before),
    }


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        _clear_scene()
        scene = bpy.context.scene
        single = _run_single(scene)
        ssbl.solver.cleanup_all_sessions()
        _clear_scene()
        multi = _run_multi(scene)
        ssbl.solver.cleanup_all_sessions()
        _clear_scene()
        manual_multi = _run_manual_multi(scene)
        ssbl.solver.cleanup_all_sessions()
        _clear_scene()
        jump = _run_jump_restart(scene)
        ssbl.solver.cleanup_all_sessions()
        _clear_scene()
        endpoint = _run_endpoint_reset(scene)
        result = {
            "single": single,
            "multi": multi,
            "manual_multi": manual_multi,
            "jump": jump,
            "endpoint": endpoint,
        }
        print("SSBL_PLAYBACK_PREVIEW_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            single["slots"] == 1
            and single["paused_keeps_preview_mesh"]
            and single["paused_status"] == ssbl.solver.STATUS_PREVIEW_PAUSED
            and single["restored_mesh"]
            and single["restore_delta"] == 0.0
            and single["finite"]
            and multi["slots"] == 2
            and multi["cross_mode"] == "all_selected"
            and multi["dynamic_upload_ms"] > 0.0
            and multi["finite"]
            and multi["restore_delta_first"] == 0.0
            and multi["restore_delta_second"] == 0.0
            and manual_multi["slots"] == 2
            and manual_multi["cross_mode"] == "all_selected"
            and manual_multi["dynamic_upload_ms"] > 0.0
            and manual_multi["finite"]
            and manual_multi["restore_delta_first"] == 0.0
            and manual_multi["restore_delta_second"] == 0.0
            and jump["first_slots"] == 1
            and jump["second_slots"] == 1
            and jump["restarted"]
            and jump["finite"]
            and jump["restore_delta"] == 0.0
            and endpoint["slots"] == 1
            and endpoint["active_before_endpoint"]
            and endpoint["reset_at_endpoint"]
            and endpoint["restored_mesh"]
            and endpoint["restore_delta"] == 0.0
        ):
            raise RuntimeError(f"Playback preview smoke failed: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
