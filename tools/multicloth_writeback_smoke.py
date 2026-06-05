from __future__ import annotations

import json
import math
import sys

import bpy
import numpy as np


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


def _finite_slot_positions(session) -> bool:
    for slot in session.slots.values():
        positions = np.asarray(slot.current_positions_world, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or not bool(np.isfinite(positions).all()):
            return False
    return True


def _finite_object(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _configure_cloth(obj: bpy.types.Object) -> None:
    settings = obj.ssbl_cloth
    settings.enabled = True
    settings.pin_vertex_group = ""
    settings.use_evaluated_mesh = True
    settings.preview_writeback_interval = 0
    settings.preview_target_fps = 60.0
    settings.dt = 1.0 / 60.0
    settings.substeps = 2
    settings.iterations = 1
    settings.frame_count = 8
    settings.damping = 0.99
    settings.gravity = (0.0, 0.0, 0.0)
    settings.hardness = 0.45
    settings.self_collision = False
    settings.use_volume_pressure = False
    settings.collision_margin = 0.004
    settings.cloth_thickness = 0.03
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.sphere_object = None
    settings.static_collider_collection = None


def _make_cloth(name: str, z: float) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=1.2, location=(0.0, 0.0, z))
    obj = bpy.context.object
    obj.name = name
    _configure_cloth(obj)
    return obj


def _step_and_diag(obj: bpy.types.Object):
    ssbl.solver.step_preview(bpy.context, obj.name)
    return ssbl.solver.session_diagnostics(obj)


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        _clear_scene()
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = 5
        scene.frame_current = 1

        first = _make_cloth("SSBL_Writeback_Multi_A", 0.0)
        second = _make_cloth("SSBL_Writeback_Multi_B", 0.02)
        before_first = _snapshot(first)
        before_second = _snapshot(second)

        bpy.ops.object.select_all(action="DESELECT")
        first.select_set(True)
        second.select_set(True)
        bpy.context.view_layer.objects.active = first

        session = ssbl.solver.start_preview(bpy.context, first)
        first_diag = _step_and_diag(first)
        first_writeback = bool(first_diag.writeback_performed)

        session.adaptive_writeback_interval = 8
        skipped_diag = _step_and_diag(first)
        skipped_writeback = bool(skipped_diag.writeback_performed)
        skipped_writeback_ms = float(skipped_diag.writeback_ms)
        skipped_download_ms = float(skipped_diag.download_ms)
        skipped_dynamic_triangles = int(skipped_diag.dynamic_triangle_count)

        session.adaptive_writeback_interval = 8
        first.ssbl_cloth.gravity = (0.0, 0.0, -18.0)
        tuning_diag = _step_and_diag(first)
        tuning_writeback = bool(tuning_diag.writeback_performed)

        session.adaptive_writeback_interval = 8
        end_diag = _step_and_diag(first)
        end_writeback = bool(end_diag.writeback_performed)

        finite = (
            _finite_slot_positions(session)
            and _finite_object(first)
            and _finite_object(second)
            and bool(end_diag.finite)
        )
        stopped = ssbl.solver.request_stop(first)
        result = {
            "slots": len(session.slots),
            "cross_mode": str(session.cross_cloth_mode),
            "first_writeback": first_writeback,
            "skipped_writeback": skipped_writeback,
            "skipped_writeback_ms": skipped_writeback_ms,
            "skipped_download_ms": skipped_download_ms,
            "skipped_dynamic_triangle_count": skipped_dynamic_triangles,
            "tuning_writeback": tuning_writeback,
            "end_writeback": end_writeback,
            "finite": bool(finite),
            "stopped": bool(stopped),
            "restore_delta_first": _max_source_delta(first, before_first),
            "restore_delta_second": _max_source_delta(second, before_second),
        }
        print("SSBL_MULTICLOTH_WRITEBACK_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            result["slots"] == 2
            and result["cross_mode"] == "all_selected"
            and result["first_writeback"]
            and not result["skipped_writeback"]
            and result["skipped_writeback_ms"] == 0.0
            and result["skipped_download_ms"] > 0.0
            and result["skipped_dynamic_triangle_count"] > 0
            and result["tuning_writeback"]
            and result["end_writeback"]
            and result["finite"]
            and result["stopped"]
            and result["restore_delta_first"] == 0.0
            and result["restore_delta_second"] == 0.0
        ):
            raise RuntimeError(f"Multi-cloth writeback smoke failed: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
