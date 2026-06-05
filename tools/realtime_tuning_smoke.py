from __future__ import annotations

import json
import math
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from ssbl import session_manager


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


def _make_cloth() -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=15, y_subdivisions=15, size=1.2, location=(0.0, 0.0, 1.0))
    obj = bpy.context.object
    obj.name = "SSBL_Realtime_Tuning_Cloth"
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([vertex.index for vertex in obj.data.vertices if vertex.co.y > 0.48], 1.0, "ADD")
    return obj


def _configure() -> None:
    settings = bpy.context.scene.ssbl_preview
    settings.pin_vertex_group = "ssbl_pin"
    settings.use_evaluated_mesh = True
    settings.preview_writeback_interval = 0
    settings.preview_target_fps = 30.0
    settings.use_volume_pressure = False
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.substeps = 2
    settings.iterations = 1
    settings.frame_count = 24


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        _clear_scene()
        obj = _make_cloth()
        bpy.context.view_layer.objects.active = obj
        original_mesh = obj.data
        before = _snapshot(obj)
        _configure()
        settings = bpy.context.scene.ssbl_preview

        session = ssbl.solver.start_preview(bpy.context, obj)
        ssbl.solver.step_preview(bpy.context, obj.name)
        first_writeback = bool(ssbl.solver.session_diagnostics(obj).writeback_performed)

        session.adaptive_writeback_interval = 1
        session.frame_ms_ewma = 0.0
        session.writeback_ms_ewma = 0.0
        session_manager._update_adaptive_writeback_interval(
            session,
            session_manager.FramePerf(frame_ms=80.0, writeback_ms=25.0, writeback_performed=True),
        )
        adaptive_raised = int(session.adaptive_writeback_interval) > 1

        session.adaptive_writeback_interval = 8
        ssbl.solver.step_preview(bpy.context, obj.name)
        skipped_writeback = not bool(ssbl.solver.session_diagnostics(obj).writeback_performed)

        settings.gravity = (0.0, 0.0, -28.0)
        settings.hardness = 0.2
        ssbl.solver.step_preview(bpy.context, obj.name)
        tuning_writeback = bool(ssbl.solver.session_diagnostics(obj).writeback_performed)
        still_running = bool(ssbl.solver.has_session(obj))
        finite = _finite_object(obj) and bool(ssbl.solver.session_diagnostics(obj).finite)
        slot = session.slots[obj.name]
        solver_signature_changed = bool(slot.solver_options_signature == session_manager._solver_options_signature(
            session_manager.settings_to_options(settings, runtime_mode_override="preview"),
            settings,
        ))

        stopped = ssbl.solver.request_stop(obj)
        restored_mesh = obj.data == original_mesh
        restore_delta = _max_source_delta(obj, before)

        result = {
            "first_writeback": first_writeback,
            "adaptive_raised": adaptive_raised,
            "skipped_writeback": skipped_writeback,
            "tuning_writeback": tuning_writeback,
            "still_running": still_running,
            "finite": finite,
            "solver_signature_changed": solver_signature_changed,
            "stopped": bool(stopped),
            "restored_mesh": bool(restored_mesh),
            "restore_delta": float(restore_delta),
            "adaptive_interval": int(session.adaptive_writeback_interval),
        }
        print("SSBL_REALTIME_TUNING_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            first_writeback
            and adaptive_raised
            and skipped_writeback
            and tuning_writeback
            and still_running
            and finite
            and solver_signature_changed
            and stopped
            and restored_mesh
            and restore_delta == 0.0
        ):
            raise RuntimeError(f"Realtime tuning smoke failed: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
