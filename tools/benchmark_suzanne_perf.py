from __future__ import annotations

import json
import os
import sys
import time

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _target_object() -> bpy.types.Object:
    obj = bpy.data.objects.get(os.environ.get("SSBL_SUZANNE_OBJECT", "Suzanne"))
    if obj is not None and obj.type == "MESH":
        return obj
    active = bpy.context.active_object
    if active is not None and active.type == "MESH":
        return active
    raise RuntimeError("Suzanne benchmark target mesh not found")


def _ensure_pin_group(obj: bpy.types.Object, group_name: str) -> None:
    if obj.vertex_groups.get(group_name) is not None:
        return
    z_values = [vertex.co.z for vertex in obj.data.vertices]
    threshold = max(z_values) - (max(z_values) - min(z_values)) * 0.18
    indices = [vertex.index for vertex in obj.data.vertices if vertex.co.z >= threshold]
    if not indices:
        indices = [max(obj.data.vertices, key=lambda vertex: vertex.co.z).index]
    group = obj.vertex_groups.new(name=group_name)
    group.add(indices, 1.0, "ADD")


def _source_snapshot(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(v.co.x), float(v.co.y), float(v.co.z)) for v in obj.data.vertices]


def _max_source_delta(obj: bpy.types.Object, before: list[tuple[float, float, float]]) -> float:
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


def _configure(settings, *, self_collision: bool, volume: bool, optimized: bool) -> None:
    settings.pin_vertex_group = "ssbl_pin"
    settings.use_evaluated_mesh = True
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.self_collision_mode = "fast" if self_collision else "off"
    settings.self_collision_interval = 4 if self_collision else 1
    settings.max_self_collision_neighbors = 256 if self_collision else 32
    settings.self_probe_interval = 2 if optimized and self_collision else 1
    settings.self_surface_pair_interval = 2 if optimized and self_collision else 1
    settings.use_volume_pressure = bool(volume)
    settings.volume_solve_interval = 2 if optimized and volume else 1
    settings.volume_compliance = 1.0e-6
    settings.pressure_strength = 1.0
    settings.volume_target_scale = 1.0
    settings.collision_margin = 0.005
    settings.substeps = 8
    settings.iterations = 1
    settings.preview_writeback_interval = 2 if optimized else 1
    settings.frame_count = max(_int_env("SSBL_SUZANNE_STEPS", 8), 1)


def _run_case(obj: bpy.types.Object, label: str, *, self_collision: bool, volume: bool, optimized: bool) -> dict[str, object]:
    settings = bpy.context.scene.ssbl_preview
    _configure(settings, self_collision=self_collision, volume=volume, optimized=optimized)
    before = _source_snapshot(obj)
    session = ssbl.solver.start_preview(bpy.context, obj)
    steps = int(settings.frame_count)
    warmup = min(max(_int_env("SSBL_SUZANNE_WARMUP", 1), 0), max(steps - 1, 0))
    started = time.perf_counter()
    measured_steps = 0
    for index in range(steps):
        finished = ssbl.solver.step_preview(bpy.context, obj.name)
        if index == warmup:
            started = time.perf_counter()
            measured_steps = 0
        if index >= warmup:
            measured_steps += 1
        if finished:
            break
    elapsed = max(time.perf_counter() - started, 1.0e-6)
    diag = ssbl.solver.session_diagnostics(obj)
    ssbl.solver.request_stop(obj)
    return {
        "case": label,
        "verts": len(session.cloth.positions_world),
        "tris": len(session.cloth.triangles),
        "measured_steps": measured_steps,
        "fps": round(max(measured_steps, 1) / elapsed, 2),
        "finite": bool(diag.finite),
        "restore_delta": _max_source_delta(obj, before),
        "step_ms": round(float(diag.step_ms), 2),
        "frame_ms": round(float(diag.frame_ms), 2),
        "input_ms": round(float(diag.input_refresh_ms), 2),
        "download_ms": round(float(diag.download_ms), 2),
        "writeback_ms": round(float(diag.writeback_ms), 2),
        "candidates": int(diag.candidate_count),
        "resolved": int(diag.resolved_contacts),
    }


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        obj = _target_object()
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        _ensure_pin_group(obj, "ssbl_pin")
        results = [
            _run_case(obj, "base_no_self_no_volume", self_collision=False, volume=False, optimized=False),
            _run_case(obj, "volume_only", self_collision=False, volume=True, optimized=False),
            _run_case(obj, "self_only", self_collision=True, volume=False, optimized=False),
            _run_case(obj, "self_and_volume", self_collision=True, volume=True, optimized=False),
            _run_case(obj, "optimized_self_and_volume", self_collision=True, volume=True, optimized=True),
        ]
        print("SSBL_SUZANNE_BENCH", json.dumps(results, sort_keys=True))
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
