from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager

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


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95 + 0.999999))
    return ordered[index]


def _rounded_or_none(value: float | None) -> float | None:
    return round(float(value), 2) if value is not None else None


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
    if self_collision:
        settings.self_collision_mode = "fast"
    settings.self_collision_interval = (
        _int_env("SSBL_SUZANNE_SELF_COLLISION_INTERVAL", 4 if optimized and self_collision else 1)
        if self_collision
        else 1
    )
    settings.max_self_collision_neighbors = 256 if self_collision else 32
    settings.self_probe_interval = (
        _int_env("SSBL_SUZANNE_SELF_PROBE_INTERVAL", 16 if optimized and self_collision else 1)
        if self_collision
        else 1
    )
    settings.self_surface_pair_interval = (
        _int_env("SSBL_SUZANNE_SELF_SURFACE_PAIR_INTERVAL", 4 if optimized and self_collision else 1)
        if self_collision
        else 1
    )
    settings.use_volume_pressure = bool(volume)
    settings.volume_solve_interval = 2 if optimized and volume else 1
    settings.volume_compliance = 1.0e-6
    settings.pressure_strength = 1.0
    settings.volume_target_scale = 1.0
    settings.collision_margin = 0.005
    settings.substeps = 8
    settings.iterations = 1
    settings.preview_writeback_interval = _int_env(
        "SSBL_SUZANNE_WRITEBACK_INTERVAL",
        4 if optimized else 1,
    )
    settings.frame_count = max(_int_env("SSBL_SUZANNE_STEPS", 8), 1)


def _run_case(obj: bpy.types.Object, label: str, *, self_collision: bool, volume: bool, optimized: bool) -> dict[str, object]:
    settings = bpy.context.scene.ssbl_preview
    _configure(settings, self_collision=self_collision, volume=volume, optimized=optimized)
    configured_writeback_interval = max(int(getattr(settings, "preview_writeback_interval", 1)), 1)
    before = _source_snapshot(obj)
    session = ssbl.solver.start_preview(bpy.context, obj)
    steps = int(settings.frame_count)
    warmup = min(max(_int_env("SSBL_SUZANNE_WARMUP", 1), 0), max(steps - 1, 0))
    started = time.perf_counter()
    measured_steps = 0
    measured_frame_ms_total = 0.0
    writeback_frame_count = 0
    writeback_frame_ms_total = 0.0
    non_writeback_frame_count = 0
    non_writeback_frame_ms_total = 0.0
    frame_ms_samples: list[float] = []
    writeback_frame_ms_samples: list[float] = []
    non_writeback_frame_ms_samples: list[float] = []
    cuda_call_ms_samples: list[float] = []
    first_non_writeback_frame_ms = None
    first_writeback_frame_ms = None
    first_writeback_diag = None
    for index in range(steps):
        if index == warmup:
            started = time.perf_counter()
            measured_steps = 0
            measured_frame_ms_total = 0.0
            writeback_frame_count = 0
            writeback_frame_ms_total = 0.0
            non_writeback_frame_count = 0
            non_writeback_frame_ms_total = 0.0
            frame_ms_samples.clear()
            writeback_frame_ms_samples.clear()
            non_writeback_frame_ms_samples.clear()
            cuda_call_ms_samples.clear()
        frame_started = time.perf_counter()
        finished = ssbl.solver.step_preview(bpy.context, obj.name)
        frame_elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
        current_diag = ssbl.solver.session_diagnostics(obj)
        if index >= warmup:
            measured_steps += 1
            measured_frame_ms_total += frame_elapsed_ms
            frame_ms_samples.append(frame_elapsed_ms)
            cuda_call_ms_samples.append(float(current_diag.cuda_step_call_ms))
            if current_diag.writeback_performed:
                writeback_frame_count += 1
                writeback_frame_ms_total += frame_elapsed_ms
                writeback_frame_ms_samples.append(frame_elapsed_ms)
                if first_writeback_frame_ms is None:
                    first_writeback_frame_ms = frame_elapsed_ms
                if first_writeback_diag is None:
                    first_writeback_diag = current_diag
            else:
                non_writeback_frame_count += 1
                non_writeback_frame_ms_total += frame_elapsed_ms
                non_writeback_frame_ms_samples.append(frame_elapsed_ms)
                if first_non_writeback_frame_ms is None:
                    first_non_writeback_frame_ms = frame_elapsed_ms
        if finished:
            break
    elapsed = max(time.perf_counter() - started, 1.0e-6)
    diag = ssbl.solver.session_diagnostics(obj)
    ssbl.solver.request_stop(obj)
    return {
        "case": label,
        "verts": len(session.cloth.positions_world),
        "tris": len(session.cloth.triangles),
        "writeback_interval": configured_writeback_interval,
        "measured_steps": measured_steps,
        "fps": round(max(measured_steps, 1) / elapsed, 2),
        "avg_frame_ms": round(measured_frame_ms_total / max(measured_steps, 1), 2),
        "p95_frame_ms": _rounded_or_none(_p95(frame_ms_samples)),
        "writeback_frame_count": writeback_frame_count,
        "synchronized_writeback_frame_count": writeback_frame_count,
        "avg_writeback_frame_ms": (
            round(writeback_frame_ms_total / writeback_frame_count, 2) if writeback_frame_count > 0 else None
        ),
        "p95_writeback_frame_ms": _rounded_or_none(_p95(writeback_frame_ms_samples)),
        "non_writeback_frame_count": non_writeback_frame_count,
        "queued_frame_count": non_writeback_frame_count,
        "avg_non_writeback_frame_ms": (
            round(non_writeback_frame_ms_total / non_writeback_frame_count, 2) if non_writeback_frame_count > 0 else None
        ),
        "p95_non_writeback_frame_ms": _rounded_or_none(_p95(non_writeback_frame_ms_samples)),
        "avg_cuda_call_ms": round(sum(cuda_call_ms_samples) / max(len(cuda_call_ms_samples), 1), 2),
        "finite": bool(diag.finite),
        "restore_delta": _max_source_delta(obj, before),
        "step_ms": round(float(diag.step_ms), 2),
        "constraints_ms": round(float(diag.constraints_ms), 2),
        "volume_ms": round(float(diag.volume_ms), 2),
        "static_collision_ms": round(float(diag.static_collision_ms), 2),
        "dynamic_collision_ms": round(float(diag.dynamic_collision_ms), 2),
        "self_hash_ms": round(float(diag.self_hash_ms), 2),
        "self_solve_ms": round(float(diag.self_solve_ms), 2),
        "self_probe_ms": round(float(diag.self_probe_ms), 2),
        "self_recovery_ms": round(float(diag.self_recovery_ms), 2),
        "sync_ms": round(float(diag.sync_ms), 2),
        "native_diag_fetch_ms": round(float(diag.diagnostics_fetch_ms), 2),
        "frame_ms": round(float(diag.frame_ms), 2),
        "input_ms": round(float(diag.input_refresh_ms), 2),
        "frame_input_upload_ms": round(float(diag.frame_input_upload_ms), 2),
        "cuda_call_ms": round(float(diag.cuda_step_call_ms), 2),
        "download_ms": round(float(diag.download_ms), 2),
        "writeback_ms": round(float(diag.writeback_ms), 2),
        "to_local_ms": round(float(diag.writeback_to_local_ms), 2),
        "foreach_set_ms": round(float(diag.writeback_foreach_set_ms), 2),
        "mesh_update_ms": round(float(diag.writeback_mesh_update_ms), 2),
        "writeback_frame": bool(diag.writeback_performed),
        "sampled_non_writeback_frame_ms": (
            round(float(first_non_writeback_frame_ms), 2) if first_non_writeback_frame_ms is not None else None
        ),
        "sampled_non_writeback_step_ms": None,
        "sampled_writeback_frame_ms": (
            round(float(first_writeback_frame_ms), 2) if first_writeback_frame_ms is not None else None
        ),
        "sampled_writeback_step_ms": (
            round(float(first_writeback_diag.step_ms), 2) if first_writeback_diag is not None else None
        ),
        "candidates": int(diag.candidate_count),
        "resolved": int(diag.resolved_contacts),
        "ccd_clamp_count": int(diag.ccd_clamp_count),
        "recovery_passes": int(diag.recovery_passes),
        "local_retry_count": int(diag.local_retry_count),
        "jitter_stabilized_vertices": int(diag.jitter_stabilized_vertices),
        "jitter_rejected_vertices": int(diag.jitter_rejected_vertices),
        "jitter_max_correction": round(float(diag.jitter_max_correction), 6),
        "force_field_count": int(getattr(diag, "force_field_count", 0)),
        "unsupported_force_field_count": int(getattr(diag, "unsupported_force_field_count", 0)),
    }


def _run_case_with_env(
    obj: bpy.types.Object,
    label: str,
    *,
    self_collision: bool,
    volume: bool,
    optimized: bool,
    env: dict[str, str | None],
) -> dict[str, object]:
    with _temporary_env(env):
        return _run_case(obj, label, self_collision=self_collision, volume=volume, optimized=optimized)


def _append_compare_cases(results: list[dict[str, object]], obj: bpy.types.Object, requested_cases: set[str]) -> None:
    compare_requested = not requested_cases or "optimized_self_and_volume" in requested_cases
    if not compare_requested:
        return

    if _bool_env("SSBL_SUZANNE_COMPARE_WRITEBACK", False):
        for interval in (1, 2, 4, 8):
            results.append(
                _run_case_with_env(
                    obj,
                    f"optimized_self_and_volume_wb{interval}",
                    self_collision=True,
                    volume=True,
                    optimized=True,
                    env={"SSBL_SUZANNE_WRITEBACK_INTERVAL": str(interval)},
                )
            )


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
        case_specs = [
            ("base_no_self_no_volume", False, False, False),
            ("volume_only", False, True, False),
            ("self_only", True, False, False),
            ("self_and_volume", True, True, False),
            ("optimized_self_and_volume", True, True, True),
        ]
        requested_cases = {
            item.strip()
            for item in os.environ.get("SSBL_SUZANNE_CASES", "").split(",")
            if item.strip()
        }
        results = [
            _run_case(obj, label, self_collision=self_collision, volume=volume, optimized=optimized)
            for label, self_collision, volume, optimized in case_specs
            if not requested_cases or label in requested_cases
        ]
        _append_compare_cases(results, obj, requested_cases)
        print("SSBL_SUZANNE_BENCH", json.dumps(results, sort_keys=True))
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
