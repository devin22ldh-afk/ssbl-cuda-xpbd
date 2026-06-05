from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import bpy


ADDONS_ROOT = os.environ.get(
    "SSBL_ADDONS_ROOT",
    r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons",
)
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

DEFAULT_BLEND_PATH = r"C:\Users\Administrator\Desktop\cs2.blend"
DEFAULT_ACTIVE_OBJECT = "Cube"
DEFAULT_OTHER_OBJECT = "Suzanne"


def _load_cs2_regression_module():
    path = Path(__file__).with_name("cs2_self_collision_intersection_regression.py")
    spec = importlib.util.spec_from_file_location("ssbl_cs2_regression_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load CS2 regression helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Run the real CS2 SSBL multi-cloth performance probe once.")
    parser.add_argument("--blend", default=os.environ.get("SSBL_CS2_BLEND", DEFAULT_BLEND_PATH))
    parser.add_argument("--active", default=os.environ.get("SSBL_CS2_ACTIVE_OBJECT", DEFAULT_ACTIVE_OBJECT))
    parser.add_argument("--other", default=os.environ.get("SSBL_CS2_OTHER_OBJECT", DEFAULT_OTHER_OBJECT))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("SSBL_CS2_STEPS", "60")))
    parser.add_argument("--hardness", type=float, default=float(os.environ.get("SSBL_CS2_HARDNESS", "1.0")))
    parser.add_argument("--max-report", type=int, default=int(os.environ.get("SSBL_CS2_MAX_REPORT", "1000")))
    parser.add_argument(
        "--fail-under-fps",
        type=float,
        default=float(os.environ.get("SSBL_CS2_FAIL_UNDER_FPS", "0.0")),
        help="Optional perf gate. 0 disables failure on FPS.",
    )
    return parser.parse_args(argv)


def _register_addon():
    import ssbl

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    return ssbl


def _mesh_object(name: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"Expected mesh object {name!r} in cs2.blend")
    return obj


def _select_multicloth(active: bpy.types.Object, other: bpy.types.Object) -> None:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    other.select_set(True)
    active.select_set(True)
    bpy.context.view_layer.objects.active = active


def _snapshot(obj: bpy.types.Object) -> list[float]:
    return [
        float(component)
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    ]


def _max_delta(before: list[float], after: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(before, after)), default=0.0)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(len(ordered) * 0.95), 0), len(ordered) - 1)
    return float(ordered[index])


def _finite_mesh(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _diag_summary(diag) -> dict[str, object]:
    return {
        "native_step_ms": float(getattr(diag, "step_ms", 0.0)),
        "native_cuda_step_call_ms": float(getattr(diag, "cuda_step_call_ms", 0.0)),
        "native_hash_build_ms": float(getattr(diag, "hash_build_ms", 0.0)),
        "native_constraints_ms": float(getattr(diag, "constraints_ms", 0.0)),
        "native_static_collision_ms": float(getattr(diag, "static_collision_ms", 0.0)),
        "native_dynamic_collision_ms": float(getattr(diag, "dynamic_collision_ms", 0.0)),
        "native_dynamic_particle_collision_ms": float(getattr(diag, "dynamic_particle_collision_ms", 0.0)),
        "native_self_hash_ms": float(getattr(diag, "self_hash_ms", 0.0)),
        "native_self_solve_ms": float(getattr(diag, "self_solve_ms", 0.0)),
        "native_self_probe_ms": float(getattr(diag, "self_probe_ms", 0.0)),
        "native_self_recovery_ms": float(getattr(diag, "self_recovery_ms", 0.0)),
        "frame_ms": float(getattr(diag, "frame_ms", 0.0)),
        "frame_set_ms": float(getattr(diag, "frame_set_ms", 0.0)),
        "input_refresh_ms": float(getattr(diag, "input_refresh_ms", 0.0)),
        "download_ms": float(getattr(diag, "download_ms", 0.0)),
        "writeback_ms": float(getattr(diag, "writeback_ms", 0.0)),
        "writeback_to_local_ms": float(getattr(diag, "writeback_to_local_ms", 0.0)),
        "writeback_foreach_set_ms": float(getattr(diag, "writeback_foreach_set_ms", 0.0)),
        "writeback_mesh_update_ms": float(getattr(diag, "writeback_mesh_update_ms", 0.0)),
        "diagnostics_ms": float(getattr(diag, "diagnostics_ms", 0.0)),
        "dynamic_upload_ms": float(getattr(diag, "dynamic_upload_ms", 0.0)),
        "dynamic_collider_pack_ms": float(getattr(diag, "dynamic_collider_pack_ms", 0.0)),
        "dynamic_triangle_upload_ms": float(getattr(diag, "dynamic_triangle_upload_ms", 0.0)),
        "dynamic_particle_upload_ms": float(getattr(diag, "dynamic_particle_upload_ms", 0.0)),
        "dynamic_collider_cache_hits": int(getattr(diag, "dynamic_collider_cache_hits", 0)),
        "dynamic_collider_cache_misses": int(getattr(diag, "dynamic_collider_cache_misses", 0)),
        "dynamic_triangle_count": int(getattr(diag, "dynamic_triangle_count", 0)),
        "dynamic_particle_count": int(getattr(diag, "dynamic_particle_count", 0)),
        "dynamic_particle_candidate_count": int(getattr(diag, "dynamic_particle_candidate_count", 0)),
        "dynamic_particle_contacts": int(getattr(diag, "dynamic_particle_contacts", 0)),
        "dynamic_particle_overflow": int(getattr(diag, "dynamic_particle_overflow", 0)),
        "resolved_contacts": int(getattr(diag, "resolved_contacts", 0)),
        "candidate_count": int(getattr(diag, "candidate_count", 0)),
        "min_gap": getattr(diag, "min_gap", None),
        "pcg_iterations": int(getattr(diag, "abi41_pcg_iterations", 0)),
        "pcg_csr_nnz": int(getattr(diag, "abi41_pcg_csr_nnz", 0)),
        "pcg_texture_ready": int(getattr(diag, "abi41_pcg_texture_ready", 0)),
        "pcg_initial_residual": float(getattr(diag, "abi41_pcg_initial_residual", 0.0)),
        "pcg_final_residual": float(getattr(diag, "abi41_pcg_final_residual", 0.0)),
        "pcg_max_delta": float(getattr(diag, "abi41_pcg_max_delta", 0.0)),
        "abi41_lra_tack_count": int(getattr(diag, "abi41_lra_tack_count", 0)),
        "abi41_bending_wing_count": int(getattr(diag, "abi41_bending_wing_count", 0)),
        "abi41_bending_texture_ready": int(getattr(diag, "abi41_bending_texture_ready", 0)),
    }


def _configure_cloth(obj: bpy.types.Object, cs2, steps: int, hardness: float) -> None:
    from ssbl.xpbd_core import sync_hardness_settings

    settings = obj.ssbl_cloth
    settings.enabled = True
    settings.preview_writeback_interval = 0
    settings.hardness = max(0.0, min(1.0, float(hardness)))
    settings.hardness_initialized = True
    sync_hardness_settings(settings)
    cs2._apply_object_overrides(settings, steps)
    sync_hardness_settings(settings)


def main() -> None:
    args = _parse_args()
    steps = max(int(args.steps), 1)
    hardness = max(0.0, min(1.0, float(args.hardness)))
    bpy.ops.wm.open_mainfile(filepath=args.blend, load_ui=False)
    ssbl = _register_addon()
    from ssbl import xpbd_core
    from ssbl.xpbd_core import settings_to_options

    cs2 = _load_cs2_regression_module()
    active = _mesh_object(args.active)
    other = _mesh_object(args.other)
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.frame_start = 1
    scene.frame_end = max(int(scene.frame_end), steps + 2)

    for obj in (active, other):
        obj.hide_viewport = False
        _configure_cloth(obj, cs2, steps, hardness)
    _select_multicloth(active, other)

    before_active = _snapshot(active)
    before_other = _snapshot(other)
    rest_triangles_active = xpbd_core.triangulated_faces(active.data)
    rest_triangles_other = xpbd_core.triangulated_faces(other.data)
    initial_active = cs2._check_intersections(active, rest_triangles_active, int(args.max_report))
    initial_other = cs2._check_intersections(other, rest_triangles_other, int(args.max_report))
    active_options = settings_to_options(active.ssbl_cloth, runtime_mode_override="preview")
    other_options = settings_to_options(other.ssbl_cloth, runtime_mode_override="preview")

    started_at = time.perf_counter()
    session = ssbl.solver.start_preview(bpy.context, active)
    step_times: list[float] = []
    max_diag: dict[str, float | int] = {
        "dynamic_upload_ms": 0.0,
        "dynamic_collider_pack_ms": 0.0,
        "native_cuda_step_call_ms": 0.0,
        "dynamic_triangle_upload_ms": 0.0,
        "dynamic_particle_upload_ms": 0.0,
        "dynamic_triangle_count": 0,
        "dynamic_particle_count": 0,
        "dynamic_particle_contacts": 0,
        "dynamic_particle_overflow": 0,
    }
    cache_hits_total = 0
    cache_misses_total = 0
    finite = True
    for _index in range(steps):
        step_started = time.perf_counter()
        ssbl.solver.step_preview(bpy.context, active.name)
        step_times.append((time.perf_counter() - step_started) * 1000.0)
        diag = ssbl.solver.session_diagnostics(active)
        finite = finite and bool(getattr(diag, "finite", True))
        for key in (
            "dynamic_upload_ms",
            "dynamic_collider_pack_ms",
            "cuda_step_call_ms",
            "dynamic_triangle_upload_ms",
            "dynamic_particle_upload_ms",
        ):
            out_key = "native_cuda_step_call_ms" if key == "cuda_step_call_ms" else key
            max_diag[out_key] = max(float(max_diag[out_key]), float(getattr(diag, key, 0.0)))
        for key in (
            "dynamic_triangle_count",
            "dynamic_particle_count",
            "dynamic_particle_contacts",
            "dynamic_particle_overflow",
        ):
            max_diag[key] = max(int(max_diag[key]), int(getattr(diag, key, 0)))
        cache_hits_total += int(getattr(diag, "dynamic_collider_cache_hits", 0))
        cache_misses_total += int(getattr(diag, "dynamic_collider_cache_misses", 0))

    elapsed = time.perf_counter() - started_at
    final_diag = ssbl.solver.session_diagnostics(active)
    stopped = ssbl.solver.request_stop(active)
    after_active = _snapshot(active)
    after_other = _snapshot(other)
    final_active = cs2._check_intersections(active, xpbd_core.triangulated_faces(active.data), int(args.max_report))
    final_other = cs2._check_intersections(other, xpbd_core.triangulated_faces(other.data), int(args.max_report))
    restore_active = _max_delta(before_active, after_active)
    restore_other = _max_delta(before_other, after_other)
    average_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    fps = (1000.0 / average_step_ms) if average_step_ms > 0.0 else 0.0
    correctness_pass = (
        bool(finite)
        and _finite_mesh(active)
        and _finite_mesh(other)
        and restore_active <= 1.0e-7
        and restore_other <= 1.0e-7
    )
    summary = {
        "blend_file": bpy.data.filepath,
        "active_object": active.name,
        "other_object": other.name,
        "selected_after_setup": [obj.name for obj in bpy.context.selected_objects],
        "active_after_setup": bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else None,
        "steps": int(steps),
        "hardness": float(hardness),
        "elapsed_s": float(elapsed),
        "average_wall_step_ms": float(average_step_ms),
        "p95_wall_step_ms": _p95(step_times),
        "sim_fps_wall": float(fps),
        "slots": len(session.slots),
        "slot_names": list(session.slots.keys()),
        "solve_order": list(session.solve_order),
        "cross_mode": str(session.cross_cloth_mode),
        "finite": bool(finite and _finite_mesh(active) and _finite_mesh(other)),
        "stopped": bool(stopped),
        "restore_delta_active": float(restore_active),
        "restore_delta_other": float(restore_other),
        "correctness_pass": bool(correctness_pass),
        "active_vertex_count": int(len(active.data.vertices)),
        "other_vertex_count": int(len(other.data.vertices)),
        "active_triangle_count": int(len(xpbd_core.triangulated_faces(active.data))),
        "other_triangle_count": int(len(xpbd_core.triangulated_faces(other.data))),
        "active_stretch_optimization_enabled": bool(active_options.stretch_optimization_enabled),
        "active_stretch_optimization_strength": float(active_options.stretch_optimization_strength),
        "other_stretch_optimization_enabled": bool(other_options.stretch_optimization_enabled),
        "other_stretch_optimization_strength": float(other_options.stretch_optimization_strength),
        "cache_hits_total": int(cache_hits_total),
        "cache_misses_total": int(cache_misses_total),
        "max_dynamic_upload_ms": float(max_diag["dynamic_upload_ms"]),
        "max_dynamic_collider_pack_ms": float(max_diag["dynamic_collider_pack_ms"]),
        "max_native_cuda_step_call_ms": float(max_diag["native_cuda_step_call_ms"]),
        "max_dynamic_triangle_upload_ms": float(max_diag["dynamic_triangle_upload_ms"]),
        "max_dynamic_particle_upload_ms": float(max_diag["dynamic_particle_upload_ms"]),
        "max_dynamic_triangle_count": int(max_diag["dynamic_triangle_count"]),
        "max_dynamic_particle_count": int(max_diag["dynamic_particle_count"]),
        "max_dynamic_particle_contacts": int(max_diag["dynamic_particle_contacts"]),
        "max_dynamic_particle_overflow": int(max_diag["dynamic_particle_overflow"]),
        "initial_active": initial_active,
        "initial_other": initial_other,
        "final_active": final_active,
        "final_other": final_other,
        **_diag_summary(final_diag),
    }
    print("SSBL_CS2_MULTICLOTH_PERF_PROBE", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not summary["finite"] or restore_active > 1.0e-7 or restore_other > 1.0e-7:
        raise RuntimeError(f"CS2 multicloth perf probe failed stability gate: {summary}")
    if float(args.fail_under_fps) > 0.0 and fps < float(args.fail_under_fps):
        raise RuntimeError(f"CS2 multicloth perf probe below FPS gate {args.fail_under_fps}: {fps:.2f}")


if __name__ == "__main__":
    main()
