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


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

DEFAULT_BLEND_PATH = r"C:\Users\Administrator\Desktop\cs2.blend"
DEFAULT_HARDNESS = (0.0, 0.4, 1.0)


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
    parser = argparse.ArgumentParser(description="Run a single-pass CS2 SSBL performance benchmark.")
    parser.add_argument("--blend", default=os.environ.get("SSBL_CS2_BLEND", DEFAULT_BLEND_PATH))
    parser.add_argument("--target", default=os.environ.get("SSBL_CS2_TARGET", "Suzanne"))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("SSBL_CS2_STEPS", "60")))
    parser.add_argument("--max-report", type=int, default=int(os.environ.get("SSBL_CS2_MAX_REPORT", "1000")))
    parser.add_argument(
        "--fail-on-intersection",
        action="store_true",
        default=os.environ.get("SSBL_CS2_BENCH_FAIL_ON_INTERSECTION", "0").strip().lower()
        in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--hardness",
        default=",".join(f"{value:g}" for value in DEFAULT_HARDNESS),
        help="Comma-separated hardness cases. Default: 0,0.4,1",
    )
    return parser.parse_args(argv)


def _parse_hardness(value: str) -> list[float]:
    result: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        hardness = max(0.0, min(1.0, float(item)))
        if hardness not in result:
            result.append(hardness)
    return result or list(DEFAULT_HARDNESS)


def _register_addon():
    import ssbl

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    return ssbl


def _find_target(name: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == "MESH":
        return obj
    active = bpy.context.view_layer.objects.active
    if active is not None and active.type == "MESH":
        return active
    raise RuntimeError(f"Target mesh object not found: {name}")


def _select_only(obj: bpy.types.Object) -> None:
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _source_snapshot(obj: bpy.types.Object) -> list[float]:
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


def _diagnostic_summary(diag) -> dict[str, object]:
    return {
        "native_step_ms": float(getattr(diag, "step_ms", 0.0)),
        "native_cuda_step_call_ms": float(getattr(diag, "cuda_step_call_ms", 0.0)),
        "native_self_hash_ms": float(getattr(diag, "self_hash_ms", 0.0)),
        "native_self_solve_ms": float(getattr(diag, "self_solve_ms", 0.0)),
        "native_self_probe_ms": float(getattr(diag, "self_probe_ms", 0.0)),
        "native_self_recovery_ms": float(getattr(diag, "self_recovery_ms", 0.0)),
        "native_static_collision_ms": float(getattr(diag, "static_collision_ms", 0.0)),
        "native_dynamic_collision_ms": float(getattr(diag, "dynamic_collision_ms", 0.0)),
        "native_dynamic_particle_collision_ms": float(getattr(diag, "dynamic_particle_collision_ms", 0.0)),
        "native_min_gap": getattr(diag, "min_gap", None),
        "native_resolved_contacts": int(getattr(diag, "resolved_contacts", 0)),
        "native_candidate_count": int(getattr(diag, "candidate_count", 0)),
        "native_recovery_passes": int(getattr(diag, "recovery_passes", 0)),
        "pcg_iterations": int(getattr(diag, "abi41_pcg_iterations", 0)),
        "pcg_guarded": int(getattr(diag, "abi41_pcg_guarded", 0)),
        "pcg_csr_nnz": int(getattr(diag, "abi41_pcg_csr_nnz", 0)),
        "pcg_texture_ready": int(getattr(diag, "abi41_pcg_texture_ready", 0)),
        "pcg_initial_residual": float(getattr(diag, "abi41_pcg_initial_residual", 0.0)),
        "pcg_final_residual": float(getattr(diag, "abi41_pcg_final_residual", 0.0)),
        "pcg_max_delta": float(getattr(diag, "abi41_pcg_max_delta", 0.0)),
    }


def _run_case(
    cs2,
    blend_path: str,
    target: str,
    steps: int,
    max_report: int,
    hardness: float,
    fail_on_intersection: bool,
) -> dict[str, object]:
    bpy.ops.wm.open_mainfile(filepath=blend_path, load_ui=False)
    ssbl = _register_addon()
    from ssbl import xpbd_core
    from ssbl.xpbd_core import settings_to_options, sync_hardness_settings

    obj = _find_target(target)
    _select_only(obj)
    if not hasattr(obj, "ssbl_cloth"):
        raise RuntimeError(f"{obj.name} has no ssbl_cloth settings")

    settings = obj.ssbl_cloth
    settings.hardness = float(hardness)
    settings.hardness_initialized = True
    sync_hardness_settings(settings)
    cs2._apply_object_overrides(settings, steps)
    sync_hardness_settings(settings)
    options = settings_to_options(settings, runtime_mode_override="preview")

    pin_count = cs2._ensure_pin_group(obj, settings)
    before = _source_snapshot(obj)
    rest_triangles = xpbd_core.triangulated_faces(obj.data)
    initial_collision = cs2._check_intersections(obj, rest_triangles, max_report)

    started_at = time.perf_counter()
    ssbl.solver.start_preview(bpy.context, obj)
    step_times: list[float] = []
    for _index in range(max(1, int(steps))):
        step_start = time.perf_counter()
        ssbl.solver.step_preview(bpy.context, obj.name)
        step_times.append((time.perf_counter() - step_start) * 1000.0)

    triangles = xpbd_core.triangulated_faces(obj.data)
    coords_finite = all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )
    collision = cs2._check_intersections(obj, triangles, max_report)
    diag = ssbl.solver.session_diagnostics(obj)
    elapsed = time.perf_counter() - started_at
    ssbl.solver.request_stop(obj)

    after = _source_snapshot(obj)
    restore_delta = _max_delta(before, after)
    average_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    solver_intersections = int(collision["solver_relevant_intersection_count_capped"])
    correctness_pass = bool(coords_finite) and restore_delta <= 1.0e-7 and solver_intersections == 0
    summary = {
        "hardness": float(hardness),
        "blend_file": bpy.data.filepath,
        "object": obj.name,
        "steps": int(steps),
        "elapsed_s": float(elapsed),
        "average_wall_step_ms": float(average_step_ms),
        "p95_wall_step_ms": _p95(step_times),
        "sim_fps_wall": float(1000.0 / average_step_ms) if average_step_ms > 0.0 else 0.0,
        "substeps": int(settings.substeps),
        "iterations": int(settings.iterations),
        "pin_vertex_count": int(pin_count),
        "self_collision": bool(settings.self_collision),
        "self_collision_mode": str(settings.self_collision_mode),
        "self_collision_interval": int(settings.self_collision_interval),
        "max_self_collision_neighbors": int(settings.max_self_collision_neighbors),
        "fast_self_collision_passes": int(getattr(settings, "fast_self_collision_passes", 4)),
        "self_probe_interval": int(settings.self_probe_interval),
        "self_surface_pair_interval": int(settings.self_surface_pair_interval),
        "collision_margin": float(settings.collision_margin),
        "cloth_thickness": float(settings.cloth_thickness),
        "volume_pressure": bool(settings.use_volume_pressure),
        "vertex_count": int(len(obj.data.vertices)),
        "triangle_count": int(len(triangles)),
        "finite": bool(coords_finite),
        "restore_delta": float(restore_delta),
        "correctness_pass": bool(correctness_pass),
        "stretch_optimization_enabled": bool(options.stretch_optimization_enabled),
        "stretch_optimization_strength": float(options.stretch_optimization_strength),
        "stretch_compliance": float(options.stretch_compliance),
        "bend_compliance": float(options.bend_compliance),
        "lra_compliance": float(options.lra_compliance),
        "initial": initial_collision,
        **_diagnostic_summary(diag),
        **collision,
    }

    if not bool(summary["finite"]):
        raise RuntimeError(f"CS2 performance case hardness={hardness:g} produced non-finite vertices")
    if float(summary["restore_delta"]) > 1.0e-7:
        raise RuntimeError(
            f"CS2 performance case hardness={hardness:g} failed restore_delta={summary['restore_delta']}"
        )
    if fail_on_intersection and solver_intersections > 0:
        raise RuntimeError(
            "CS2 performance case hardness="
            f"{hardness:g} failed solver intersections={solver_intersections}"
        )
    return summary


def main() -> None:
    args = _parse_args()
    blend_path = str(Path(args.blend))
    if not Path(blend_path).exists():
        raise RuntimeError(f"CS2 blend file not found: {blend_path}")
    cs2 = _load_cs2_regression_module()
    cases = [
        _run_case(
            cs2,
            blend_path,
            args.target,
            max(1, int(args.steps)),
            max(1, int(args.max_report)),
            hardness,
            bool(args.fail_on_intersection),
        )
        for hardness in _parse_hardness(args.hardness)
    ]
    result = {
        "blend": blend_path,
        "target": str(args.target),
        "steps": max(1, int(args.steps)),
        "pcg_device_scalar_env": os.environ.get("SSBL_ABI41_PCG_DEVICE_SCALAR", "<default>"),
        "cases": cases,
    }
    print("SSBL_CS2_PERF_BENCH " + json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
