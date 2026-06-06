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
from mathutils.bvhtree import BVHTree


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
        "--check-intersections",
        action="store_true",
        default=os.environ.get("SSBL_CS2_PERF_CHECK_INTERSECTIONS", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Run the optional bounded Python self-intersection summary. Disabled by default for timing probes.",
    )
    parser.add_argument(
        "--intersection-pair-limit",
        type=int,
        default=int(os.environ.get("SSBL_CS2_PERF_INTERSECTION_PAIR_LIMIT", "20000")),
        help="Perf-probe-only cap for exact Python triangle-pair tests. Full correctness is checked by regression tools.",
    )
    parser.add_argument(
        "--intersection-budget-ms",
        type=float,
        default=float(os.environ.get("SSBL_CS2_PERF_INTERSECTION_BUDGET_MS", "1500")),
        help="Perf-probe-only approximate budget for exact Python intersection tests. 0 disables this budget.",
    )
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


def _phase_logger():
    path = os.environ.get("SSBL_CS2_PERF_PHASE_LOG", "").strip()
    if not path:
        return lambda _name, **_data: None

    def log(name: str, **data) -> None:
        row = {"phase": name, "t": time.perf_counter(), **data}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except (OSError, ValueError):
        pass
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
    return log


def _heartbeat_enabled() -> bool:
    raw = os.environ.get("SSBL_CS2_PERF_HEARTBEAT", "0").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _heartbeat_interval() -> int:
    try:
        return max(int(os.environ.get("SSBL_CS2_PERF_HEARTBEAT_INTERVAL", "1")), 1)
    except ValueError:
        return 1


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


def _check_intersections_limited(
    cs2,
    obj: bpy.types.Object,
    triangles,
    max_report: int,
    *,
    pair_test_limit: int,
    budget_ms: float,
) -> dict[str, object]:
    started = time.perf_counter()
    vertices = cs2._world_vertices(obj)
    polygons = [tuple(int(vertex) for vertex in tri) for tri in triangles]
    neighbors = cs2._neighbor_sets(triangles, len(vertices))
    tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True, epsilon=0.0)
    overlap_pairs = tree.overlap(tree)
    deadline = started + max(float(budget_ms), 0.0) / 1000.0 if budget_ms > 0.0 else None
    tested = 0
    intersections = []
    solver_relevant_tested = 0
    solver_relevant_intersections = []
    limited_reason = ""

    for ia, ib in overlap_pairs:
        if ia >= ib:
            continue
        tri_a = polygons[ia]
        tri_b = polygons[ib]
        if tri_a[0] in tri_b or tri_a[1] in tri_b or tri_a[2] in tri_b:
            continue
        if pair_test_limit > 0 and tested >= pair_test_limit:
            limited_reason = "pair_test_limit"
            break
        if deadline is not None and time.perf_counter() >= deadline:
            limited_reason = "time_budget_ms"
            break

        tested += 1
        solver_topology_neighbor = cs2._triangles_are_solver_topology_neighbors(tri_a, tri_b, neighbors)
        if not solver_topology_neighbor:
            solver_relevant_tested += 1
        if cs2._triangles_intersect(vertices, tri_a, tri_b):
            intersections.append((int(ia), int(ib), tri_a, tri_b))
            if not solver_topology_neighbor:
                solver_relevant_intersections.append((int(ia), int(ib), tri_a, tri_b))
            if len(intersections) >= max_report:
                limited_reason = "max_report"
                break

    return {
        "overlap_pairs": int(len(overlap_pairs)),
        "tested_non_adjacent_pairs": int(tested),
        "intersection_count_capped": int(len(intersections)),
        "sample_intersections": intersections[:10],
        "solver_relevant_tested_pairs": int(solver_relevant_tested),
        "solver_relevant_intersection_count_capped": int(len(solver_relevant_intersections)),
        "solver_relevant_sample_intersections": solver_relevant_intersections[:10],
        "intersection_check_complete": limited_reason == "",
        "intersection_check_limited_reason": limited_reason,
        "intersection_check_elapsed_ms": float((time.perf_counter() - started) * 1000.0),
        "intersection_check_pair_test_limit": int(pair_test_limit),
        "intersection_check_budget_ms": float(budget_ms),
    }


def _skipped_intersection_check(triangle_count: int) -> dict[str, object]:
    return {
        "overlap_pairs": 0,
        "tested_non_adjacent_pairs": 0,
        "intersection_count_capped": 0,
        "sample_intersections": [],
        "solver_relevant_tested_pairs": 0,
        "solver_relevant_intersection_count_capped": 0,
        "solver_relevant_sample_intersections": [],
        "intersection_check_complete": False,
        "intersection_check_limited_reason": "disabled_for_perf_probe",
        "intersection_check_elapsed_ms": 0.0,
        "intersection_check_pair_test_limit": 0,
        "intersection_check_budget_ms": 0.0,
        "intersection_check_triangle_count": int(triangle_count),
    }


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
        "dynamic_pair_cache_hits": int(getattr(diag, "dynamic_pair_cache_hits", 0)),
        "dynamic_pair_cache_misses": int(getattr(diag, "dynamic_pair_cache_misses", 0)),
        "dynamic_pair_cache_reused_triangles": int(getattr(diag, "dynamic_pair_cache_reused_triangles", 0)),
        "dynamic_pair_cache_reused_particles": int(getattr(diag, "dynamic_pair_cache_reused_particles", 0)),
        "dynamic_collision_skipped_launches": int(getattr(diag, "dynamic_collision_skipped_launches", 0)),
        "self_collision_skipped_launches": int(getattr(diag, "self_collision_skipped_launches", 0)),
        "self_candidate_count": int(getattr(diag, "self_candidate_count", 0)),
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
        "pcg_solve_ms": float(getattr(diag, "abi41_pcg_solve_ms", 0.0)),
        "pcg_system_ms": float(getattr(diag, "abi41_pcg_system_ms", 0.0)),
        "pcg_ad_ms": float(getattr(diag, "abi41_pcg_ad_ms", 0.0)),
        "direct_stretch_ms": float(getattr(diag, "abi41_direct_stretch_ms", 0.0)),
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
    log_phase = _phase_logger()
    heartbeat = _heartbeat_enabled()
    heartbeat_interval = _heartbeat_interval()
    steps = max(int(args.steps), 1)
    hardness = max(0.0, min(1.0, float(args.hardness)))
    log_phase("open_mainfile_start", blend=args.blend)
    bpy.ops.wm.open_mainfile(filepath=args.blend, load_ui=False)
    log_phase("open_mainfile_done")
    ssbl = _register_addon()
    log_phase("addon_registered")
    from ssbl import xpbd_core
    from ssbl.xpbd_core import settings_to_options

    cs2 = _load_cs2_regression_module()
    log_phase("helpers_loaded")
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
    log_phase("configured", active=active.name, other=other.name)

    before_active = _snapshot(active)
    before_other = _snapshot(other)
    log_phase("snapshots_before_done")
    rest_triangles_active = xpbd_core.triangulated_faces(active.data)
    rest_triangles_other = xpbd_core.triangulated_faces(other.data)
    log_phase(
        "rest_triangulated",
        active_triangles=int(len(rest_triangles_active)),
        other_triangles=int(len(rest_triangles_other)),
    )
    if bool(args.check_intersections):
        initial_active = _check_intersections_limited(
            cs2,
            active,
            rest_triangles_active,
            int(args.max_report),
            pair_test_limit=int(args.intersection_pair_limit),
            budget_ms=float(args.intersection_budget_ms),
        )
        initial_other = _check_intersections_limited(
            cs2,
            other,
            rest_triangles_other,
            int(args.max_report),
            pair_test_limit=int(args.intersection_pair_limit),
            budget_ms=float(args.intersection_budget_ms),
        )
    else:
        initial_active = _skipped_intersection_check(len(rest_triangles_active))
        initial_other = _skipped_intersection_check(len(rest_triangles_other))
    active_options = settings_to_options(active.ssbl_cloth, runtime_mode_override="preview")
    other_options = settings_to_options(other.ssbl_cloth, runtime_mode_override="preview")

    started_at = time.perf_counter()
    log_phase("start_preview_start")
    session = ssbl.solver.start_preview(bpy.context, active)
    log_phase("start_preview_done", slots=len(session.slots), solve_order=list(session.solve_order))
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
        "dynamic_pair_cache_hits": 0,
        "dynamic_pair_cache_misses": 0,
        "dynamic_pair_cache_reused_triangles": 0,
        "dynamic_pair_cache_reused_particles": 0,
        "dynamic_collision_skipped_launches": 0,
        "self_collision_skipped_launches": 0,
        "self_candidate_count": 0,
    }
    cache_hits_total = 0
    cache_misses_total = 0
    pair_cache_hits_total = 0
    pair_cache_misses_total = 0
    finite = True
    for _index in range(steps):
        step_number = _index + 1
        log_phase("step_start", step=step_number)
        step_started = time.perf_counter()
        ssbl.solver.step_preview(bpy.context, active.name)
        step_wall_ms = (time.perf_counter() - step_started) * 1000.0
        step_times.append(step_wall_ms)
        log_phase("step_preview_done", step=step_number, wall_ms=float(step_wall_ms))
        diag = ssbl.solver.session_diagnostics(active)
        if heartbeat and (step_number == 1 or step_number % heartbeat_interval == 0 or step_number == steps):
            print(
                "SSBL_CS2_MULTICLOTH_PERF_STEP "
                + json.dumps(
                    {
                        "step": int(step_number),
                        "wall_ms": float(step_wall_ms),
                        "cuda_step_ms": float(getattr(diag, "cuda_step_call_ms", 0.0)),
                        "dynamic_collision_ms": float(getattr(diag, "dynamic_collision_ms", 0.0)),
                        "dynamic_particle_collision_ms": float(getattr(diag, "dynamic_particle_collision_ms", 0.0)),
                        "dynamic_particle_contacts": int(getattr(diag, "dynamic_particle_contacts", 0)),
                        "dynamic_particle_candidates": int(getattr(diag, "dynamic_particle_candidate_count", 0)),
                        "dynamic_triangle_count": int(getattr(diag, "dynamic_triangle_count", 0)),
                        "dynamic_pair_cache_hits": int(getattr(diag, "dynamic_pair_cache_hits", 0)),
                        "dynamic_pair_cache_misses": int(getattr(diag, "dynamic_pair_cache_misses", 0)),
                        "dynamic_skipped_launches": int(getattr(diag, "dynamic_collision_skipped_launches", 0)),
                        "exact_contacts": int(getattr(diag, "abi41_exact_impulse_contact_count", 0)),
                        "dynamic_upload_ms": float(getattr(diag, "dynamic_upload_ms", 0.0)),
                        "pcg_solve_ms": float(getattr(diag, "abi41_pcg_solve_ms", 0.0)),
                        "resolved_contacts": int(getattr(diag, "resolved_contacts", 0)),
                        "self_hash_ms": float(getattr(diag, "self_hash_ms", 0.0)),
                        "self_candidates": int(getattr(diag, "self_candidate_count", 0)),
                        "self_skipped_launches": int(getattr(diag, "self_collision_skipped_launches", 0)),
                        "self_solve_ms": float(getattr(diag, "self_solve_ms", 0.0)),
                        "soft_contacts": int(getattr(diag, "abi41_soft_contact_count", 0)),
                        "static_collision_ms": float(getattr(diag, "static_collision_ms", 0.0)),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        log_phase(
            "diag_done",
            step=step_number,
            finite=bool(getattr(diag, "finite", True)),
            cuda_step_ms=float(getattr(diag, "cuda_step_call_ms", 0.0)),
            dynamic_pack_ms=float(getattr(diag, "dynamic_collider_pack_ms", 0.0)),
            dynamic_upload_ms=float(getattr(diag, "dynamic_upload_ms", 0.0)),
            dynamic_triangles=int(getattr(diag, "dynamic_triangle_count", 0)),
            dynamic_particles=int(getattr(diag, "dynamic_particle_count", 0)),
            dynamic_pair_hits=int(getattr(diag, "dynamic_pair_cache_hits", 0)),
            dynamic_pair_misses=int(getattr(diag, "dynamic_pair_cache_misses", 0)),
            dynamic_skipped_launches=int(getattr(diag, "dynamic_collision_skipped_launches", 0)),
            dynamic_particle_candidates=int(getattr(diag, "dynamic_particle_candidate_count", 0)),
            dynamic_particle_contacts=int(getattr(diag, "dynamic_particle_contacts", 0)),
            candidate_count=int(getattr(diag, "candidate_count", 0)),
            resolved_contacts=int(getattr(diag, "resolved_contacts", 0)),
            self_candidates=int(getattr(diag, "self_candidate_count", 0)),
            self_skipped_launches=int(getattr(diag, "self_collision_skipped_launches", 0)),
            self_hash_ms=float(getattr(diag, "self_hash_ms", 0.0)),
            self_solve_ms=float(getattr(diag, "self_solve_ms", 0.0)),
            pcg_solve_ms=float(getattr(diag, "abi41_pcg_solve_ms", 0.0)),
            pcg_guarded=int(getattr(diag, "abi41_pcg_guarded", 0)),
            pcg_initial_residual=float(getattr(diag, "abi41_pcg_initial_residual", 0.0)),
            pcg_final_residual=float(getattr(diag, "abi41_pcg_final_residual", 0.0)),
            pcg_max_delta=float(getattr(diag, "abi41_pcg_max_delta", 0.0)),
        )
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
            "dynamic_pair_cache_hits",
            "dynamic_pair_cache_misses",
            "dynamic_pair_cache_reused_triangles",
            "dynamic_pair_cache_reused_particles",
            "dynamic_collision_skipped_launches",
            "self_collision_skipped_launches",
            "self_candidate_count",
        ):
            max_diag[key] = max(int(max_diag[key]), int(getattr(diag, key, 0)))
        cache_hits_total += int(getattr(diag, "dynamic_collider_cache_hits", 0))
        cache_misses_total += int(getattr(diag, "dynamic_collider_cache_misses", 0))
        pair_cache_hits_total += int(getattr(diag, "dynamic_pair_cache_hits", 0))
        pair_cache_misses_total += int(getattr(diag, "dynamic_pair_cache_misses", 0))
        log_phase("step_done", step=step_number, wall_ms=float(step_wall_ms))

    elapsed = time.perf_counter() - started_at
    final_diag = ssbl.solver.session_diagnostics(active)
    log_phase("request_stop_start")
    stopped = ssbl.solver.request_stop(active)
    log_phase("request_stop_done", stopped=bool(stopped))
    after_active = _snapshot(active)
    after_other = _snapshot(other)
    log_phase("snapshots_after_done")
    if bool(args.check_intersections):
        final_triangles_active = xpbd_core.triangulated_faces(active.data)
        final_triangles_other = xpbd_core.triangulated_faces(other.data)
        log_phase(
            "final_triangulated",
            active_triangles=int(len(final_triangles_active)),
            other_triangles=int(len(final_triangles_other)),
        )
        final_active = _check_intersections_limited(
            cs2,
            active,
            final_triangles_active,
            int(args.max_report),
            pair_test_limit=int(args.intersection_pair_limit),
            budget_ms=float(args.intersection_budget_ms),
        )
        final_other = _check_intersections_limited(
            cs2,
            other,
            final_triangles_other,
            int(args.max_report),
            pair_test_limit=int(args.intersection_pair_limit),
            budget_ms=float(args.intersection_budget_ms),
        )
    else:
        final_active = _skipped_intersection_check(len(rest_triangles_active))
        final_other = _skipped_intersection_check(len(rest_triangles_other))
    log_phase("intersection_summary_done", checked=bool(args.check_intersections))
    restore_active = _max_delta(before_active, after_active)
    restore_other = _max_delta(before_other, after_other)
    average_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    fps = (1000.0 / average_step_ms) if average_step_ms > 0.0 else 0.0
    active_finite_after = _finite_mesh(active)
    other_finite_after = _finite_mesh(other)
    correctness_pass = (
        bool(finite)
        and active_finite_after
        and other_finite_after
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
        "finite": bool(finite and active_finite_after and other_finite_after),
        "stopped": bool(stopped),
        "restore_delta_active": float(restore_active),
        "restore_delta_other": float(restore_other),
        "correctness_pass": bool(correctness_pass),
        "active_vertex_count": int(len(active.data.vertices)),
        "other_vertex_count": int(len(other.data.vertices)),
        "active_triangle_count": int(len(rest_triangles_active)),
        "other_triangle_count": int(len(rest_triangles_other)),
        "active_stretch_optimization_enabled": bool(active_options.stretch_optimization_enabled),
        "active_stretch_optimization_strength": float(active_options.stretch_optimization_strength),
        "other_stretch_optimization_enabled": bool(other_options.stretch_optimization_enabled),
        "other_stretch_optimization_strength": float(other_options.stretch_optimization_strength),
        "cache_hits_total": int(cache_hits_total),
        "cache_misses_total": int(cache_misses_total),
        "pair_cache_hits_total": int(pair_cache_hits_total),
        "pair_cache_misses_total": int(pair_cache_misses_total),
        "max_dynamic_upload_ms": float(max_diag["dynamic_upload_ms"]),
        "max_dynamic_collider_pack_ms": float(max_diag["dynamic_collider_pack_ms"]),
        "max_native_cuda_step_call_ms": float(max_diag["native_cuda_step_call_ms"]),
        "max_dynamic_triangle_upload_ms": float(max_diag["dynamic_triangle_upload_ms"]),
        "max_dynamic_particle_upload_ms": float(max_diag["dynamic_particle_upload_ms"]),
        "max_dynamic_triangle_count": int(max_diag["dynamic_triangle_count"]),
        "max_dynamic_particle_count": int(max_diag["dynamic_particle_count"]),
        "max_dynamic_particle_contacts": int(max_diag["dynamic_particle_contacts"]),
        "max_dynamic_particle_overflow": int(max_diag["dynamic_particle_overflow"]),
        "max_dynamic_pair_cache_hits": int(max_diag["dynamic_pair_cache_hits"]),
        "max_dynamic_pair_cache_misses": int(max_diag["dynamic_pair_cache_misses"]),
        "max_dynamic_pair_cache_reused_triangles": int(max_diag["dynamic_pair_cache_reused_triangles"]),
        "max_dynamic_pair_cache_reused_particles": int(max_diag["dynamic_pair_cache_reused_particles"]),
        "max_dynamic_collision_skipped_launches": int(max_diag["dynamic_collision_skipped_launches"]),
        "max_self_collision_skipped_launches": int(max_diag["self_collision_skipped_launches"]),
        "max_self_candidate_count": int(max_diag["self_candidate_count"]),
        "initial_active": initial_active,
        "initial_other": initial_other,
        "final_active": final_active,
        "final_other": final_other,
        **_diag_summary(final_diag),
    }
    log_phase("summary_ready", fps=float(fps), average_wall_step_ms=float(average_step_ms))
    print("SSBL_CS2_MULTICLOTH_PERF_PROBE", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not summary["finite"] or restore_active > 1.0e-7 or restore_other > 1.0e-7:
        raise RuntimeError(f"CS2 multicloth perf probe failed stability gate: {summary}")
    if float(args.fail_under_fps) > 0.0 and fps < float(args.fail_under_fps):
        raise RuntimeError(f"CS2 multicloth perf probe below FPS gate {args.fail_under_fps}: {fps:.2f}")


if __name__ == "__main__":
    main()
