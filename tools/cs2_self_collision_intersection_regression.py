from __future__ import annotations

import json
import math
import os
import sys
import time

import bpy
from mathutils.bvhtree import BVHTree


BLEND_PATH = os.environ.get("SSBL_CS2_BLEND", r"C:\Users\Administrator\Desktop\cs2.blend")
TARGET_OBJECT = os.environ.get("SSBL_CS2_TARGET", "Suzanne")
ADDONS_ROOT = os.environ.get(
    "SSBL_ADDONS_ROOT",
    r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons",
)
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _find_target() -> bpy.types.Object:
    obj = bpy.data.objects.get(TARGET_OBJECT)
    if obj is not None and obj.type == "MESH":
        return obj
    active = bpy.context.view_layer.objects.active
    if active is not None and active.type == "MESH":
        return active
    raise RuntimeError(f"Target mesh object not found: {TARGET_OBJECT}")


def _select_only(obj: bpy.types.Object) -> None:
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _ensure_pin_group(obj: bpy.types.Object, settings) -> int:
    group_name = str(getattr(settings, "pin_vertex_group", "") or "ssbl_pin")
    settings.pin_vertex_group = group_name
    group = obj.vertex_groups.get(group_name)
    if group is None:
        z_values = [vert.co.z for vert in obj.data.vertices]
        threshold = max(z_values) - (max(z_values) - min(z_values)) * 0.18
        indices = [vert.index for vert in obj.data.vertices if vert.co.z >= threshold]
        if not indices:
            indices = [max(obj.data.vertices, key=lambda vert: vert.co.z).index]
        group = obj.vertex_groups.new(name=group_name)
        group.add(indices, 1.0, "ADD")
        return len(indices)

    count = 0
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group == group.index and assignment.weight > 0.0:
                count += 1
                break
    return count


def _segment_triangle_intersects(p0, p1, a, b, c, eps: float = 1.0e-7) -> bool:
    direction = p1 - p0
    edge1 = b - a
    edge2 = c - a
    h = direction.cross(edge2)
    det = edge1.dot(h)
    if abs(det) < eps:
        return False
    inv_det = 1.0 / det
    s = p0 - a
    u = inv_det * s.dot(h)
    if u < eps or u > 1.0 - eps:
        return False
    q = s.cross(edge1)
    v = inv_det * direction.dot(q)
    if v < eps or u + v > 1.0 - eps:
        return False
    t = inv_det * edge2.dot(q)
    return eps < t < 1.0 - eps


def _triangles_intersect(vertices, tri_a, tri_b) -> bool:
    a0, a1, a2 = [vertices[index] for index in tri_a]
    b0, b1, b2 = [vertices[index] for index in tri_b]
    for p0, p1 in ((a0, a1), (a1, a2), (a2, a0)):
        if _segment_triangle_intersects(p0, p1, b0, b1, b2):
            return True
    for p0, p1 in ((b0, b1), (b1, b2), (b2, b0)):
        if _segment_triangle_intersects(p0, p1, a0, a1, a2):
            return True
    return False


def _neighbor_sets(triangles, vertex_count: int) -> list[set[int]]:
    neighbors = [set() for _index in range(vertex_count)]
    for tri in triangles:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def _triangles_are_solver_topology_neighbors(tri_a, tri_b, neighbors: list[set[int]]) -> bool:
    for a in tri_a:
        for b in tri_b:
            if a == b or b in neighbors[a]:
                return True
    return False


def _world_vertices(obj: bpy.types.Object):
    return [obj.matrix_world @ vert.co for vert in obj.data.vertices]


def _signed_volume(obj: bpy.types.Object, triangles) -> float:
    vertices = _world_vertices(obj)
    total = 0.0
    for ia, ib, ic in triangles:
        a = vertices[int(ia)]
        b = vertices[int(ib)]
        c = vertices[int(ic)]
        total += a.dot(b.cross(c)) / 6.0
    return float(total)


def _bbox(obj: bpy.types.Object) -> dict[str, list[float]]:
    vertices = _world_vertices(obj)
    mins = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maxs = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    return {
        "min": [float(value) for value in mins],
        "max": [float(value) for value in maxs],
        "size": [float(maxs[axis] - mins[axis]) for axis in range(3)],
    }


def _check_intersections(obj: bpy.types.Object, triangles, max_report: int) -> dict[str, object]:
    vertices = _world_vertices(obj)
    polygons = [tuple(int(vertex) for vertex in tri) for tri in triangles]
    neighbors = _neighbor_sets(triangles, len(vertices))
    tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True, epsilon=0.0)
    overlap_pairs = tree.overlap(tree)
    tested = 0
    intersections = []
    solver_relevant_tested = 0
    solver_relevant_intersections = []
    for ia, ib in overlap_pairs:
        if ia >= ib:
            continue
        tri_a = polygons[ia]
        tri_b = polygons[ib]
        if set(tri_a).intersection(tri_b):
            continue
        tested += 1
        solver_topology_neighbor = _triangles_are_solver_topology_neighbors(tri_a, tri_b, neighbors)
        if not solver_topology_neighbor:
            solver_relevant_tested += 1
        if _triangles_intersect(vertices, tri_a, tri_b):
            intersections.append((int(ia), int(ib), tri_a, tri_b))
            if not solver_topology_neighbor:
                solver_relevant_intersections.append((int(ia), int(ib), tri_a, tri_b))
            if len(intersections) >= max_report:
                break
    return {
        "overlap_pairs": int(len(overlap_pairs)),
        "tested_non_adjacent_pairs": int(tested),
        "intersection_count_capped": int(len(intersections)),
        "sample_intersections": intersections[:10],
        "solver_relevant_tested_pairs": int(solver_relevant_tested),
        "solver_relevant_intersection_count_capped": int(len(solver_relevant_intersections)),
        "solver_relevant_sample_intersections": solver_relevant_intersections[:10],
    }


def _apply_object_overrides(settings, steps: int) -> None:
    settings.enabled = True
    settings.frame_count = max(_int_env("SSBL_CS2_FRAME_COUNT", int(getattr(settings, "frame_count", steps + 1))), steps + 1)
    settings.self_collision = _bool_env("SSBL_CS2_SELF_COLLISION", True)
    settings.self_collision_mode = os.environ.get("SSBL_CS2_SELF_COLLISION_MODE", "fast")
    settings.self_collision_interval = _int_env(
        "SSBL_CS2_SELF_COLLISION_INTERVAL",
        int(getattr(settings, "self_collision_interval", 2)),
    )
    settings.max_self_collision_neighbors = _int_env(
        "SSBL_CS2_MAX_SELF_NEIGHBORS",
        int(getattr(settings, "max_self_collision_neighbors", 32)),
    )
    if hasattr(settings, "fast_self_collision_passes"):
        settings.fast_self_collision_passes = _int_env(
            "SSBL_CS2_FAST_SELF_COLLISION_PASSES",
            int(getattr(settings, "fast_self_collision_passes", 4)),
        )
    settings.collision_margin = _float_env("SSBL_CS2_COLLISION_MARGIN", float(getattr(settings, "collision_margin", 0.005)))
    settings.cloth_thickness = _float_env("SSBL_CS2_CLOTH_THICKNESS", float(getattr(settings, "cloth_thickness", 0.02)))
    settings.substeps = _int_env("SSBL_CS2_SUBSTEPS", int(getattr(settings, "substeps", 14)))
    settings.iterations = _int_env("SSBL_CS2_ITERATIONS", int(getattr(settings, "iterations", 2)))
    settings.use_volume_pressure = _bool_env(
        "SSBL_CS2_VOLUME_PRESSURE",
        bool(getattr(settings, "use_volume_pressure", False)),
    )
    settings.self_probe_interval = _int_env("SSBL_CS2_SELF_PROBE_INTERVAL", int(getattr(settings, "self_probe_interval", 8)))
    settings.self_surface_pair_interval = _int_env(
        "SSBL_CS2_SELF_SURFACE_PAIR_INTERVAL",
        int(getattr(settings, "self_surface_pair_interval", 8)),
    )


def main() -> None:
    if BLEND_PATH:
        bpy.ops.wm.open_mainfile(filepath=BLEND_PATH, load_ui=False)

    import ssbl
    from ssbl import xpbd_core

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    obj = _find_target()
    _select_only(obj)
    if not hasattr(obj, "ssbl_cloth"):
        raise RuntimeError(f"{obj.name} has no ssbl_cloth settings")
    settings = obj.ssbl_cloth
    steps = _int_env("SSBL_CS2_STEPS", 60)
    max_report = _int_env("SSBL_CS2_MAX_REPORT", 1000)
    _apply_object_overrides(settings, steps)
    pin_count = _ensure_pin_group(obj, settings)

    source_mesh = obj.data
    source_coords_before = [
        float(component)
        for vert in source_mesh.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    ]
    rest_triangles = xpbd_core.triangulated_faces(obj.data)
    rest_volume = _signed_volume(obj, rest_triangles)
    initial_collision = _check_intersections(obj, rest_triangles, max_report)

    started_at = time.perf_counter()
    ssbl.solver.start_preview(bpy.context, obj)
    step_times = []
    for _index in range(steps):
        step_start = time.perf_counter()
        ssbl.solver.step_preview(bpy.context, obj.name)
        step_times.append((time.perf_counter() - step_start) * 1000.0)

    triangles = xpbd_core.triangulated_faces(obj.data)
    coords_finite = all(
        math.isfinite(float(component))
        for vert in obj.data.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    )
    collision = _check_intersections(obj, triangles, max_report)
    diag = ssbl.solver.session_diagnostics(obj)
    elapsed = time.perf_counter() - started_at
    ssbl.solver.request_stop(obj)

    source_coords_after = [
        float(component)
        for vert in source_mesh.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    ]
    restore_delta = max(
        (abs(after - before) for before, after in zip(source_coords_before, source_coords_after)),
        default=0.0,
    )
    step_times_sorted = sorted(step_times)
    p95_index = min(max(int(len(step_times_sorted) * 0.95), 0), len(step_times_sorted) - 1) if step_times_sorted else 0
    average_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    summary = {
        "blend_file": bpy.data.filepath,
        "object": obj.name,
        "steps": int(steps),
        "elapsed_s": float(elapsed),
        "average_wall_step_ms": float(average_step_ms),
        "p95_wall_step_ms": float(step_times_sorted[p95_index] if step_times_sorted else 0.0),
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
        "rest_volume": float(rest_volume),
        "volume": float(_signed_volume(obj, triangles)),
        "bbox": _bbox(obj),
        "restore_delta": float(restore_delta),
        "native_step_ms": float(getattr(diag, "step_ms", 0.0)),
        "native_self_hash_ms": float(getattr(diag, "self_hash_ms", 0.0)),
        "native_self_solve_ms": float(getattr(diag, "self_solve_ms", 0.0)),
        "native_self_probe_ms": float(getattr(diag, "self_probe_ms", 0.0)),
        "native_self_recovery_ms": float(getattr(diag, "self_recovery_ms", 0.0)),
        "native_min_gap": getattr(diag, "min_gap", None),
        "native_max_penetration": float(max(0.0, -float(getattr(diag, "min_gap", 0.0) or 0.0))),
        "native_resolved_contacts": int(getattr(diag, "resolved_contacts", 0)),
        "native_candidate_count": int(getattr(diag, "candidate_count", 0)),
        "native_recovery_passes": int(getattr(diag, "recovery_passes", 0)),
        "native_local_retry_count": int(getattr(diag, "local_retry_count", 0)),
        "native_fast_exact_vt_candidates": int(getattr(diag, "fast_exact_vt_candidates", 0)),
        "native_fast_exact_vt_projected": int(getattr(diag, "fast_exact_vt_projected", 0)),
        "native_fast_exact_vt_guarded": int(getattr(diag, "fast_exact_vt_guarded", 0)),
        "native_fast_exact_vt_skipped_rest": int(getattr(diag, "fast_exact_vt_skipped_rest", 0)),
        "native_fast_soft_repulsion_candidates": int(getattr(diag, "fast_soft_repulsion_candidates", 0)),
        "native_fast_soft_repulsion_applied": int(getattr(diag, "fast_soft_repulsion_applied", 0)),
        "native_fast_soft_repulsion_max_push": float(getattr(diag, "fast_soft_repulsion_max_push", 0.0)),
        "native_fast_hard_projection_count": int(getattr(diag, "fast_hard_projection_count", 0)),
        "native_fast_manifold_contacts": int(getattr(diag, "fast_manifold_contacts", 0)),
        "native_fast_manifold_reused": int(getattr(diag, "fast_manifold_reused", 0)),
        "native_fast_barrier_projected": int(getattr(diag, "fast_barrier_projected", 0)),
        "native_fast_barrier_smoothed_vertices": int(getattr(diag, "fast_barrier_smoothed_vertices", 0)),
        "native_fast_barrier_overflow": int(getattr(diag, "fast_barrier_overflow", 0)),
        "native_fast_barrier_max_delta": float(getattr(diag, "fast_barrier_max_delta", 0.0)),
        "native_fast_edge_edge_candidates": int(getattr(diag, "fast_edge_edge_candidates", 0)),
        "native_fast_edge_edge_contacts": int(getattr(diag, "fast_edge_edge_contacts", 0)),
        "native_fast_triangle_pair_candidates": int(getattr(diag, "fast_triangle_pair_candidates", 0)),
        "native_fast_triangle_pair_contacts": int(getattr(diag, "fast_triangle_pair_contacts", 0)),
        "native_fast_triangle_pair_skipped_rest": int(getattr(diag, "fast_triangle_pair_skipped_rest", 0)),
        "native_fast_contact_classification_guarded": int(getattr(diag, "fast_contact_classification_guarded", 0)),
        "native_fast_region_cluster_candidates": int(getattr(diag, "fast_region_cluster_candidates", 0)),
        "native_fast_region_cluster_contacts": int(getattr(diag, "fast_region_cluster_contacts", 0)),
        "native_fast_region_cluster_guarded": int(getattr(diag, "fast_region_cluster_guarded", 0)),
        "native_fast_overlap_island_candidates": int(getattr(diag, "fast_overlap_island_candidates", 0)),
        "native_fast_overlap_island_clusters": int(getattr(diag, "fast_overlap_island_clusters", 0)),
        "native_fast_overlap_island_vertex_refs": int(getattr(diag, "fast_overlap_island_vertex_refs", 0)),
        "native_fast_overlap_island_applied_vertices": int(getattr(diag, "fast_overlap_island_applied_vertices", 0)),
        "native_fast_overlap_island_guarded": int(getattr(diag, "fast_overlap_island_guarded", 0)),
        "native_fast_overlap_island_max_delta": float(getattr(diag, "fast_overlap_island_max_delta", 0.0)),
        "native_fast_cc_overlap_components": int(getattr(diag, "fast_cc_overlap_components", 0)),
        "native_fast_cc_overlap_seed_triangles": int(getattr(diag, "fast_cc_overlap_seed_triangles", 0)),
        "native_fast_cc_overlap_owned_vertices": int(getattr(diag, "fast_cc_overlap_owned_vertices", 0)),
        "native_fast_cc_overlap_union_edges": int(getattr(diag, "fast_cc_overlap_union_edges", 0)),
        "native_fast_cc_overlap_guarded": int(getattr(diag, "fast_cc_overlap_guarded", 0)),
        "native_fast_cc_overlap_applied_vertices": int(getattr(diag, "fast_cc_overlap_applied_vertices", 0)),
        "native_fast_cc_overlap_max_delta": float(getattr(diag, "fast_cc_overlap_max_delta", 0.0)),
        "initial": initial_collision,
        **collision,
    }
    print("SSBL_CS2_SELF_INTERSECTION_REGRESSION " + json.dumps(summary, ensure_ascii=False, sort_keys=True))

    should_fail = _bool_env("SSBL_CS2_FAIL_ON_INTERSECTION", True)
    if should_fail:
        if not bool(summary["finite"]):
            raise RuntimeError("CS2 self-collision regression failed: non-finite vertices")
        if float(summary["restore_delta"]) > 1.0e-7:
            raise RuntimeError(f"CS2 self-collision regression failed: restore_delta={summary['restore_delta']}")
        fail_on_raw = _bool_env("SSBL_CS2_FAIL_ON_RAW_INTERSECTION", False)
        intersection_key = (
            "intersection_count_capped"
            if fail_on_raw
            else "solver_relevant_intersection_count_capped"
        )
        if int(summary[intersection_key]) > 0:
            raise RuntimeError(
                f"CS2 self-collision regression failed: {intersection_key}={summary[intersection_key]}"
            )


if __name__ == "__main__":
    main()
