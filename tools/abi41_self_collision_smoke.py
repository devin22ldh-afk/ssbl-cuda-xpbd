from __future__ import annotations

import json
import math
import sys

import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

from ssbl.native_backend import NativeXpbdSolver
from ssbl.xpbd_core import ClothBuildData, SolverOptions, SELF_COLLISION_FAST


def _distance(values: np.ndarray, a: int, b: int) -> float:
    delta = values[a] - values[b]
    return float(np.linalg.norm(delta))


def _edge_average_z(values: np.ndarray, a: int, b: int) -> float:
    return float((values[a, 2] + values[b, 2]) * 0.5)


def _make_options() -> SolverOptions:
    return SolverOptions(
        dt=1.0 / 60.0,
        damping=1.0,
        gravity=np.zeros(3, dtype=np.float32),
        stretch_compliance=0.0,
        bend_compliance=0.0,
        lra_compliance=0.0,
        collision_margin=0.0,
        use_ground=False,
        ground_height=0.0,
        use_wall=False,
        wall_origin=np.zeros(3, dtype=np.float32),
        wall_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        use_sphere=False,
        sphere_center=np.zeros(3, dtype=np.float32),
        sphere_radius=0.0,
        self_collision=True,
        self_collision_mode=SELF_COLLISION_FAST,
        cloth_thickness=0.10,
        self_collision_interval=1,
        max_self_collision_neighbors=32,
        fast_self_collision_passes=4,
        use_volume_pressure=False,
        volume_compliance=0.0,
        pressure_strength=0.0,
        volume_target_scale=1.0,
        volume_solve_interval=1,
        self_probe_interval=4,
        self_surface_pair_interval=4,
        self_sleep_enabled=False,
        self_sleep_still_frames=4,
        self_sleep_full_scan_interval=16,
        self_compaction_enabled=False,
        self_sleep_motion_scale=1.0,
        self_compaction_active_fraction_threshold=0.25,
        self_pair_compaction_enabled=False,
        jitter_stabilizer_enabled=False,
        contact_friction=0.0,
        contact_tangent_damping=0.0,
        contact_compliance=0.0,
    )


def _make_cloth(vertex_count: int) -> ClothBuildData:
    positions = np.zeros((vertex_count, 3), dtype=np.float32)
    positions[:, 0] = np.arange(vertex_count, dtype=np.float32)
    return ClothBuildData(
        positions_world=positions,
        inv_mass=np.ones(vertex_count, dtype=np.float32),
        triangles=np.empty((0, 3), dtype=np.int32),
        edges=np.empty((0, 2), dtype=np.int32),
        edge_rest_lengths=np.empty(0, dtype=np.float32),
        edge_color_offsets=np.empty(0, dtype=np.int32),
        bends=np.empty((0, 2), dtype=np.int32),
        bend_rest_lengths=np.empty(0, dtype=np.float32),
        bend_color_offsets=np.empty(0, dtype=np.int32),
        lra_edges=np.empty((0, 2), dtype=np.int32),
        lra_rest_lengths=np.empty(0, dtype=np.float32),
        lra_color_offsets=np.empty(0, dtype=np.int32),
        pin_indices=np.empty(0, dtype=np.int32),
        pin_targets_world=np.empty((0, 3), dtype=np.float32),
        matrix_world_inv=np.eye(4, dtype=np.float32),
        rest_volume=0.0,
    )


def _make_vertex_triangle_cloth(vertex_count: int, triangle_count: int) -> ClothBuildData:
    positions = np.zeros((vertex_count, 3), dtype=np.float32)
    inv_mass = np.ones(vertex_count, dtype=np.float32)
    positions[0] = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
    positions[1] = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    positions[2] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    inv_mass[:3] = 0.0
    for index in range(3, vertex_count):
        positions[index] = np.array(
            [20.0 + float(index), 20.0 + float(index % 7) * 0.25, float(index % 5) * 0.20],
            dtype=np.float32,
        )
    positions[vertex_count - 1] = np.array([5.0, 5.0, 1.0], dtype=np.float32)
    triangles = np.empty((triangle_count, 3), dtype=np.int32)
    triangles[0] = np.array([0, 1, 2], dtype=np.int32)
    for tri_index in range(1, triangle_count):
        base = 3 + tri_index * 3
        triangles[tri_index] = np.array([base, base + 1, base + 2], dtype=np.int32)
    return ClothBuildData(
        positions_world=positions,
        inv_mass=inv_mass,
        triangles=triangles,
        edges=np.empty((0, 2), dtype=np.int32),
        edge_rest_lengths=np.empty(0, dtype=np.float32),
        edge_color_offsets=np.empty(0, dtype=np.int32),
        bends=np.empty((0, 2), dtype=np.int32),
        bend_rest_lengths=np.empty(0, dtype=np.float32),
        bend_color_offsets=np.empty(0, dtype=np.int32),
        lra_edges=np.empty((0, 2), dtype=np.int32),
        lra_rest_lengths=np.empty(0, dtype=np.float32),
        lra_color_offsets=np.empty(0, dtype=np.int32),
        pin_indices=np.empty(0, dtype=np.int32),
        pin_targets_world=np.empty((0, 3), dtype=np.float32),
        matrix_world_inv=np.eye(4, dtype=np.float32),
        rest_volume=0.0,
    )


def _make_edge_edge_cloth(vertex_count: int, edge_count: int) -> ClothBuildData:
    positions = np.zeros((vertex_count, 3), dtype=np.float32)
    inv_mass = np.ones(vertex_count, dtype=np.float32)
    for index in range(vertex_count):
        positions[index] = np.array(
            [200.0 + float(index) * 3.0, 100.0 + float(index % 11), 0.0],
            dtype=np.float32,
        )

    positions[0] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    positions[1] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    positions[2] = np.array([5.0, -1.0, 0.0], dtype=np.float32)
    positions[3] = np.array([5.0, 1.0, 0.0], dtype=np.float32)
    inv_mass[:2] = 0.0

    edges = np.empty((edge_count, 2), dtype=np.int32)
    edges[0] = np.array([0, 1], dtype=np.int32)
    edges[1] = np.array([2, 3], dtype=np.int32)
    for edge_index in range(2, edge_count):
        a = 4 + (edge_index - 2) * 2
        b = a + 1
        positions[a] = np.array([50.0 + float(edge_index) * 4.0, 30.0, 0.0], dtype=np.float32)
        positions[b] = np.array([51.0 + float(edge_index) * 4.0, 30.0, 0.0], dtype=np.float32)
        edges[edge_index] = np.array([a, b], dtype=np.int32)
    rest_lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1).astype(np.float32)

    return ClothBuildData(
        positions_world=positions,
        inv_mass=inv_mass,
        triangles=np.empty((0, 3), dtype=np.int32),
        edges=edges,
        edge_rest_lengths=rest_lengths,
        edge_color_offsets=np.empty(0, dtype=np.int32),
        bends=np.empty((0, 2), dtype=np.int32),
        bend_rest_lengths=np.empty(0, dtype=np.float32),
        bend_color_offsets=np.empty(0, dtype=np.int32),
        lra_edges=np.empty((0, 2), dtype=np.int32),
        lra_rest_lengths=np.empty(0, dtype=np.float32),
        lra_color_offsets=np.empty(0, dtype=np.int32),
        pin_indices=np.empty(0, dtype=np.int32),
        pin_targets_world=np.empty((0, 3), dtype=np.float32),
        matrix_world_inv=np.eye(4, dtype=np.float32),
        rest_volume=0.0,
    )


def _run_vertex_vertex_smoke() -> None:
    vertex_count = 600
    pair_vertex = vertex_count - 1
    cloth = _make_cloth(vertex_count)
    solver = NativeXpbdSolver(cloth, _make_options(), np.empty((0, 3, 3), dtype=np.float32))
    try:
        close_positions = np.array(cloth.positions_world, copy=True)
        close_positions[pair_vertex] = np.array([0.03, 0.0, 0.0], dtype=np.float32)
        initial_gap = _distance(close_positions, 0, pair_vertex)
        solver.update_positions(close_positions)
        solver.step(1, 6, diagnostics=True, synchronize=True)
        out = solver.download_positions()
        diag = solver.cached_diagnostics()
    finally:
        solver.close()

    final_gap = _distance(out, 0, pair_vertex)
    finite = bool(np.isfinite(out).all() and diag.finite)
    result = {
        "vertices": vertex_count,
        "finite": finite,
        "initial_gap": initial_gap,
        "final_gap": final_gap,
        "candidate_count": int(diag.candidate_count),
        "resolved_contacts": int(diag.resolved_contacts),
        "abi41_soft_contact_count": int(diag.abi41_soft_contact_count),
        "fast_soft_repulsion_candidates": int(diag.fast_soft_repulsion_candidates),
    }
    if not finite:
        raise RuntimeError(f"non-finite ABI37 self collision output: {json.dumps(result)}")
    if diag.abi41_soft_contact_count <= 0 or diag.fast_soft_repulsion_candidates <= 0:
        raise RuntimeError(f"ABI37 self collision did not report contacts: {json.dumps(result)}")
    if not final_gap > initial_gap + 1.0e-4:
        raise RuntimeError(f"ABI37 self collision did not separate vertices: {json.dumps(result)}")
    if not all(math.isfinite(float(value)) for value in (initial_gap, final_gap)):
        raise RuntimeError(f"invalid gap values: {json.dumps(result)}")
    print("SSBL_ABI41_SELF_COLLISION_SMOKE", json.dumps(result, sort_keys=True))


def _run_vertex_triangle_smoke() -> None:
    vertex_count = 600
    triangle_count = 64
    probe_vertex = vertex_count - 1
    cloth = _make_vertex_triangle_cloth(vertex_count, triangle_count)
    solver = NativeXpbdSolver(cloth, _make_options(), np.empty((0, 3, 3), dtype=np.float32))
    try:
        close_positions = np.array(cloth.positions_world, copy=True)
        close_positions[probe_vertex] = np.array([0.0, -0.2, 0.03], dtype=np.float32)
        initial_height = float(close_positions[probe_vertex, 2])
        solver.update_positions(close_positions)
        solver.step(1, 6, diagnostics=True, synchronize=True)
        out = solver.download_positions()
        diag = solver.cached_diagnostics()
    finally:
        solver.close()

    final_height = float(out[probe_vertex, 2])
    finite = bool(np.isfinite(out).all() and diag.finite)
    result = {
        "vertices": vertex_count,
        "triangles": triangle_count,
        "finite": finite,
        "initial_height": initial_height,
        "final_height": final_height,
        "candidate_count": int(diag.candidate_count),
        "resolved_contacts": int(diag.resolved_contacts),
        "abi41_soft_contact_count": int(diag.abi41_soft_contact_count),
        "fast_soft_repulsion_candidates": int(diag.fast_soft_repulsion_candidates),
    }
    if not finite:
        raise RuntimeError(f"non-finite ABI37 self VT output: {json.dumps(result)}")
    if diag.abi41_soft_contact_count <= 0 or diag.fast_soft_repulsion_candidates <= 0:
        raise RuntimeError(f"ABI37 self VT did not report contacts: {json.dumps(result)}")
    if not final_height > initial_height + 1.0e-4:
        raise RuntimeError(f"ABI37 self VT did not separate from triangle: {json.dumps(result)}")
    if not all(math.isfinite(float(value)) for value in (initial_height, final_height)):
        raise RuntimeError(f"invalid height values: {json.dumps(result)}")
    print("SSBL_ABI41_SELF_VT_SMOKE", json.dumps(result, sort_keys=True))


def _run_edge_edge_smoke() -> None:
    vertex_count = 600
    edge_count = 64
    cloth = _make_edge_edge_cloth(vertex_count, edge_count)
    solver = NativeXpbdSolver(cloth, _make_options(), np.empty((0, 3, 3), dtype=np.float32))
    try:
        close_positions = np.array(cloth.positions_world, copy=True)
        close_positions[2] = np.array([0.0, -1.0, 0.03], dtype=np.float32)
        close_positions[3] = np.array([0.0, 1.0, 0.03], dtype=np.float32)
        initial_height = _edge_average_z(close_positions, 2, 3)
        solver.update_positions(close_positions)
        solver.step(1, 6, diagnostics=True, synchronize=True)
        out = solver.download_positions()
        diag = solver.cached_diagnostics()
    finally:
        solver.close()

    final_height = _edge_average_z(out, 2, 3)
    finite = bool(np.isfinite(out).all() and diag.finite)
    result = {
        "vertices": vertex_count,
        "edges": edge_count,
        "finite": finite,
        "initial_height": initial_height,
        "final_height": final_height,
        "candidate_count": int(diag.candidate_count),
        "resolved_contacts": int(diag.resolved_contacts),
        "abi41_soft_contact_count": int(diag.abi41_soft_contact_count),
        "abi41_edge_edge_contact_count": int(diag.abi41_edge_edge_contact_count),
        "fast_soft_repulsion_candidates": int(diag.fast_soft_repulsion_candidates),
    }
    if not finite:
        raise RuntimeError(f"non-finite ABI37 self EE output: {json.dumps(result)}")
    if diag.abi41_edge_edge_contact_count <= 0 or diag.resolved_contacts <= 0:
        raise RuntimeError(f"ABI37 self EE did not report contacts: {json.dumps(result)}")
    if not final_height > initial_height + 1.0e-4:
        raise RuntimeError(f"ABI37 self EE did not separate crossing edges: {json.dumps(result)}")
    if not all(math.isfinite(float(value)) for value in (initial_height, final_height)):
        raise RuntimeError(f"invalid edge height values: {json.dumps(result)}")
    print("SSBL_ABI41_SELF_EE_SMOKE", json.dumps(result, sort_keys=True))


def main() -> None:
    _run_vertex_vertex_smoke()
    _run_vertex_triangle_smoke()
    _run_edge_edge_smoke()


if __name__ == "__main__":
    main()
