from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


ADDONS_ROOT = Path(r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons")
if str(ADDONS_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDONS_ROOT))

from ssbl.native_backend import NativeXpbdSolver  # noqa: E402
from ssbl.xpbd_core import ClothBuildData, SolverOptions, SELF_COLLISION_FAST  # noqa: E402


def _empty_i32(shape):
    return np.empty(shape, dtype=np.int32)


def _empty_f32(shape):
    return np.empty(shape, dtype=np.float32)


def _options(**overrides) -> SolverOptions:
    values = dict(
        dt=1.0 / 60.0,
        damping=1.0,
        gravity=np.zeros(3, dtype=np.float32),
        stretch_compliance=0.0,
        stretch_optimization_enabled=False,
        stretch_optimization_strength=0.0,
        bend_compliance=0.0,
        lra_compliance=0.0,
        collision_margin=0.05,
        use_ground=False,
        ground_height=0.0,
        use_wall=False,
        wall_origin=np.zeros(3, dtype=np.float32),
        wall_normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        use_sphere=False,
        sphere_center=np.zeros(3, dtype=np.float32),
        sphere_radius=0.0,
        self_collision=False,
        self_collision_mode=SELF_COLLISION_FAST,
        cloth_thickness=0.04,
        self_collision_distance=0.0,
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
        jitter_stabilizer_enabled=False,
        contact_friction=4.0,
        contact_tangent_damping=1.0,
        contact_compliance=0.0,
        static_sdf_voxel_size=0.02,
        static_sdf_band_voxels=3,
        static_sdf_max_resolution=64,
    )
    values.update(overrides)
    return SolverOptions(**values)


def _cloth(
    positions,
    *,
    inv_mass=None,
    triangles=None,
    edges=None,
    edge_rest_lengths=None,
) -> ClothBuildData:
    pos = np.ascontiguousarray(positions, dtype=np.float32).reshape((-1, 3))
    vertex_count = len(pos)
    inv = np.ones(vertex_count, dtype=np.float32) if inv_mass is None else np.ascontiguousarray(inv_mass, dtype=np.float32)
    tri = _empty_i32((0, 3)) if triangles is None else np.ascontiguousarray(triangles, dtype=np.int32).reshape((-1, 3))
    edg = _empty_i32((0, 2)) if edges is None else np.ascontiguousarray(edges, dtype=np.int32).reshape((-1, 2))
    rests = _empty_f32(0) if edge_rest_lengths is None else np.ascontiguousarray(edge_rest_lengths, dtype=np.float32).reshape((-1,))
    return ClothBuildData(
        positions_world=pos,
        inv_mass=inv,
        triangles=tri,
        edges=edg,
        edge_rest_lengths=rests,
        edge_color_offsets=_empty_i32(0),
        bends=_empty_i32((0, 2)),
        bend_rest_lengths=_empty_f32(0),
        bend_color_offsets=_empty_i32(0),
        lra_edges=_empty_i32((0, 2)),
        lra_rest_lengths=_empty_f32(0),
        lra_color_offsets=_empty_i32(0),
        pin_indices=_empty_i32(0),
        pin_targets_world=_empty_f32((0, 3)),
        matrix_world_inv=np.eye(4, dtype=np.float32),
        rest_volume=0.0,
    )


def _run_solver(cloth: ClothBuildData, options: SolverOptions, static_triangles=None, *, steps=24, substeps=4, iterations=1):
    static = _empty_f32((0, 3, 3)) if static_triangles is None else np.ascontiguousarray(static_triangles, dtype=np.float32).reshape((-1, 3, 3))
    solver = NativeXpbdSolver(cloth, options, static)
    samples = []
    try:
        for _ in range(steps):
            started = time.perf_counter()
            solver.step(substeps, iterations, diagnostics=True, synchronize=True)
            samples.append((time.perf_counter() - started) * 1000.0)
        out = np.array(solver.download_positions(), copy=True)
        diag = solver.cached_diagnostics()
    finally:
        solver.close()
    return out, diag, samples


def _analytic_case(kind: str) -> dict[str, object]:
    if kind == "ground":
        pos = [[0.0, 0.0, 0.02]]
        opts = _options(use_ground=True, gravity=np.array([6.0, 0.0, -6.0], dtype=np.float32))
        penetration_fn = lambda out: max(0.0, 0.05 - float(out[0, 2]))
        tangent_fn = lambda out: abs(float(out[0, 0]))
    elif kind == "corner":
        pos = [[0.02, 0.0, 0.02]]
        opts = _options(
            use_ground=True,
            use_wall=True,
            wall_origin=np.zeros(3, dtype=np.float32),
            wall_normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            gravity=np.array([-6.0, 6.0, -6.0], dtype=np.float32),
        )
        penetration_fn = lambda out: max(0.0, max(0.05 - float(out[0, 0]), 0.05 - float(out[0, 2])))
        tangent_fn = lambda out: abs(float(out[0, 1]))
    elif kind == "wall":
        pos = [[0.02, 0.0, 0.0]]
        opts = _options(use_wall=True, gravity=np.array([-6.0, 6.0, 0.0], dtype=np.float32))
        penetration_fn = lambda out: max(0.0, 0.05 - float(out[0, 0]))
        tangent_fn = lambda out: abs(float(out[0, 1]))
    elif kind == "sphere":
        pos = [[0.30, 0.0, 0.0]]
        opts = _options(
            use_sphere=True,
            sphere_center=np.zeros(3, dtype=np.float32),
            sphere_radius=0.30,
            gravity=np.array([0.0, 6.0, -1.0], dtype=np.float32),
        )
        penetration_fn = lambda out: max(0.0, 0.35 - float(np.linalg.norm(out[0])))
        tangent_fn = lambda out: abs(float(out[0, 1]))
    else:
        raise ValueError(kind)
    out, diag, samples = _run_solver(_cloth(pos), opts, steps=28, substeps=4, iterations=1)
    return {
        "case": f"analytic_{kind}",
        "finite": bool(np.isfinite(out).all() and diag.finite),
        "penetration": float(penetration_fn(out)),
        "tangent_abs": float(tangent_fn(out)),
        "avg_step_ms": float(sum(samples) / max(len(samples), 1)),
        "analytic_collision_ms": float(diag.analytic_collision_ms),
        "external_friction_corrections": int(diag.external_friction_corrections),
        "position": out.tolist(),
    }


def _static_sdf_case() -> dict[str, object]:
    static = np.array(
        [
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    opts = _options(gravity=np.array([6.0, 0.0, -6.0], dtype=np.float32))
    out, diag, samples = _run_solver(_cloth([[0.0, 0.0, 0.02]]), opts, static, steps=28, substeps=4, iterations=1)
    return {
        "case": "static_sdf_plane",
        "finite": bool(np.isfinite(out).all() and diag.finite),
        "penetration": max(0.0, 0.05 - float(out[0, 2])),
        "tangent_abs": abs(float(out[0, 0])),
        "avg_step_ms": float(sum(samples) / max(len(samples), 1)),
        "static_collision_ms": float(diag.static_collision_ms),
        "static_sdf_contact_count": int(diag.static_sdf_contact_count),
        "external_friction_corrections": int(diag.external_friction_corrections),
        "position": out.tolist(),
    }


def _self_vv_case() -> dict[str, object]:
    vertex_count = 600
    positions = np.zeros((vertex_count, 3), dtype=np.float32)
    positions[:, 0] = np.arange(vertex_count, dtype=np.float32)
    positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    inv = np.ones(vertex_count, dtype=np.float32)
    opts = _options(
        self_collision=True,
        cloth_thickness=0.10,
        collision_margin=0.0,
        contact_friction=0.0,
        contact_tangent_damping=0.0,
    )
    close_positions = np.array(positions, copy=True)
    close_positions[-1] = np.array([0.03, 0.0, 0.0], dtype=np.float32)
    initial_gap = float(np.linalg.norm(close_positions[-1] - close_positions[0]))
    solver = NativeXpbdSolver(_cloth(positions, inv_mass=inv), opts, _empty_f32((0, 3, 3)))
    samples = []
    try:
        solver.update_positions(close_positions)
        started = time.perf_counter()
        solver.step(1, 6, diagnostics=True, synchronize=True)
        samples.append((time.perf_counter() - started) * 1000.0)
        out = np.array(solver.download_positions(), copy=True)
        diag = solver.cached_diagnostics()
        started = time.perf_counter()
        solver.step(1, 6, diagnostics=True, synchronize=True)
        samples.append((time.perf_counter() - started) * 1000.0)
        out_followup = np.array(solver.download_positions(), copy=True)
        followup_diag = solver.cached_diagnostics()
    finally:
        solver.close()
    final_gap = float(np.linalg.norm(out[-1] - out[0]))
    followup_gap = float(np.linalg.norm(out_followup[-1] - out_followup[0]))
    return {
        "case": "self_vv",
        "finite": bool(np.isfinite(out).all() and np.isfinite(out_followup).all() and diag.finite and followup_diag.finite),
        "initial_gap": initial_gap,
        "final_gap": final_gap,
        "followup_gap": followup_gap,
        "followup_overshoot": max(0.0, followup_gap - final_gap),
        "separation_gain": final_gap - initial_gap,
        "avg_step_ms": float(sum(samples) / max(len(samples), 1)),
        "soft_contacts": int(diag.abi41_soft_contact_count),
        "exact_contacts": int(diag.abi41_exact_impulse_contact_count),
        "followup_exact_contacts": int(followup_diag.abi41_exact_impulse_contact_count),
        "max_smoothed_delta": float(diag.abi41_max_smoothed_delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["analytic_ground", "analytic_wall", "analytic_sphere", "analytic_corner", "static_sdf", "self_vv"],
        required=True,
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if args.case.startswith("analytic_"):
        result = _analytic_case(args.case.split("_", 1)[1])
    elif args.case == "static_sdf":
        result = _static_sdf_case()
    elif args.case == "self_vv":
        result = _self_vv_case()
    else:
        raise ValueError(args.case)
    if not result.get("finite"):
        raise RuntimeError(f"non-finite contact group result: {result}")
    print("SSBL_NATIVE_CONTACT_GROUP_PROBE", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
