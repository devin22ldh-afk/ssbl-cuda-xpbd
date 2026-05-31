import math
import os
import sys
import time

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from ssbl import solver


def _clear_scene():
    for existing in list(bpy.context.scene.objects):
        bpy.data.objects.remove(existing, do_unlink=True)


def _make_grid(name):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=101, y_subdivisions=101, size=2.4, location=(0, 0, 1.4))
    obj = bpy.context.object
    obj.name = name
    pin = obj.vertex_groups.new(name="ssbl_pin")
    pin.add([v.index for v in obj.data.vertices if v.co.y > 1.05], 1.0, "ADD")
    return obj


def _configure(settings, mode, hardness):
    settings.pin_vertex_group = "ssbl_pin"
    settings.hardness = float(hardness)
    settings.hardness_initialized = True
    settings.self_collision = False
    settings.self_collision_mode = mode
    settings.self_collision_interval = 2 if mode == "fast" else 1
    settings.max_self_collision_neighbors = 32
    settings.use_ground = True
    settings.ground_height = 0.0
    settings.collision_margin = 0.012
    settings.substeps = 4
    settings.iterations = 1
    settings.frame_count = 120


def _assert_finite(obj, label):
    values = [component for vertex in obj.data.vertices for component in (vertex.co.x, vertex.co.y, vertex.co.z)]
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError(f"{label} produced non-finite vertex coordinates")


def run_case(label, mode, hardness, steps):
    _clear_scene()
    obj = _make_grid(f"SSBL_10K_{label}")
    settings = bpy.context.scene.ssbl_preview
    _configure(settings, mode, hardness)

    session = solver.start_preview(bpy.context, obj)
    start = time.perf_counter()
    min_z = 999.0
    free_sum_z = 0.0
    free_count = 0
    worst_gap = None
    peak_penetration = 0.0
    peak_candidates = 0
    peak_resolved = 0
    peak_ccd = 0
    peak_recovery = 0
    peak_retry = 0
    for _ in range(steps):
        solver.step_preview(bpy.context, obj.name)
        _assert_finite(obj, label)
        min_z = min(min_z, min((obj.matrix_world @ v.co).z for v in obj.data.vertices))
        diag = solver.session_diagnostics(obj)
        if diag.min_gap is not None:
            worst_gap = diag.min_gap if worst_gap is None else min(worst_gap, diag.min_gap)
        peak_penetration = max(peak_penetration, float(diag.penetration_depth))
        peak_candidates = max(peak_candidates, int(diag.candidate_count))
        peak_resolved = max(peak_resolved, int(diag.resolved_contacts))
        peak_ccd = max(peak_ccd, int(diag.ccd_clamp_count))
        peak_recovery = max(peak_recovery, int(diag.recovery_passes))
        peak_retry = max(peak_retry, int(diag.local_retry_count))
    for vertex in obj.data.vertices:
        if vertex.co.y <= 1.05:
            free_sum_z += float((obj.matrix_world @ vertex.co).z)
            free_count += 1
    elapsed = max(time.perf_counter() - start, 1.0e-6)
    fps = steps / elapsed
    free_mean_z = free_sum_z / max(free_count, 1)
    diag = solver.session_diagnostics(obj)
    solver.request_stop(obj)

    print(
        "SSBL_10K_BENCH",
        f"label={label}",
        f"mode={mode}",
        f"hardness={hardness:.2f}",
        f"polys={len(obj.data.polygons)}",
        f"tris={len(session.cloth.triangles)}",
        f"verts={len(session.cloth.positions_world)}",
        f"edges={len(session.cloth.edges)}",
        f"tethers={len(session.cloth.lra_edges)}",
        f"fps={fps:.2f}",
        f"step_ms={diag.step_ms:.2f}",
        f"hash_build_ms={diag.hash_build_ms:.2f}",
        f"peak_candidates={peak_candidates}",
        f"peak_resolved={peak_resolved}",
        f"peak_ccd={peak_ccd}",
        f"peak_penetration={peak_penetration:.5f}",
        f"peak_recovery={peak_recovery}",
        f"peak_retry={peak_retry}",
        f"worst_gap={(worst_gap if worst_gap is not None else float('nan')):.5f}",
        f"finite={int(diag.finite)}",
        f"min_z={min_z:.4f}",
        f"free_mean_z={free_mean_z:.4f}",
    )


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        steps = max(int(os.environ.get("SSBL_10K_STEPS", "24")), 1)
        for mode in ("off", "fast"):
            for hardness in (0.0, 0.49, 0.5, 0.7, 1.0):
                run_case(f"{mode}_h{hardness:.2f}", mode, hardness, steps)
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
