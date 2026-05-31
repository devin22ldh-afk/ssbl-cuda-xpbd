import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
TOOLS_DIR = str(Path(ADDONS_ROOT) / "ssbl" / "tools")
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import ssbl
from record_hook_driven_wring_preview import (
    FRAME_END,
    OBJECT_NAME,
    OUT_DIR,
    _configure_scene,
)


WEIGHTS = np.asarray(
    [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    ],
    dtype=np.float32,
)


def _ensure_registered():
    if not hasattr(bpy.context.scene, "ssbl_preview"):
        ssbl.register()


def _sample_points(points, triangles):
    tri_points = points[triangles]
    samples = []
    sample_triangles = []
    sample_weights = []
    for weights in WEIGHTS:
        samples.append(np.sum(tri_points * weights.reshape(1, 3, 1), axis=1))
        sample_triangles.append(np.arange(len(triangles), dtype=np.int32))
        sample_weights.append(np.repeat(weights.reshape(1, 3), len(triangles), axis=0))
    return (
        np.ascontiguousarray(np.concatenate(samples, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(sample_triangles, axis=0), dtype=np.int32),
        np.ascontiguousarray(np.concatenate(sample_weights, axis=0), dtype=np.float32),
    )


def _vertex_to_sample_stats(positions, rest, triangles, thickness):
    samples, sample_triangles, _sample_weights = _sample_points(positions, triangles)
    rest_samples, _rest_sample_triangles, _rest_sample_weights = _sample_points(rest, triangles)
    tri_vertices = triangles[sample_triangles]
    close_threshold = max(float(thickness) * 2.0, 1.0e-5)
    min_per_vertex = np.full(len(positions), np.inf, dtype=np.float32)
    under_thickness = 0
    under_penetration = 0
    valid_count = 0
    chunk = 192
    for start in range(0, len(positions), chunk):
        end = min(start + chunk, len(positions))
        vertex_ids = np.arange(start, end, dtype=np.int32)
        delta = positions[start:end, None, :] - samples[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        rest_delta = rest[start:end, None, :] - rest_samples[None, :, :]
        rest_distances = np.linalg.norm(rest_delta, axis=2)
        same_triangle = (
            (vertex_ids[:, None] == tri_vertices[None, :, 0])
            | (vertex_ids[:, None] == tri_vertices[None, :, 1])
            | (vertex_ids[:, None] == tri_vertices[None, :, 2])
        )
        valid = (~same_triangle) & (rest_distances > close_threshold)
        if not np.any(valid):
            continue
        valid_count += int(np.count_nonzero(valid))
        valid_distances = np.where(valid, distances, np.inf)
        min_per_vertex[start:end] = np.min(valid_distances, axis=1)
        under_thickness += int(np.count_nonzero(valid & (distances < thickness)))
        under_penetration += int(np.count_nonzero(valid & (distances < thickness * 0.75)))
    finite_min = min_per_vertex[np.isfinite(min_per_vertex)]
    if len(finite_min) == 0:
        percentiles = [float("inf")] * 4
    else:
        percentiles = np.percentile(finite_min, [1, 5, 50, 95]).astype(float).tolist()
    return {
        "valid_vertex_sample_pairs": valid_count,
        "vertex_count": int(len(positions)),
        "sample_count": int(len(samples)),
        "thickness": float(thickness),
        "min_nonlocal_distance": float(np.min(finite_min)) if len(finite_min) else float("inf"),
        "p01_min_per_vertex": percentiles[0],
        "p05_min_per_vertex": percentiles[1],
        "p50_min_per_vertex": percentiles[2],
        "p95_min_per_vertex": percentiles[3],
        "pairs_below_thickness": under_thickness,
        "pairs_below_75pct_thickness": under_penetration,
        "vertices_below_thickness": int(np.count_nonzero(finite_min < thickness)) if len(finite_min) else 0,
        "vertices_below_75pct_thickness": int(np.count_nonzero(finite_min < thickness * 0.75)) if len(finite_min) else 0,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_registered()
    scene = bpy.context.scene
    obj = bpy.data.objects.get(OBJECT_NAME)
    if obj is None:
        raise RuntimeError(f"Missing object: {OBJECT_NAME}")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    _configure_scene(obj)

    session = ssbl.solver.start_preview(bpy.context, obj)
    started = time.perf_counter()
    try:
        finished_early_at = None
        for frame in range(2, FRAME_END + 1):
            if ssbl.solver.step_preview(bpy.context, obj.name):
                finished_early_at = frame
                break
        elapsed = time.perf_counter() - started
        slot = session.slots[obj.name]
        positions = np.asarray(slot.current_positions_world, dtype=np.float32)
        rest = np.asarray(slot.cloth.positions_world, dtype=np.float32)
        triangles = np.asarray(slot.cloth.triangles, dtype=np.int32)
        thickness = float(scene.ssbl_preview.cloth_thickness)
        stats = _vertex_to_sample_stats(positions, rest, triangles, thickness)
        result = {
            "blend": bpy.data.filepath,
            "object": obj.name,
            "steps_requested": int(FRAME_END - 1),
            "finished_early_at": finished_early_at,
            "elapsed_seconds": elapsed,
            "simulated_fps_no_render": float((FRAME_END - 1) / elapsed) if elapsed > 0.0 else 0.0,
            "finite": bool(np.isfinite(positions).all()),
            "self_collision_mode": str(scene.ssbl_preview.self_collision_mode),
            "self_collision_interval": int(scene.ssbl_preview.self_collision_interval),
            "max_self_collision_neighbors": int(scene.ssbl_preview.max_self_collision_neighbors),
            "use_volume_pressure": bool(scene.ssbl_preview.use_volume_pressure),
            "hardness": float(scene.ssbl_preview.hardness),
            "stats": stats,
        }
    finally:
        ssbl.solver.request_stop(obj)

    out_path = OUT_DIR / "penetration_diagnosis.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SSBL_WRING_SELF_PENETRATION", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
