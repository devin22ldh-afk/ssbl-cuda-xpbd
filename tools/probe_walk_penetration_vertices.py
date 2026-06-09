from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
ADDONS_ROOT = TOOLS_DIR.parent.parent
for path in (str(ADDONS_ROOT), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ssbl
from record_walk_60_preview import (  # noqa: E402
    BETA_NAME,
    BLEND_PATH,
    DYNAMIC_COLLECTION_NAME,
    SKIRT_NAME,
    _assign_dynamic_collider_collection,
    _register_addon,
    _step_record_frame,
)


FRAME_COUNT = max(int(os.environ.get("SSBL_WALK_PENETRATION_FRAMES", "60")), 1)
WARMUP_FRAME_COUNT = max(int(os.environ.get("SSBL_WALK_PENETRATION_WARMUP_FRAMES", "1")), 0)


def _inside_vertices(points: np.ndarray, source) -> list[dict[str, float | int]]:
    vertices = np.asarray(source.current_positions_world, dtype=np.float64)
    triangles = np.asarray(source.triangle_indices, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or len(triangles) == 0:
        return []
    if np.min(triangles) < 0 or np.max(triangles) >= len(vertices):
        return []
    tree = BVHTree.FromPolygons(
        [Vector((float(x), float(y), float(z))) for x, y, z in vertices],
        [tuple(int(v) for v in tri) for tri in triangles],
        all_triangles=True,
    )
    rows: list[dict[str, float | int]] = []
    for index, row in enumerate(np.asarray(points, dtype=np.float64).reshape((-1, 3))):
        point = Vector((float(row[0]), float(row[1]), float(row[2])))
        nearest = tree.find_nearest(point)
        if nearest is None or nearest[0] is None or nearest[1] is None:
            continue
        location, normal, tri_index, _distance = nearest
        if normal.length <= 1.0e-9:
            continue
        signed = float((point - location).dot(normal.normalized()))
        if signed < 0.0:
            rows.append({"vertex": int(index), "depth": float(-signed), "triangle": int(tri_index)})
    rows.sort(key=lambda item: float(item["depth"]), reverse=True)
    return rows


def main() -> None:
    if not BLEND_PATH.exists():
        raise RuntimeError(f"Missing walk blend: {BLEND_PATH}")
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH), load_ui=False)
    _register_addon()
    ssbl.solver.cleanup_all_sessions()

    scene = bpy.context.scene
    scene.frame_end = max(int(scene.frame_end), int(scene.frame_start) + WARMUP_FRAME_COUNT + FRAME_COUNT + 1)
    scene.frame_set(int(scene.frame_start))
    skirt = bpy.data.objects.get(SKIRT_NAME)
    beta = bpy.data.objects.get(BETA_NAME)
    if skirt is None or skirt.type != "MESH":
        raise RuntimeError(f"Missing skirt mesh: {SKIRT_NAME}")
    if beta is None or beta.type != "MESH":
        raise RuntimeError(f"Missing dynamic collider mesh: {BETA_NAME}")
    _assign_dynamic_collider_collection(scene, beta)

    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        raise RuntimeError("walk timeline preview did not start")
    worst: dict[str, object] = {"depth": 0.0}
    try:
        for warmup_index in range(WARMUP_FRAME_COUNT):
            _step_record_frame(bpy.context, scene, session, int(scene.frame_start) + warmup_index + 1)
        for index in range(FRAME_COUNT):
            frame = int(scene.frame_start) + WARMUP_FRAME_COUNT + index + 1
            _step_record_frame(bpy.context, scene, session, frame)
            slot = session.slots.get(SKIRT_NAME)
            source = session.dynamic_collision_sources.get(BETA_NAME)
            if slot is None or source is None:
                raise RuntimeError("walk session lost skirt slot or Beta dynamic source")
            rows = _inside_vertices(np.asarray(slot.current_positions_world, dtype=np.float32), source)
            if rows and float(rows[0]["depth"]) > float(worst["depth"]):
                inv_mass = np.asarray(slot.cloth.inv_mass, dtype=np.float32).reshape((-1,))
                top = rows[:10]
                for item in top:
                    vertex = int(item["vertex"])
                    item["inv_mass"] = float(inv_mass[vertex]) if 0 <= vertex < len(inv_mass) else math.nan
                    item["pinned"] = bool(0 <= vertex < len(inv_mass) and inv_mass[vertex] <= 1.0e-8)
                worst = {
                    "frame": int(frame),
                    "depth": float(top[0]["depth"]),
                    "inside_count": int(len(rows)),
                    "top": top,
                }
    finally:
        ssbl.solver.stop_timeline_preview(scene)

    print("SSBL_WALK_PENETRATION_VERTICES", json.dumps(worst, ensure_ascii=False, sort_keys=True))
    if bpy.app.background:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
