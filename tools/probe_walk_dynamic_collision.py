from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


BLEND_PATH = Path(os.environ.get("SSBL_WALK_BLEND", r"C:\Users\Administrator\Desktop\演示视频\walk.blend"))
FRAME_COUNT = max(int(os.environ.get("SSBL_WALK_PROBE_FRAMES", "40")), 1)
ENABLE_RUNTIME_BETA_COLLIDER = os.environ.get("SSBL_WALK_ENABLE_RUNTIME_BETA_COLLIDER", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RUNTIME_BETA_NAME = os.environ.get("SSBL_WALK_BETA_OBJECT", "Beta_Surface")
EXPECTED_SLOT_NAME = os.environ.get("SSBL_WALK_EXPECTED_SLOT", "Codex_Skirt_Pleated")
RUNTIME_DYNAMIC_COLLECTION_NAME = os.environ.get(
    "SSBL_WALK_DYNAMIC_COLLIDER_COLLECTION",
    "SSBL_Runtime_Dynamic_Colliders",
)


def _register_addon() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()


def _enabled_cloth_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    return [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and hasattr(obj, "ssbl_cloth") and bool(obj.ssbl_cloth.enabled)
    ]


def _mesh_rows(scene: bpy.types.Scene) -> list[dict[str, object]]:
    rows = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        settings = getattr(obj, "ssbl_cloth", None)
        rows.append(
            {
                "name": obj.name,
                "vertices": int(len(obj.data.vertices)),
                "triangles": int(sum(max(len(poly.vertices) - 2, 0) for poly in obj.data.polygons)),
                "ssbl": bool(settings),
                "enabled": bool(getattr(settings, "enabled", False)) if settings else False,
                "use_evaluated_mesh": bool(getattr(settings, "use_evaluated_mesh", False)) if settings else False,
                "collision_layer": int(getattr(settings, "collision_layer", 1)) if settings else 1,
            }
        )
    return rows


def _finite_slot_positions(session) -> bool:
    for slot in session.slots.values():
        positions = np.asarray(slot.current_positions_world, dtype=np.float32)
        if positions.size and not bool(np.isfinite(positions).all()):
            return False
    return True


def _runtime_assign_dynamic_collider_collection(scene: bpy.types.Scene, obj: bpy.types.Object) -> bool:
    if obj is None or obj.type != "MESH":
        return False
    collection = bpy.data.collections.get(RUNTIME_DYNAMIC_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(RUNTIME_DYNAMIC_COLLECTION_NAME)
    if not any(child.name == collection.name for child in scene.collection.children):
        scene.collection.children.link(collection)
    if not any(item.name == obj.name for item in collection.objects):
        collection.objects.link(obj)
    assigned = False
    scene.ssbl_preview.dynamic_collider_collection = collection
    for cloth_obj in _enabled_cloth_objects(scene):
        if cloth_obj.name == obj.name:
            continue
        settings = getattr(cloth_obj, "ssbl_cloth", None)
        if settings is None:
            continue
        settings.dynamic_collider_collection = collection
        assigned = True
    obj["ssbl_enable_cross_cloth_collision"] = True
    return assigned


def main() -> None:
    if not BLEND_PATH.exists():
        raise RuntimeError(f"Missing walk blend: {BLEND_PATH}")

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH), load_ui=False)
    _register_addon()

    scene = bpy.context.scene
    scene.frame_set(int(scene.frame_start))
    ssbl.solver.cleanup_all_sessions()
    runtime_collider_enabled = False
    if ENABLE_RUNTIME_BETA_COLLIDER:
        runtime_collider_enabled = _runtime_assign_dynamic_collider_collection(
            scene,
            bpy.data.objects.get(RUNTIME_BETA_NAME),
        )
    enabled = _enabled_cloth_objects(scene)
    object_rows = _mesh_rows(scene)
    if not enabled:
        result = {"blend_path": str(BLEND_PATH), "objects": object_rows, "failures": ["no enabled SSBL cloth objects"]}
        print("SSBL_WALK_DYNAMIC_COLLISION_PROBE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        raise RuntimeError("walk.blend has no enabled SSBL cloth objects")

    started = time.perf_counter()
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        result = {"blend_path": str(BLEND_PATH), "objects": _mesh_rows(scene), "failures": ["timeline preview did not start"]}
        print("SSBL_WALK_DYNAMIC_COLLISION_PROBE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        raise RuntimeError("timeline preview did not start")

    finite = True
    max_dynamic_triangles = 0
    max_dynamic_particles = 0
    max_resolved_contacts = 0
    max_candidates = 0
    max_aabb_rejects = 0
    max_bucket_overflow = 0
    max_bucket_occupancy = 0
    max_large_primitives = 0
    max_dynamic_upload_ms = 0.0
    max_dynamic_triangle_upload_ms = 0.0
    max_dynamic_collision_ms = 0.0
    max_particle_collision_ms = 0.0
    max_frame_set_ms = 0.0
    max_input_refresh_ms = 0.0
    max_cuda_step_call_ms = 0.0
    max_download_ms = 0.0
    max_writeback_ms = 0.0
    frames_run = 0

    try:
        for frame_offset in range(FRAME_COUNT):
            if frame_offset > 0:
                scene.frame_set(int(scene.frame_start) + frame_offset)
                ssbl.solver.step_timeline_preview(bpy.context, scene)
            diag = session.last_diagnostics
            finite = finite and bool(getattr(diag, "finite", True)) and _finite_slot_positions(session)
            max_dynamic_triangles = max(max_dynamic_triangles, int(getattr(diag, "dynamic_triangle_count", 0)))
            max_dynamic_particles = max(max_dynamic_particles, int(getattr(diag, "dynamic_particle_count", 0)))
            max_resolved_contacts = max(max_resolved_contacts, int(getattr(diag, "resolved_contacts", 0)))
            max_candidates = max(max_candidates, int(getattr(diag, "dynamic_triangle_candidate_count", 0)))
            max_aabb_rejects = max(max_aabb_rejects, int(getattr(diag, "dynamic_triangle_aabb_reject_count", 0)))
            max_bucket_overflow = max(max_bucket_overflow, int(getattr(diag, "dynamic_triangle_bucket_overflow", 0)))
            max_bucket_occupancy = max(max_bucket_occupancy, int(getattr(diag, "dynamic_triangle_max_bucket_occupancy", 0)))
            max_large_primitives = max(max_large_primitives, int(getattr(diag, "dynamic_triangle_large_primitive_count", 0)))
            max_dynamic_upload_ms = max(max_dynamic_upload_ms, float(getattr(diag, "dynamic_upload_ms", 0.0)))
            max_dynamic_triangle_upload_ms = max(
                max_dynamic_triangle_upload_ms,
                float(getattr(diag, "dynamic_triangle_upload_ms", 0.0)),
            )
            max_dynamic_collision_ms = max(max_dynamic_collision_ms, float(getattr(diag, "dynamic_collision_ms", 0.0)))
            max_particle_collision_ms = max(max_particle_collision_ms, float(getattr(diag, "dynamic_particle_collision_ms", 0.0)))
            max_frame_set_ms = max(max_frame_set_ms, float(getattr(diag, "frame_set_ms", 0.0)))
            max_input_refresh_ms = max(max_input_refresh_ms, float(getattr(diag, "input_refresh_ms", 0.0)))
            max_cuda_step_call_ms = max(max_cuda_step_call_ms, float(getattr(diag, "cuda_step_call_ms", 0.0)))
            max_download_ms = max(max_download_ms, float(getattr(diag, "download_ms", 0.0)))
            max_writeback_ms = max(max_writeback_ms, float(getattr(diag, "writeback_ms", 0.0)))
            frames_run += 1
            if not finite:
                break
    finally:
        for obj in enabled:
            ssbl.solver.request_stop(obj)

    elapsed = time.perf_counter() - started
    result = {
        "blend_path": str(BLEND_PATH),
        "frames_run": int(frames_run),
        "elapsed_s": float(elapsed),
        "sim_fps_wall": float(frames_run / elapsed) if elapsed > 0.0 else 0.0,
        "slot_names": list(session.solve_order),
        "dynamic_collision_source_names": sorted(session.dynamic_collision_sources.keys()),
        "runtime_beta_collider_enabled": bool(runtime_collider_enabled),
        "objects": object_rows,
        "finite": bool(finite),
        "max_dynamic_triangle_count": int(max_dynamic_triangles),
        "max_dynamic_particle_count": int(max_dynamic_particles),
        "max_resolved_contacts": int(max_resolved_contacts),
        "max_dynamic_triangle_candidate_count": int(max_candidates),
        "max_dynamic_triangle_aabb_reject_count": int(max_aabb_rejects),
        "max_dynamic_triangle_bucket_overflow": int(max_bucket_overflow),
        "max_dynamic_triangle_max_bucket_occupancy": int(max_bucket_occupancy),
        "max_dynamic_triangle_large_primitive_count": int(max_large_primitives),
        "max_dynamic_upload_ms": float(max_dynamic_upload_ms),
        "max_dynamic_triangle_upload_ms": float(max_dynamic_triangle_upload_ms),
        "max_dynamic_collision_ms": float(max_dynamic_collision_ms),
        "max_dynamic_particle_collision_ms": float(max_particle_collision_ms),
        "max_frame_set_ms": float(max_frame_set_ms),
        "max_input_refresh_ms": float(max_input_refresh_ms),
        "max_cuda_step_call_ms": float(max_cuda_step_call_ms),
        "max_download_ms": float(max_download_ms),
        "max_writeback_ms": float(max_writeback_ms),
    }
    failures = []
    if not finite:
        failures.append("non-finite slot positions or diagnostics")
    if ENABLE_RUNTIME_BETA_COLLIDER:
        if not runtime_collider_enabled:
            failures.append(f"runtime dynamic collider collection was not assigned: {RUNTIME_BETA_NAME}")
        elif RUNTIME_BETA_NAME not in session.dynamic_collision_sources:
            failures.append(f"runtime dynamic collider did not enter session sources: {RUNTIME_BETA_NAME}")
        if RUNTIME_BETA_NAME in session.slots:
            failures.append(f"runtime dynamic collider incorrectly entered cloth slots: {RUNTIME_BETA_NAME}")
    if EXPECTED_SLOT_NAME and list(session.solve_order) != [EXPECTED_SLOT_NAME]:
        failures.append(f"unexpected simulated slots: {list(session.solve_order)}")
    if (session.dynamic_collision_sources or len(session.slots) > 1) and max_dynamic_triangles <= 0:
        failures.append("no dynamic triangles were uploaded despite dynamic collision sources")
    if max_bucket_overflow > 0:
        failures.append(f"dynamic triangle hash bucket overflow: {max_bucket_overflow}")
    if failures:
        result["failures"] = failures
        print("SSBL_WALK_DYNAMIC_COLLISION_PROBE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        ssbl.unregister()
        raise RuntimeError("; ".join(failures))

    result["passed"] = True
    print("SSBL_WALK_DYNAMIC_COLLISION_PROBE", json.dumps(result, ensure_ascii=False, sort_keys=True))
    ssbl.unregister()
    if bpy.app.background:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
