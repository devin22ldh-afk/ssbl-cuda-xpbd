from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

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
import record_realtime_demo_pack as demo


BLEND_PATH = Path(os.environ.get("SSBL_WALK_BLEND", r"C:\Users\Administrator\Desktop\演示视频\walk.blend"))
FRAME_COUNT = max(int(os.environ.get("SSBL_WALK_RECORD_FRAMES", "60")), 1)
WARMUP_FRAME_COUNT = max(int(os.environ.get("SSBL_WALK_RECORD_WARMUP_FRAMES", "1")), 0)
SKIRT_NAME = os.environ.get("SSBL_WALK_SKIRT_OBJECT", "Codex_Skirt_Pleated")
BETA_NAME = os.environ.get("SSBL_WALK_BETA_OBJECT", "Beta_Surface")
DYNAMIC_COLLECTION_NAME = os.environ.get("SSBL_WALK_DYNAMIC_COLLIDER_COLLECTION", "SSBL_Runtime_Dynamic_Colliders")
CASE_NAME = os.environ.get("SSBL_WALK_RECORD_NAME", "walk_60_dynamic_collision")
SUBSTEPS_OVERRIDE = int(os.environ.get("SSBL_WALK_RECORD_SUBSTEPS", "0") or "0")
ITERATIONS_OVERRIDE = int(os.environ.get("SSBL_WALK_RECORD_ITERATIONS", "0") or "0")


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


def _assign_dynamic_collider_collection(scene: bpy.types.Scene, collider: bpy.types.Object) -> None:
    collection = bpy.data.collections.get(DYNAMIC_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(DYNAMIC_COLLECTION_NAME)
    if not any(child.name == collection.name for child in scene.collection.children):
        scene.collection.children.link(collection)
    if not any(item.name == collider.name for item in collection.objects):
        collection.objects.link(collider)
    scene.ssbl_preview.dynamic_collider_collection = collection
    for cloth_obj in _enabled_cloth_objects(scene):
        if cloth_obj.name == collider.name:
            continue
        cloth_obj.ssbl_cloth.dynamic_collider_collection = collection
    collider["ssbl_enable_cross_cloth_collision"] = True


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _ensure_camera(scene: bpy.types.Scene, objects: list[bpy.types.Object]) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x = demo.RESOLUTION[0]
    scene.render.resolution_y = demo.RESOLUTION[1]
    scene.render.fps = int(demo.VIDEO_FPS)
    scene.render.image_settings.file_format = "PNG"
    if scene.camera is not None:
        return
    centers = []
    for obj in objects:
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        if corners:
            centers.append(sum(corners, Vector()) / len(corners))
    center = sum(centers, Vector()) / max(len(centers), 1)
    camera_data = bpy.data.cameras.new("SSBL_Walk_Record_Camera")
    camera = bpy.data.objects.new("SSBL_Walk_Record_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (center.x + 2.8, center.y - 4.5, center.z + 1.4)
    camera_data.lens = 48
    _look_at(camera, center)
    scene.camera = camera


def _signed_surface_summary(points: np.ndarray, source) -> dict[str, float | int | None]:
    vertices = np.asarray(source.current_positions_world, dtype=np.float64)
    triangles = np.asarray(source.triangle_indices, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or len(triangles) == 0:
        return {"min_signed_distance": None, "max_inside_depth": None, "inside_vertex_count": 0}
    if np.min(triangles) < 0 or np.max(triangles) >= len(vertices):
        return {"min_signed_distance": None, "max_inside_depth": None, "inside_vertex_count": 0}
    tree = BVHTree.FromPolygons(
        [Vector((float(x), float(y), float(z))) for x, y, z in vertices],
        [tuple(int(v) for v in tri) for tri in triangles],
        all_triangles=True,
    )
    min_signed = math.inf
    max_inside_depth = 0.0
    inside_count = 0
    for row in np.asarray(points, dtype=np.float64).reshape((-1, 3)):
        point = Vector((float(row[0]), float(row[1]), float(row[2])))
        nearest = tree.find_nearest(point)
        if nearest is None or nearest[0] is None or nearest[1] is None:
            continue
        location, normal, _index, _distance = nearest
        if normal.length <= 1.0e-9:
            continue
        signed = float((point - location).dot(normal.normalized()))
        min_signed = min(min_signed, signed)
        if signed < 0.0:
            inside_count += 1
            max_inside_depth = max(max_inside_depth, -signed)
    if not math.isfinite(min_signed):
        return {"min_signed_distance": None, "max_inside_depth": None, "inside_vertex_count": 0}
    return {
        "min_signed_distance": float(min_signed),
        "max_inside_depth": float(max_inside_depth),
        "inside_vertex_count": int(inside_count),
    }


def _step_record_frame(context: bpy.types.Context, scene: bpy.types.Scene, session, frame: int) -> None:
    scene.frame_set(int(frame))
    previous_frame_index = int(session.frame_index)
    ended = bool(ssbl.solver.step_timeline_preview(context, scene))
    if ended:
        raise RuntimeError(f"walk timeline preview ended before recording frame {frame}")
    if int(session.frame_index) <= previous_frame_index:
        raise RuntimeError(f"walk timeline preview skipped solver step at frame {frame}")


def main() -> None:
    if not BLEND_PATH.exists():
        raise RuntimeError(f"Missing walk blend: {BLEND_PATH}")
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH), load_ui=False)
    _register_addon()
    ssbl.solver.cleanup_all_sessions()

    scene = bpy.context.scene
    scene.frame_start = int(scene.frame_start)
    scene.frame_end = max(int(scene.frame_end), int(scene.frame_start) + WARMUP_FRAME_COUNT + FRAME_COUNT + 1)
    scene.frame_set(int(scene.frame_start))
    skirt = bpy.data.objects.get(SKIRT_NAME)
    beta = bpy.data.objects.get(BETA_NAME)
    if skirt is None or skirt.type != "MESH":
        raise RuntimeError(f"Missing skirt mesh: {SKIRT_NAME}")
    if beta is None or beta.type != "MESH":
        raise RuntimeError(f"Missing dynamic collider mesh: {BETA_NAME}")
    if SUBSTEPS_OVERRIDE > 0:
        skirt.ssbl_cloth.substeps = int(SUBSTEPS_OVERRIDE)
    if ITERATIONS_OVERRIDE > 0:
        skirt.ssbl_cloth.iterations = int(ITERATIONS_OVERRIDE)
    _assign_dynamic_collider_collection(scene, beta)
    _ensure_camera(scene, [skirt, beta])

    case_dir, frames_dir, video_path = demo._ensure_output_dir(CASE_NAME)
    frame_paths: list[str] = []
    overlay_text_frames: list[str] = []
    step_ms_samples: list[float] = []
    viewport_fps_samples: list[float] = []
    signed_samples: list[dict[str, float | int | None]] = []

    started = time.perf_counter()
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        raise RuntimeError("walk timeline preview did not start")
    finite = True
    max_dynamic_triangles = 0
    max_cuda_ms = 0.0
    max_input_ms = 0.0
    max_dynamic_upload_ms = 0.0
    max_dynamic_collision_ms = 0.0
    max_dynamic_particle_collision_ms = 0.0
    max_dynamic_triangle_candidates = 0
    max_dynamic_triangle_aabb_rejects = 0
    max_dynamic_triangle_bucket_occupancy = 0
    max_dynamic_triangle_large_primitives = 0
    max_download_ms = 0.0
    max_writeback_ms = 0.0
    max_inside_depth = 0.0
    max_inside_count = 0
    min_signed_distance = math.inf
    try:
        for warmup_index in range(WARMUP_FRAME_COUNT):
            warmup_frame = int(scene.frame_start) + warmup_index + 1
            _step_record_frame(bpy.context, scene, session, warmup_frame)
        for index in range(FRAME_COUNT):
            frame = int(scene.frame_start) + WARMUP_FRAME_COUNT + index + 1
            frame_started = time.perf_counter()
            _step_record_frame(bpy.context, scene, session, frame)
            step_ms = (time.perf_counter() - frame_started) * 1000.0
            diag = ssbl.solver.session_diagnostics(skirt)
            slot = session.slots.get(SKIRT_NAME)
            source = session.dynamic_collision_sources.get(BETA_NAME)
            if slot is None or source is None:
                raise RuntimeError("walk session lost skirt slot or Beta dynamic source")
            signed = _signed_surface_summary(np.asarray(slot.current_positions_world, dtype=np.float32), source)
            signed_samples.append(signed)
            if signed["min_signed_distance"] is not None:
                min_signed_distance = min(min_signed_distance, float(signed["min_signed_distance"]))
            if signed["max_inside_depth"] is not None:
                max_inside_depth = max(max_inside_depth, float(signed["max_inside_depth"]))
            max_inside_count = max(max_inside_count, int(signed["inside_vertex_count"] or 0))
            finite = finite and bool(getattr(diag, "finite", True)) and bool(np.isfinite(slot.current_positions_world).all())
            max_dynamic_triangles = max(max_dynamic_triangles, int(getattr(diag, "dynamic_triangle_count", 0)))
            max_cuda_ms = max(max_cuda_ms, float(getattr(diag, "cuda_step_call_ms", 0.0)))
            max_input_ms = max(max_input_ms, float(getattr(diag, "input_refresh_ms", 0.0)))
            max_dynamic_upload_ms = max(max_dynamic_upload_ms, float(getattr(diag, "dynamic_upload_ms", 0.0)))
            max_dynamic_collision_ms = max(max_dynamic_collision_ms, float(getattr(diag, "dynamic_collision_ms", 0.0)))
            max_dynamic_particle_collision_ms = max(
                max_dynamic_particle_collision_ms,
                float(getattr(diag, "dynamic_particle_collision_ms", 0.0)),
            )
            max_dynamic_triangle_candidates = max(
                max_dynamic_triangle_candidates,
                int(getattr(diag, "dynamic_triangle_candidate_count", 0)),
            )
            max_dynamic_triangle_aabb_rejects = max(
                max_dynamic_triangle_aabb_rejects,
                int(getattr(diag, "dynamic_triangle_aabb_reject_count", 0)),
            )
            max_dynamic_triangle_bucket_occupancy = max(
                max_dynamic_triangle_bucket_occupancy,
                int(getattr(diag, "dynamic_triangle_max_bucket_occupancy", 0)),
            )
            max_dynamic_triangle_large_primitives = max(
                max_dynamic_triangle_large_primitives,
                int(getattr(diag, "dynamic_triangle_large_primitive_count", 0)),
            )
            max_download_ms = max(max_download_ms, float(getattr(diag, "download_ms", 0.0)))
            max_writeback_ms = max(max_writeback_ms, float(getattr(diag, "writeback_ms", 0.0)))
            step_ms_samples.append(step_ms)
            fps = 1000.0 / max(step_ms, 1.0e-6)
            viewport_fps_samples.append(fps)
            overlay_text_frames.append(demo._compose_overlay_text("Walk Dynamic Collider", f"Viewport FPS: {fps:5.1f}", ""))
            frame_paths.append(demo._render_frame(scene, frames_dir, index + 1))
    finally:
        ssbl.solver.stop_timeline_preview(scene)

    elapsed = time.perf_counter() - started
    demo._encode_video(frames_dir, video_path, overlay_text_frames=overlay_text_frames)
    probe = demo._ffprobe(video_path)
    summary = {
        "blend_path": str(BLEND_PATH),
        "video": str(video_path),
        "frames_dir": str(frames_dir),
        "frame_count": int(len(frame_paths)),
        "warmup_frame_count": int(WARMUP_FRAME_COUNT),
        "simulation_elapsed_s": float(elapsed),
        "average_record_step_fps": float(len(step_ms_samples) / max(elapsed, 1.0e-9)),
        "average_step_ms": float(sum(step_ms_samples) / max(len(step_ms_samples), 1)),
        "p95_step_ms": float(demo._p95(step_ms_samples)),
        "average_overlay_fps": float(demo._average_fps(viewport_fps_samples)),
        "finite": bool(finite),
        "slot_names": list(session.solve_order),
        "dynamic_collision_source_names": sorted(session.dynamic_collision_sources.keys()),
        "max_dynamic_triangle_count": int(max_dynamic_triangles),
        "max_cuda_step_call_ms": float(max_cuda_ms),
        "max_input_refresh_ms": float(max_input_ms),
        "max_dynamic_upload_ms": float(max_dynamic_upload_ms),
        "max_dynamic_collision_ms": float(max_dynamic_collision_ms),
        "max_dynamic_particle_collision_ms": float(max_dynamic_particle_collision_ms),
        "max_dynamic_triangle_candidate_count": int(max_dynamic_triangle_candidates),
        "max_dynamic_triangle_aabb_reject_count": int(max_dynamic_triangle_aabb_rejects),
        "max_dynamic_triangle_max_bucket_occupancy": int(max_dynamic_triangle_bucket_occupancy),
        "max_dynamic_triangle_large_primitive_count": int(max_dynamic_triangle_large_primitives),
        "max_download_ms": float(max_download_ms),
        "max_writeback_ms": float(max_writeback_ms),
        "min_signed_distance": None if not math.isfinite(min_signed_distance) else float(min_signed_distance),
        "max_inside_depth": float(max_inside_depth),
        "max_inside_vertex_count": int(max_inside_count),
        "signed_samples": signed_samples,
        "ffprobe": probe,
    }
    summary_path = case_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SSBL_WALK_60_RECORD", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not finite or not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"walk recording failed: {summary}")
    if bpy.app.background:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
