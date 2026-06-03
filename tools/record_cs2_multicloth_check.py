from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import bpy
from mathutils import Vector
import numpy as np


BLEND_PATH = os.environ.get("SSBL_CS2_BLEND", r"C:\Users\Administrator\Desktop\cs2.blend")
ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

OUTPUT_DIR = Path(os.environ.get(
    "SSBL_CS2_RECORD_DIR",
    Path(__file__).resolve().parents[1] / "recordings" / "cs2_multicloth_check",
))
FRAME_COUNT = max(int(os.environ.get("SSBL_CS2_RECORD_FRAMES", "72")), 1)
RENDER_STRIDE = max(int(os.environ.get("SSBL_CS2_RENDER_STRIDE", "3")), 1)
ACTIVE_OBJECT_NAME = os.environ.get("SSBL_CS2_ACTIVE_OBJECT", "Cube")
EXPECTED_SLOTS = int(os.environ.get("SSBL_CS2_EXPECTED_SLOTS", "2"))
EXPECTED_CROSS_MODE = os.environ.get("SSBL_CS2_EXPECTED_CROSS_MODE", "all_selected")
SELECT_CUBE = os.environ.get("SSBL_CS2_SELECT_CUBE", "1").strip().lower() not in {"0", "false", "no", "off"}
SELECT_SUZANNE = os.environ.get("SSBL_CS2_SELECT_SUZANNE", "1").strip().lower() not in {"0", "false", "no", "off"}
DISABLE_SELF_COLLISION = (
    os.environ.get("SSBL_CS2_DISABLE_SELF_COLLISION", "0").strip().lower() in {"1", "true", "yes", "on"}
)


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _bbox_world_from_positions(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(positions, dtype=np.float64)
    return np.min(array, axis=0), np.max(array, axis=0)


def _aabb_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    gap = np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.0)
    return float(np.linalg.norm(gap))


def _slot_max_edge_ratio(slot) -> float:
    positions = np.asarray(slot.current_positions_world, dtype=np.float64)
    edges = np.asarray(slot.cloth.edges, dtype=np.int32)
    rest = np.asarray(slot.cloth.edge_rest_lengths, dtype=np.float64)
    if len(edges) == 0 or len(rest) == 0:
        return 1.0
    current = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    ratios = current / np.maximum(rest, 1.0e-8)
    finite = ratios[np.isfinite(ratios)]
    return float(np.max(finite)) if len(finite) else float("inf")


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int32)
    return np.linspace(0, count - 1, limit, dtype=np.int32)


def _point_triangle_distances(points: np.ndarray, triangles: np.ndarray, point_chunk: int = 128) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.float64)
    if len(pts) == 0 or len(tris) == 0:
        return np.empty(0, dtype=np.float64)
    a = tris[:, 0, :]
    b = tris[:, 1, :]
    c = tris[:, 2, :]
    ab = b - a
    ac = c - a
    out = np.empty(len(pts), dtype=np.float64)
    for start in range(0, len(pts), point_chunk):
        p = pts[start : start + point_chunk]
        ap = p[:, None, :] - a[None, :, :]
        d1 = np.einsum("ntc,tc->nt", ap, ab)
        d2 = np.einsum("ntc,tc->nt", ap, ac)

        bp = p[:, None, :] - b[None, :, :]
        d3 = np.einsum("ntc,tc->nt", bp, ab)
        d4 = np.einsum("ntc,tc->nt", bp, ac)

        cp = p[:, None, :] - c[None, :, :]
        d5 = np.einsum("ntc,tc->nt", cp, ab)
        d6 = np.einsum("ntc,tc->nt", cp, ac)

        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = np.maximum(va + vb + vc, 1.0e-30)
        v = vb / denom
        w = vc / denom
        closest_face = a[None, :, :] + ab[None, :, :] * v[:, :, None] + ac[None, :, :] * w[:, :, None]

        closest = closest_face
        mask_a = (d1 <= 0.0) & (d2 <= 0.0)
        closest = np.where(mask_a[:, :, None], a[None, :, :], closest)

        mask_b = (d3 >= 0.0) & (d4 <= d3)
        closest = np.where(mask_b[:, :, None], b[None, :, :], closest)

        mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        edge_v = d1 / np.maximum(d1 - d3, 1.0e-30)
        closest_ab = a[None, :, :] + ab[None, :, :] * edge_v[:, :, None]
        closest = np.where(mask_ab[:, :, None], closest_ab, closest)

        mask_c = (d6 >= 0.0) & (d5 <= d6)
        closest = np.where(mask_c[:, :, None], c[None, :, :], closest)

        mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        edge_w = d2 / np.maximum(d2 - d6, 1.0e-30)
        closest_ac = a[None, :, :] + ac[None, :, :] * edge_w[:, :, None]
        closest = np.where(mask_ac[:, :, None], closest_ac, closest)

        mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
        edge_w_bc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1.0e-30)
        closest_bc = b[None, :, :] + (c - b)[None, :, :] * edge_w_bc[:, :, None]
        closest = np.where(mask_bc[:, :, None], closest_bc, closest)

        dist_sq = np.sum((p[:, None, :] - closest) ** 2, axis=2)
        out[start : start + len(p)] = np.sqrt(np.min(dist_sq, axis=1))
    return out


def _cross_surface_metrics(slot_a, slot_b, contact_distance: float) -> dict[str, float | int | None]:
    positions_a = np.asarray(slot_a.current_positions_world, dtype=np.float64)
    positions_b = np.asarray(slot_b.current_positions_world, dtype=np.float64)
    triangles_a = np.asarray(slot_a.cloth.triangles, dtype=np.int32)
    triangles_b = np.asarray(slot_b.cloth.triangles, dtype=np.int32)
    if len(positions_a) == 0 or len(positions_b) == 0 or len(triangles_a) == 0 or len(triangles_b) == 0:
        return {"min_surface_distance": None, "near_contact_vertices": 0}
    sample_a = _sample_indices(len(positions_a), 2048)
    sample_b = _sample_indices(len(positions_b), 2048)
    sample_tri_a = _sample_indices(len(triangles_a), 4096)
    sample_tri_b = _sample_indices(len(triangles_b), 4096)
    dist_a = _point_triangle_distances(positions_a[sample_a], positions_b[triangles_b[sample_tri_b]], point_chunk=64)
    dist_b = _point_triangle_distances(positions_b[sample_b], positions_a[triangles_a[sample_tri_a]], point_chunk=64)
    distances = np.concatenate([dist_a, dist_b])
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return {"min_surface_distance": None, "near_contact_vertices": 0}
    return {
        "min_surface_distance": float(np.min(finite)),
        "near_contact_vertices": int(np.count_nonzero(finite < contact_distance)),
    }


def _sphere_gap_metrics(slot, sphere_obj: bpy.types.Object, margin: float) -> dict[str, float | int | None]:
    if slot is None or sphere_obj is None:
        return {"min_sphere_gap": None, "sphere_penetrating_vertices": 0}
    positions = np.asarray(slot.current_positions_world, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        return {"min_sphere_gap": None, "sphere_penetrating_vertices": 0}
    center = np.array(sphere_obj.matrix_world.translation, dtype=np.float64)
    radius = max(float(max(sphere_obj.dimensions)) * 0.5 + float(margin), 0.0)
    gaps = np.linalg.norm(positions - center[None, :], axis=1) - radius
    finite = gaps[np.isfinite(gaps)]
    if len(finite) == 0:
        return {"min_sphere_gap": None, "sphere_penetrating_vertices": 0}
    return {
        "min_sphere_gap": float(np.min(finite)),
        "sphere_penetrating_vertices": int(np.count_nonzero(finite < -1.0e-4)),
    }


def _ensure_camera(scene: bpy.types.Scene, objects: list[bpy.types.Object]) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 12
    if scene.world is not None:
        scene.world.color = (0.025, 0.03, 0.035)
    centers = []
    for obj in objects:
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        centers.append(sum(corners, Vector()) / max(len(corners), 1))
    center = sum(centers, Vector()) / max(len(centers), 1)
    camera_data = bpy.data.cameras.new("SSBL_CS2_Check_Camera")
    camera = bpy.data.objects.new("SSBL_CS2_Check_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (center.x + 3.4, center.y - 4.8, center.z + 2.1)
    camera_data.lens = 45
    _look_at(camera, center)
    scene.camera = camera


def _render_frame(scene: bpy.types.Scene, frames_dir: Path, render_index: int) -> str:
    path = frames_dir / f"frame_{render_index:04d}.png"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    try:
        bpy.ops.render.opengl(write_still=True, view_context=False)
    except RuntimeError:
        bpy.ops.render.render(write_still=True)
    return str(path)


def _encode_video(frames_dir: Path, video_path: Path) -> str | None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "12",
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None
    return str(video_path)


def _restore_mesh_delta(obj: bpy.types.Object, before: list[tuple[float, float, float]]) -> float:
    if len(obj.data.vertices) != len(before):
        return float("inf")
    return max(
        (
            max(
                abs(float(vertex.co.x) - old[0]),
                abs(float(vertex.co.y) - old[1]),
                abs(float(vertex.co.z) - old[2]),
            )
            for vertex, old in zip(obj.data.vertices, before)
        ),
        default=0.0,
    )


def _snapshot(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in obj.data.vertices]


def main() -> None:
    bpy.ops.wm.open_mainfile(filepath=BLEND_PATH, load_ui=False)
    import ssbl

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    frames_dir = OUTPUT_DIR / "frames"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    scene = bpy.context.scene
    scene.frame_set(1)
    cube = bpy.data.objects.get("Cube")
    suzanne = bpy.data.objects.get("Suzanne")
    if cube is None or suzanne is None:
        raise RuntimeError("Expected Cube and Suzanne in cs2.blend")
    if cube.type != "MESH" or suzanne.type != "MESH":
        raise RuntimeError("Cube and Suzanne must both be mesh objects")
    active_obj = bpy.data.objects.get(ACTIVE_OBJECT_NAME)
    if active_obj is None or active_obj.type != "MESH":
        raise RuntimeError(f"Expected active mesh object {ACTIVE_OBJECT_NAME!r} in cs2.blend")

    for obj in (cube, suzanne):
        obj.hide_viewport = False
        obj.ssbl_cloth.enabled = True
        obj.ssbl_cloth.preview_writeback_interval = 0
        if DISABLE_SELF_COLLISION:
            obj.ssbl_cloth.self_collision = False
            obj.ssbl_cloth.self_collision_mode = "off"
    cube.select_set(SELECT_CUBE)
    suzanne.select_set(SELECT_SUZANNE)
    active_obj.select_set(True)
    bpy.context.view_layer.objects.active = active_obj
    scene.frame_start = 1
    scene.frame_end = max(scene.frame_end, FRAME_COUNT + 2)
    _ensure_camera(scene, [cube, suzanne])

    before_cube = _snapshot(cube)
    before_suzanne = _snapshot(suzanne)
    session = ssbl.solver.start_preview(bpy.context, active_obj)
    render_index = 0
    samples = []
    max_edge_ratio = 0.0
    finite = True
    first_dynamic_frame = None
    max_analytic_collision_ms = 0.0
    max_dynamic_collision_ms = 0.0
    max_dynamic_particle_collision_ms = 0.0
    max_dynamic_particle_count = 0
    max_dynamic_particle_contacts = 0
    max_dynamic_particle_overflow = 0
    min_aabb_distance = float("inf")
    min_surface_distance = float("inf")
    max_near_contact_vertices = 0
    min_sphere_gap = float("inf")
    max_sphere_penetrating_vertices = 0
    for frame in range(1, FRAME_COUNT + 1):
        finished = ssbl.solver.step_preview(bpy.context, active_obj.name)
        diag = ssbl.solver.session_diagnostics(active_obj)
        finite = finite and bool(diag.finite)
        max_analytic_collision_ms = max(max_analytic_collision_ms, float(diag.analytic_collision_ms))
        max_dynamic_collision_ms = max(max_dynamic_collision_ms, float(diag.dynamic_collision_ms))
        max_dynamic_particle_collision_ms = max(
            max_dynamic_particle_collision_ms,
            float(diag.dynamic_particle_collision_ms),
        )
        max_dynamic_particle_count = max(max_dynamic_particle_count, int(diag.dynamic_particle_count))
        max_dynamic_particle_contacts = max(max_dynamic_particle_contacts, int(diag.dynamic_particle_contacts))
        max_dynamic_particle_overflow = max(max_dynamic_particle_overflow, int(diag.dynamic_particle_overflow))
        for slot in session.slots.values():
            max_edge_ratio = max(max_edge_ratio, _slot_max_edge_ratio(slot))
        active_slot = session.slots.get(active_obj.name)
        if active_slot is not None and cube.name not in session.slots:
            sphere_metrics = _sphere_gap_metrics(
                active_slot,
                cube,
                float(getattr(active_obj.ssbl_cloth, "collision_margin", 0.0)),
            )
            if sphere_metrics["min_sphere_gap"] is not None:
                min_sphere_gap = min(min_sphere_gap, float(sphere_metrics["min_sphere_gap"]))
            max_sphere_penetrating_vertices = max(
                max_sphere_penetrating_vertices,
                int(sphere_metrics["sphere_penetrating_vertices"]),
            )
        else:
            sphere_metrics = {"min_sphere_gap": None, "sphere_penetrating_vertices": 0}
        slot_a = session.slots.get(cube.name)
        slot_b = session.slots.get(suzanne.name)
        aabb_distance = None
        if slot_a is not None and slot_b is not None:
            aabb_distance = _aabb_distance(
                _bbox_world_from_positions(slot_a.current_positions_world),
                _bbox_world_from_positions(slot_b.current_positions_world),
            )
            min_aabb_distance = min(min_aabb_distance, aabb_distance)
            contact_distance = max(float(slot_a.external_contact_distance), float(slot_b.external_contact_distance))
            if aabb_distance <= max(contact_distance * 4.0, 0.08) or int(diag.dynamic_triangle_count) > 0:
                surface_metrics = _cross_surface_metrics(slot_a, slot_b, contact_distance)
            else:
                surface_metrics = {"min_surface_distance": None, "near_contact_vertices": 0}
            if surface_metrics["min_surface_distance"] is not None:
                min_surface_distance = min(min_surface_distance, float(surface_metrics["min_surface_distance"]))
            max_near_contact_vertices = max(max_near_contact_vertices, int(surface_metrics["near_contact_vertices"]))
        else:
            surface_metrics = {"min_surface_distance": None, "near_contact_vertices": 0}
        if first_dynamic_frame is None and int(diag.dynamic_triangle_count) > 0:
            first_dynamic_frame = frame
        item = {
            "frame": frame,
            "finite": bool(diag.finite),
            "dynamic_triangle_count": int(diag.dynamic_triangle_count),
            "dynamic_particle_count": int(diag.dynamic_particle_count),
            "dynamic_particle_candidate_count": int(diag.dynamic_particle_candidate_count),
            "dynamic_particle_contacts": int(diag.dynamic_particle_contacts),
            "dynamic_particle_overflow": int(diag.dynamic_particle_overflow),
            "analytic_collision_ms": float(diag.analytic_collision_ms),
            "static_collision_ms": float(diag.static_collision_ms),
            "dynamic_upload_ms": float(diag.dynamic_upload_ms),
            "dynamic_collision_ms": float(diag.dynamic_collision_ms),
            "dynamic_particle_collision_ms": float(diag.dynamic_particle_collision_ms),
            "resolved_contacts": int(diag.resolved_contacts),
            "candidate_count": int(diag.candidate_count),
            "min_gap": None if diag.min_gap is None else float(diag.min_gap),
            "aabb_distance": aabb_distance,
            "min_surface_distance": surface_metrics["min_surface_distance"],
            "near_contact_vertices": surface_metrics["near_contact_vertices"],
            "min_sphere_gap": sphere_metrics["min_sphere_gap"],
            "sphere_penetrating_vertices": sphere_metrics["sphere_penetrating_vertices"],
            "max_edge_ratio_so_far": float(max_edge_ratio),
        }
        samples.append(item)
        if frame == 1 or frame % RENDER_STRIDE == 0 or int(diag.dynamic_triangle_count) > 0:
            _render_frame(scene, frames_dir, render_index)
            render_index += 1
        if finished:
            break
    final_diag = ssbl.solver.session_diagnostics(active_obj)
    stopped = ssbl.solver.request_stop(active_obj)
    video = _encode_video(frames_dir, OUTPUT_DIR / "cs2_multicloth_check.mp4")
    summary = {
        "blend_file": bpy.data.filepath,
        "output_dir": str(OUTPUT_DIR),
        "video": video,
        "frames_requested": FRAME_COUNT,
        "frames_sampled": len(samples),
        "slots": len(session.slots),
        "slot_names": list(session.slots.keys()),
        "solve_order": list(session.solve_order),
        "cross_mode": str(session.cross_cloth_mode),
        "selected_after_setup": [obj.name for obj in bpy.context.selected_objects],
        "active_after_setup": bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else None,
        "first_dynamic_frame": first_dynamic_frame,
        "min_aabb_distance": None if not math.isfinite(min_aabb_distance) else min_aabb_distance,
        "min_surface_distance": None if not math.isfinite(min_surface_distance) else min_surface_distance,
        "max_near_contact_vertices": int(max_near_contact_vertices),
        "min_sphere_gap": None if not math.isfinite(min_sphere_gap) else min_sphere_gap,
        "max_sphere_penetrating_vertices": int(max_sphere_penetrating_vertices),
        "max_edge_ratio": float(max_edge_ratio),
        "max_analytic_collision_ms": float(max_analytic_collision_ms),
        "max_dynamic_collision_ms": float(max_dynamic_collision_ms),
        "max_dynamic_particle_collision_ms": float(max_dynamic_particle_collision_ms),
        "max_dynamic_particle_count": int(max_dynamic_particle_count),
        "max_dynamic_particle_contacts": int(max_dynamic_particle_contacts),
        "max_dynamic_particle_overflow": int(max_dynamic_particle_overflow),
        "finite": bool(finite),
        "stopped": bool(stopped),
        "restore_delta_cube": _restore_mesh_delta(cube, before_cube),
        "restore_delta_suzanne": _restore_mesh_delta(suzanne, before_suzanne),
        "final_dynamic_triangle_count": int(final_diag.dynamic_triangle_count),
        "final_dynamic_particle_count": int(final_diag.dynamic_particle_count),
        "final_dynamic_particle_contacts": int(final_diag.dynamic_particle_contacts),
        "final_dynamic_particle_overflow": int(final_diag.dynamic_particle_overflow),
        "final_analytic_collision_ms": float(final_diag.analytic_collision_ms),
        "final_dynamic_collision_ms": float(final_diag.dynamic_collision_ms),
        "final_dynamic_particle_collision_ms": float(final_diag.dynamic_particle_collision_ms),
        "samples": samples,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SSBL_CS2_MULTICLOTH_CHECK", json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False, sort_keys=True))
    if len(session.slots) != EXPECTED_SLOTS or str(session.cross_cloth_mode) != EXPECTED_CROSS_MODE or not finite:
        raise RuntimeError(f"cs2 multi-cloth recording validation failed: {summary}")


if __name__ == "__main__":
    main()
