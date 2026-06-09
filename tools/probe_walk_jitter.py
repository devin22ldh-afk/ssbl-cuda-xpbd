from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


BLEND_PATH = Path(os.environ.get("SSBL_WALK_BLEND", r"C:\Users\Administrator\Desktop\演示视频\walk.blend"))
FRAME_COUNT = max(int(os.environ.get("SSBL_WALK_JITTER_FRAMES", "60")), 4)
WARMUP_FRAMES = max(int(os.environ.get("SSBL_WALK_JITTER_WARMUP", "1")), 0)
SKIRT_NAME = os.environ.get("SSBL_WALK_SKIRT_OBJECT", "Codex_Skirt_Pleated")
BETA_NAME = os.environ.get("SSBL_WALK_BETA_OBJECT", "Beta_Surface")
DYNAMIC_COLLECTION_NAME = os.environ.get("SSBL_WALK_DYNAMIC_COLLIDER_COLLECTION", "SSBL_Runtime_Dynamic_Colliders")
MODE = os.environ.get("SSBL_WALK_JITTER_MODE", "default").strip().lower()


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


def _patch_no_dynamic_particles() -> None:
    from ssbl import session_manager

    original = session_manager._collect_cross_cloth_colliders

    def wrapper(session, target, perf=None):
        triangles, indexed, particles = original(session, target, perf)
        _ = particles
        return triangles, indexed, session_manager._empty_dynamic_particles()

    session_manager._collect_cross_cloth_colliders = wrapper


def _step_record_frame(scene: bpy.types.Scene, session, frame: int) -> None:
    scene.frame_set(int(frame))
    previous_frame_index = int(session.frame_index)
    ended = bool(ssbl.solver.step_timeline_preview(bpy.context, scene))
    if ended:
        raise RuntimeError(f"timeline preview ended before frame {frame}")
    if int(session.frame_index) <= previous_frame_index:
        raise RuntimeError(f"timeline preview skipped solver step at frame {frame}")


def _percentile(values: np.ndarray, percent: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape((-1,))
    if arr.size <= 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return 0.0
    return float(np.percentile(finite, percent))


def _pin_reference_mask(slot) -> np.ndarray:
    inv_mass = np.asarray(slot.cloth.inv_mass, dtype=np.float32).reshape((-1,))
    pin_weights = np.asarray(getattr(slot.cloth, "pin_weights", np.empty(0, dtype=np.float32)), dtype=np.float32).reshape((-1,))
    if len(pin_weights) == len(inv_mass):
        mask = pin_weights >= 0.99
    else:
        mask = inv_mass <= 1.0e-8
    if not bool(np.any(mask)):
        mask = inv_mass <= 1.0e-8
    return mask


def _summarize_motion(samples: np.ndarray, pin_mask: np.ndarray, free_mask: np.ndarray) -> dict[str, object]:
    if samples.ndim != 3 or samples.shape[0] < 4:
        raise RuntimeError("not enough position samples for jitter summary")
    if not bool(np.any(pin_mask)):
        pin_mask = np.ones(samples.shape[1], dtype=bool)
    if not bool(np.any(free_mask)):
        free_mask = np.ones(samples.shape[1], dtype=bool)

    ref = np.mean(samples[:, pin_mask, :], axis=1)
    rel = samples - ref[:, None, :]
    velocity = np.diff(rel, axis=0)
    accel = rel[2:] - 2.0 * rel[1:-1] + rel[:-2]
    free_velocity = velocity[:, free_mask, :]
    free_accel = accel[:, free_mask, :]
    speed = np.linalg.norm(free_velocity, axis=2)
    accel_mag = np.linalg.norm(free_accel, axis=2)

    prev_v = free_velocity[:-1]
    next_v = free_velocity[1:]
    prev_len = np.linalg.norm(prev_v, axis=2)
    next_len = np.linalg.norm(next_v, axis=2)
    dot = np.einsum("tvc,tvc->tv", prev_v, next_v)
    flip_mask = (dot < 0.0) & (prev_len > 1.0e-5) & (next_len > 1.0e-5)
    flip_ratio = float(np.count_nonzero(flip_mask) / max(int(flip_mask.size), 1))

    per_vertex_accel = np.max(accel_mag, axis=0)
    free_indices = np.flatnonzero(free_mask)
    top_order = np.argsort(per_vertex_accel)[::-1][:10]
    top_vertices = [
        {
            "vertex": int(free_indices[index]),
            "max_rel_accel_m": float(per_vertex_accel[index]),
        }
        for index in top_order
    ]
    frame_accel_p95 = np.percentile(accel_mag, 95, axis=1)
    peak_frame_index = int(np.argmax(frame_accel_p95)) + 2
    return {
        "speed_p50_m_per_frame": _percentile(speed, 50),
        "speed_p95_m_per_frame": _percentile(speed, 95),
        "speed_p99_m_per_frame": _percentile(speed, 99),
        "rel_accel_p50_m": _percentile(accel_mag, 50),
        "rel_accel_p95_m": _percentile(accel_mag, 95),
        "rel_accel_p99_m": _percentile(accel_mag, 99),
        "rel_accel_max_m": float(np.max(accel_mag)) if accel_mag.size else 0.0,
        "velocity_flip_ratio": flip_ratio,
        "peak_accel_sample": peak_frame_index,
        "peak_accel_frame_p95_m": float(np.max(frame_accel_p95)) if frame_accel_p95.size else 0.0,
        "top_jitter_vertices": top_vertices,
    }


def main() -> None:
    if not BLEND_PATH.exists():
        raise RuntimeError(f"Missing walk blend: {BLEND_PATH}")
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH), load_ui=False)
    _register_addon()
    ssbl.solver.cleanup_all_sessions()

    scene = bpy.context.scene
    scene.frame_set(int(scene.frame_start))
    skirt = bpy.data.objects.get(SKIRT_NAME)
    beta = bpy.data.objects.get(BETA_NAME)
    if skirt is None or skirt.type != "MESH":
        raise RuntimeError(f"Missing skirt mesh: {SKIRT_NAME}")
    if beta is None or beta.type != "MESH":
        raise RuntimeError(f"Missing dynamic collider mesh: {BETA_NAME}")

    if MODE == "no_dynamic_particles":
        _patch_no_dynamic_particles()
    elif MODE == "no_self_collision":
        skirt.ssbl_cloth.self_collision = False
    elif MODE == "no_dynamic_collider":
        scene.ssbl_preview.dynamic_collider_collection = None
        skirt.ssbl_cloth.dynamic_collider_collection = None
    elif MODE != "default":
        raise RuntimeError(f"Unknown jitter probe mode: {MODE}")

    if MODE != "no_dynamic_collider":
        _assign_dynamic_collider_collection(scene, beta)

    frame_end = int(scene.frame_start) + WARMUP_FRAMES + FRAME_COUNT + 2
    scene.frame_end = max(int(scene.frame_end), frame_end)
    started = time.perf_counter()
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        raise RuntimeError("timeline preview did not start")

    positions: list[np.ndarray] = []
    diag_rows: list[dict[str, object]] = []
    try:
        for warmup_index in range(WARMUP_FRAMES):
            _step_record_frame(scene, session, int(scene.frame_start) + warmup_index + 1)
        for index in range(FRAME_COUNT):
            frame = int(scene.frame_start) + WARMUP_FRAMES + index + 1
            _step_record_frame(scene, session, frame)
            diag = ssbl.solver.session_diagnostics(skirt)
            slot = session.slots.get(SKIRT_NAME)
            if slot is None:
                raise RuntimeError("walk session lost skirt slot")
            positions.append(np.asarray(slot.current_positions_world, dtype=np.float32).copy())
            diag_rows.append(
                {
                    "frame": int(frame),
                    "dynamic_triangle_contacts": int(getattr(diag, "resolved_contacts", 0)),
                    "dynamic_particle_candidates": int(getattr(diag, "dynamic_particle_candidate_count", 0)),
                    "dynamic_particle_contacts": int(getattr(diag, "dynamic_particle_contacts", 0)),
                    "dynamic_triangle_candidates": int(getattr(diag, "dynamic_triangle_candidate_count", 0)),
                    "dynamic_triangle_aabb_rejects": int(getattr(diag, "dynamic_triangle_aabb_reject_count", 0)),
                    "self_candidates": int(getattr(diag, "self_candidate_count", 0)),
                    "pcg_guarded": int(getattr(diag, "abi41_pcg_guarded", 0)),
                    "pcg_max_delta": float(getattr(diag, "abi41_pcg_max_delta", 0.0)),
                    "jitter_stabilized_vertices": int(getattr(diag, "jitter_stabilized_vertices", 0)),
                    "jitter_rejected_vertices": int(getattr(diag, "jitter_rejected_vertices", 0)),
                    "jitter_max_correction": float(getattr(diag, "jitter_max_correction", 0.0)),
                }
            )
    finally:
        ssbl.solver.stop_timeline_preview(scene)

    samples = np.stack(positions, axis=0)
    slot = session.slots.get(SKIRT_NAME)
    if slot is None:
        raise RuntimeError("walk session lost skirt slot after stop")
    inv_mass = np.asarray(slot.cloth.inv_mass, dtype=np.float32).reshape((-1,))
    pin_weights = np.asarray(getattr(slot.cloth, "pin_weights", np.empty(0, dtype=np.float32)), dtype=np.float32).reshape((-1,))
    pin_mask = _pin_reference_mask(slot)
    free_mask = inv_mass > 1.0e-8
    motion = _summarize_motion(samples, pin_mask, free_mask)
    particle_contacts = np.asarray([row["dynamic_particle_contacts"] for row in diag_rows], dtype=np.float64)
    triangle_candidates = np.asarray([row["dynamic_triangle_candidates"] for row in diag_rows], dtype=np.float64)
    self_candidates = np.asarray([row["self_candidates"] for row in diag_rows], dtype=np.float64)

    result = {
        "blend_path": str(BLEND_PATH),
        "mode": MODE,
        "frames": int(FRAME_COUNT),
        "warmup_frames": int(WARMUP_FRAMES),
        "elapsed_s": float(time.perf_counter() - started),
        "slot_names": list(session.solve_order),
        "dynamic_collision_source_names": sorted(session.dynamic_collision_sources.keys()),
        "vertex_count": int(samples.shape[1]),
        "free_vertex_count": int(np.count_nonzero(free_mask)),
        "pin_reference_count": int(np.count_nonzero(pin_mask)),
        "hard_pin_weight_count": int(np.count_nonzero(pin_weights >= 0.99)) if len(pin_weights) == len(inv_mass) else 0,
        "pin_weight_min": float(np.min(pin_weights)) if pin_weights.size else None,
        "pin_weight_max": float(np.max(pin_weights)) if pin_weights.size else None,
        "motion": motion,
        "diag_max": {
            "dynamic_particle_contacts": int(np.max(particle_contacts)) if particle_contacts.size else 0,
            "dynamic_particle_candidates": int(max(row["dynamic_particle_candidates"] for row in diag_rows)) if diag_rows else 0,
            "dynamic_triangle_candidates": int(np.max(triangle_candidates)) if triangle_candidates.size else 0,
            "self_candidates": int(np.max(self_candidates)) if self_candidates.size else 0,
            "pcg_guarded": int(max(row["pcg_guarded"] for row in diag_rows)) if diag_rows else 0,
            "pcg_max_delta": float(max(row["pcg_max_delta"] for row in diag_rows)) if diag_rows else 0.0,
            "jitter_stabilized_vertices": int(max(row["jitter_stabilized_vertices"] for row in diag_rows)) if diag_rows else 0,
            "jitter_rejected_vertices": int(max(row["jitter_rejected_vertices"] for row in diag_rows)) if diag_rows else 0,
        },
        "frames_with_particle_contacts": int(np.count_nonzero(particle_contacts > 0)),
        "frames_with_self_candidates": int(np.count_nonzero(self_candidates > 0)),
        "sample_diag_rows": diag_rows[:8] + diag_rows[-8:],
    }
    print("SSBL_WALK_JITTER_PROBE", json.dumps(result, ensure_ascii=False, sort_keys=True))
    ssbl.unregister()
    if bpy.app.background:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
