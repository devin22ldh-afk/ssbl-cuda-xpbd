import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_if_present(settings: object, name: str, value: object) -> None:
    if hasattr(settings, name):
        setattr(settings, name, value)


def _find_cloth() -> bpy.types.Object:
    active = bpy.context.active_object
    if active is not None and active.type == "MESH":
        return active
    selected = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if selected:
        return selected[0]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh object found in current scene")
    return max(meshes, key=lambda obj: len(obj.data.vertices))


def _ensure_camera(scene: bpy.types.Scene, obj: bpy.types.Object) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x = _int_env("SSBL_EXPLOSION_RES_X", 1280)
    scene.render.resolution_y = _int_env("SSBL_EXPLOSION_RES_Y", 720)
    scene.render.fps = _int_env("SSBL_EXPLOSION_VIDEO_FPS", 12)
    if scene.world is not None:
        scene.world.color = (0.025, 0.03, 0.035)
    if scene.camera is not None:
        return
    center = obj.location
    camera_data = bpy.data.cameras.new("SSBL_Explosion_Diag_Camera")
    camera = bpy.data.objects.new("SSBL_Explosion_Diag_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (center.x, center.y - 5.5, center.z + 1.5)
    camera.rotation_euler = (math.radians(75.0), 0.0, 0.0)
    camera_data.lens = 45
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


def _encode_video(frames_dir: Path, video_path: Path, fps: int) -> str | None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
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


def _signed_volume(positions: np.ndarray, triangles: np.ndarray) -> float:
    tri = positions[triangles]
    return float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0)


def _area_ratios(positions: np.ndarray, rest_positions: np.ndarray, triangles: np.ndarray) -> tuple[float, float]:
    current = positions[triangles]
    rest = rest_positions[triangles]
    current_area = 0.5 * np.linalg.norm(np.cross(current[:, 1] - current[:, 0], current[:, 2] - current[:, 0]), axis=1)
    rest_area = 0.5 * np.linalg.norm(np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), axis=1)
    valid = rest_area > 1.0e-10
    if not np.any(valid):
        return 0.0, 0.0
    ratio = current_area[valid] / rest_area[valid]
    return float(np.percentile(ratio, 99.0)), float(np.max(ratio))


def _metrics(
    frame: int,
    obj: bpy.types.Object,
    positions: np.ndarray,
    rest_positions: np.ndarray,
    prev_positions: np.ndarray,
    edges: np.ndarray,
    edge_rest: np.ndarray,
    triangles: np.ndarray,
    rest_volume: float,
) -> dict[str, object]:
    finite = bool(np.isfinite(positions).all())
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    step_disp = np.linalg.norm(positions - prev_positions, axis=1)
    rest_disp = np.linalg.norm(positions - rest_positions, axis=1)
    edge_vec = positions[edges[:, 0]] - positions[edges[:, 1]]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    valid_edges = edge_rest > 1.0e-10
    edge_ratio = edge_len[valid_edges] / edge_rest[valid_edges]
    volume = _signed_volume(positions, triangles)
    p99_area_ratio, max_area_ratio = _area_ratios(positions, rest_positions, triangles)
    diag = ssbl.solver.session_diagnostics(obj)
    return {
        "sim_frame": int(frame),
        "scene_frame": int(bpy.context.scene.frame_current),
        "finite": finite,
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "bbox_extent": (bbox_max - bbox_min).astype(float).tolist(),
        "max_step_disp": float(np.max(step_disp)),
        "p99_step_disp": float(np.percentile(step_disp, 99.0)),
        "max_rest_disp": float(np.max(rest_disp)),
        "max_edge_len": float(np.max(edge_len)),
        "p99_edge_ratio": float(np.percentile(edge_ratio, 99.0)) if len(edge_ratio) else 0.0,
        "max_edge_ratio": float(np.max(edge_ratio)) if len(edge_ratio) else 0.0,
        "signed_volume": volume,
        "volume_ratio": float(volume / rest_volume) if abs(rest_volume) > 1.0e-10 else None,
        "p99_area_ratio": p99_area_ratio,
        "max_area_ratio": max_area_ratio,
        "step_ms": float(diag.step_ms),
        "self_solve_ms": float(diag.self_solve_ms),
        "self_probe_ms": float(diag.self_probe_ms),
        "self_recovery_ms": float(diag.self_recovery_ms),
        "candidate_count": int(diag.candidate_count),
        "resolved_contacts": int(diag.resolved_contacts),
        "min_gap": None if diag.min_gap is None else float(diag.min_gap),
        "penetration_depth": float(diag.penetration_depth),
        "ccd_clamp_count": int(diag.ccd_clamp_count),
        "recovery_passes": int(diag.recovery_passes),
        "local_retry_count": int(diag.local_retry_count),
        "finite_flag": bool(diag.finite),
        "self_active_regions": int(diag.self_active_regions),
        "self_sleeping_regions": int(diag.self_sleeping_regions),
        "self_compaction_used": int(diag.self_compaction_used),
        "self_vs_pair_used": int(diag.self_vs_pair_compaction_used),
        "jitter_stabilized_vertices": int(diag.jitter_stabilized_vertices),
        "jitter_rejected_vertices": int(diag.jitter_rejected_vertices),
        "jitter_max_correction": float(diag.jitter_max_correction),
    }


def _is_anomaly(item: dict[str, object], previous: dict[str, object] | None) -> bool:
    if not bool(item["finite"]) or not bool(item["finite_flag"]):
        return True
    if float(item["max_edge_ratio"]) > 5.0 or float(item["max_area_ratio"]) > 25.0:
        return True
    if float(item["max_step_disp"]) > 0.25:
        return True
    if previous is not None:
        prev_extent = np.asarray(previous["bbox_extent"], dtype=np.float32)
        extent = np.asarray(item["bbox_extent"], dtype=np.float32)
        if float(np.max(extent / np.maximum(prev_extent, 1.0e-6))) > 1.5:
            return True
    return False


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    out_root = Path(os.environ.get("SSBL_EXPLOSION_OUTPUT_DIR", "")) if os.environ.get("SSBL_EXPLOSION_OUTPUT_DIR") else Path(tempfile.gettempdir()) / "ssbl_current_scene_explosion_diag"
    frames_dir = out_root / "frames"
    out_root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("frame_*.png"):
        path.unlink()

    scene = bpy.context.scene
    scene.frame_set(int(scene.frame_start))
    obj = _find_cloth()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    settings = scene.ssbl_preview
    original_writeback_interval = int(getattr(settings, "preview_writeback_interval", 1))
    original_overrides = {
        "use_volume_pressure": bool(getattr(settings, "use_volume_pressure", False)),
        "self_collision": bool(getattr(settings, "self_collision", False)),
        "self_collision_mode": str(getattr(settings, "self_collision_mode", "")),
        "self_probe_interval": int(getattr(settings, "self_probe_interval", 1)),
        "self_sleep_enabled": bool(getattr(settings, "self_sleep_enabled", False)),
        "self_compaction_enabled": bool(getattr(settings, "self_compaction_enabled", False)),
        "self_pair_compaction_enabled": bool(getattr(settings, "self_pair_compaction_enabled", False)),
        "self_sleep_motion_scale": float(getattr(settings, "self_sleep_motion_scale", 1.0)),
    }
    settings.preview_writeback_interval = 1
    self_collision_mode = os.environ.get("SSBL_EXPLOSION_SELF_COLLISION_MODE")
    if self_collision_mode is not None:
        _set_if_present(settings, "self_collision_mode", self_collision_mode)
        _set_if_present(settings, "self_collision", self_collision_mode.strip().lower() != "off")
    else:
        _set_if_present(
            settings,
            "self_collision",
            _bool_env("SSBL_EXPLOSION_SELF_COLLISION", bool(getattr(settings, "self_collision", False))),
        )
    _set_if_present(
        settings,
        "use_volume_pressure",
        _bool_env("SSBL_EXPLOSION_VOLUME_PRESSURE", bool(getattr(settings, "use_volume_pressure", False))),
    )
    _set_if_present(
        settings,
        "self_probe_interval",
        _int_env("SSBL_EXPLOSION_SELF_PROBE_INTERVAL", int(getattr(settings, "self_probe_interval", 1))),
    )
    _set_if_present(
        settings,
        "self_sleep_enabled",
        _bool_env("SSBL_EXPLOSION_SELF_SLEEP_ENABLED", bool(getattr(settings, "self_sleep_enabled", False))),
    )
    _set_if_present(
        settings,
        "self_compaction_enabled",
        _bool_env("SSBL_EXPLOSION_SELF_COMPACTION_ENABLED", bool(getattr(settings, "self_compaction_enabled", False))),
    )
    _set_if_present(
        settings,
        "self_pair_compaction_enabled",
        _bool_env("SSBL_EXPLOSION_SELF_PAIR_COMPACTION_ENABLED", bool(getattr(settings, "self_pair_compaction_enabled", False))),
    )
    _set_if_present(
        settings,
        "self_sleep_motion_scale",
        _float_env("SSBL_EXPLOSION_SELF_SLEEP_MOTION_SCALE", float(getattr(settings, "self_sleep_motion_scale", 1.0))),
    )
    effective_settings = {
        "frame_count": int(getattr(settings, "frame_count", 0)),
        "substeps": int(getattr(settings, "substeps", 0)),
        "iterations": int(getattr(settings, "iterations", 0)),
        "self_collision_mode": str(getattr(settings, "self_collision_mode", "")),
        "self_collision": bool(getattr(settings, "self_collision", False)),
        "use_volume_pressure": bool(getattr(settings, "use_volume_pressure", False)),
        "cloth_thickness": float(getattr(settings, "cloth_thickness", 0.0)),
        "collision_margin": float(getattr(settings, "collision_margin", 0.0)),
        "static_collider_collection": None
        if getattr(settings, "static_collider_collection", None) is None
        else settings.static_collider_collection.name,
        "self_sleep_enabled": bool(getattr(settings, "self_sleep_enabled", False)),
        "self_compaction_enabled": bool(getattr(settings, "self_compaction_enabled", False)),
        "self_pair_compaction_enabled": bool(getattr(settings, "self_pair_compaction_enabled", False)),
        "self_sleep_motion_scale": float(getattr(settings, "self_sleep_motion_scale", 1.0)),
        "self_probe_interval": int(getattr(settings, "self_probe_interval", 1)),
    }
    _ensure_camera(scene, obj)

    session = ssbl.solver.start_preview(bpy.context, obj)
    slot = session.slots[session.object_name]
    rest_positions = np.asarray(slot.cloth.positions_world, dtype=np.float32)
    edges = np.asarray(slot.cloth.edges, dtype=np.int32)
    edge_rest = np.asarray(slot.cloth.edge_rest_lengths, dtype=np.float32)
    triangles = np.asarray(slot.cloth.triangles, dtype=np.int32)
    rest_volume = float(slot.cloth.rest_volume)

    steps = _int_env("SSBL_EXPLOSION_STEPS", min(max(int(getattr(settings, "frame_count", 120)), 1), 600))
    render_stride = max(_int_env("SSBL_EXPLOSION_RENDER_STRIDE", 10), 1)
    fps = max(_int_env("SSBL_EXPLOSION_VIDEO_FPS", 12), 1)
    rendered = []
    metrics = []
    anomaly_frames = []
    render_index = 0
    prev_positions = rest_positions.copy()
    previous_metric = None

    render_enabled = _bool_env("SSBL_EXPLOSION_RENDER", True)
    if render_enabled:
        rendered.append({"sim_frame": 0, "path": _render_frame(scene, frames_dir, render_index), "reason": "start"})
        render_index += 1

    try:
        for frame in range(1, steps + 1):
            finished = ssbl.solver.step_preview(bpy.context, obj.name)
            positions = np.asarray(slot.current_positions_world, dtype=np.float32)
            item = _metrics(
                frame,
                obj,
                positions,
                rest_positions,
                prev_positions,
                edges,
                edge_rest,
                triangles,
                rest_volume,
            )
            metrics.append(item)
            anomaly = _is_anomaly(item, previous_metric)
            if anomaly:
                anomaly_frames.append(item)
            if render_enabled and (frame == 1 or frame % render_stride == 0 or anomaly):
                rendered.append(
                    {
                        "sim_frame": frame,
                        "path": _render_frame(scene, frames_dir, render_index),
                        "reason": "anomaly" if anomaly else "sample",
                    }
                )
                render_index += 1
            prev_positions = positions.copy()
            previous_metric = item
            if finished:
                break
    finally:
        ssbl.solver.request_stop(obj)
        settings.preview_writeback_interval = original_writeback_interval
        for name, value in original_overrides.items():
            _set_if_present(settings, name, value)

    video_path = _encode_video(frames_dir, out_root / "current_scene_explosion_diag.mp4", fps) if render_enabled else None
    worst_edge = max(metrics, key=lambda item: float(item["max_edge_ratio"])) if metrics else None
    worst_step = max(metrics, key=lambda item: float(item["max_step_disp"])) if metrics else None
    worst_area = max(metrics, key=lambda item: float(item["max_area_ratio"])) if metrics else None
    min_volume = min(metrics, key=lambda item: abs(float(item["volume_ratio"] or 0.0))) if metrics else None
    summary = {
        "blend": bpy.data.filepath,
        "object": obj.name,
        "settings": effective_settings,
        "overrides": {
            "SSBL_EXPLOSION_VOLUME_PRESSURE": os.environ.get("SSBL_EXPLOSION_VOLUME_PRESSURE"),
            "SSBL_EXPLOSION_SELF_COLLISION": os.environ.get("SSBL_EXPLOSION_SELF_COLLISION"),
            "SSBL_EXPLOSION_SELF_COLLISION_MODE": os.environ.get("SSBL_EXPLOSION_SELF_COLLISION_MODE"),
            "SSBL_EXPLOSION_SELF_PROBE_INTERVAL": os.environ.get("SSBL_EXPLOSION_SELF_PROBE_INTERVAL"),
            "SSBL_EXPLOSION_SELF_SLEEP_ENABLED": os.environ.get("SSBL_EXPLOSION_SELF_SLEEP_ENABLED"),
            "SSBL_EXPLOSION_SELF_COMPACTION_ENABLED": os.environ.get("SSBL_EXPLOSION_SELF_COMPACTION_ENABLED"),
            "SSBL_EXPLOSION_SELF_PAIR_COMPACTION_ENABLED": os.environ.get("SSBL_EXPLOSION_SELF_PAIR_COMPACTION_ENABLED"),
            "SSBL_EXPLOSION_SELF_SLEEP_MOTION_SCALE": os.environ.get("SSBL_EXPLOSION_SELF_SLEEP_MOTION_SCALE"),
        },
        "vertex_count": int(len(rest_positions)),
        "triangle_count": int(len(triangles)),
        "rest_volume": rest_volume,
        "steps_recorded": int(len(metrics)),
        "video": video_path,
        "frames_dir": str(frames_dir),
        "rendered_frames": rendered,
        "anomaly_frame_count": int(len(anomaly_frames)),
        "first_anomaly": anomaly_frames[0] if anomaly_frames else None,
        "worst_edge_frame": worst_edge,
        "worst_step_frame": worst_step,
        "worst_area_frame": worst_area,
        "min_abs_volume_ratio_frame": min_volume,
        "metrics": metrics,
    }
    summary_path = out_root / "diagnosis.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SSBL_EXPLOSION_DIAGNOSIS", str(summary_path))
    print(json.dumps({key: summary[key] for key in summary if key != "metrics"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
