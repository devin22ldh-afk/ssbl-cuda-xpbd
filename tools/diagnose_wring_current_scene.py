import json
import math
import os
import struct
from pathlib import Path

import bpy
import numpy as np


OBJECT_NAME = "SSBL_Wring_Towel_CurrentScene"
OUT_DIR = Path(
    r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\ssbl\recordings\wring_towel_current_scene_diagnosis"
)
SAMPLE_FRAMES = (1, 20, 40, 60, 80, 120, 226)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _mesh_cache_modifier_info(obj):
    infos = []
    for modifier in obj.modifiers:
        info = {
            "name": modifier.name,
            "type": modifier.type,
            "show_viewport": bool(modifier.show_viewport),
            "show_render": bool(modifier.show_render),
        }
        if modifier.type == "MESH_CACHE":
            path = bpy.path.abspath(modifier.filepath)
            info.update(
                {
                    "cache_format": getattr(modifier, "cache_format", ""),
                    "filepath": path,
                    "filepath_exists": bool(path and Path(path).exists()),
                    "frame_start": float(getattr(modifier, "frame_start", 0.0)),
                    "frame_scale": float(getattr(modifier, "frame_scale", 0.0)),
                }
            )
        infos.append(info)
    return infos


def _read_pc2_header(path):
    if not path or not Path(path).exists():
        return None
    with open(path, "rb") as handle:
        data = handle.read(32)
    if len(data) != 32:
        return {"valid": False, "size": Path(path).stat().st_size, "reason": "short header"}
    signature, version, vertex_count, start_frame, sample_rate, sample_count = struct.unpack(
        "<12siiffi", data
    )
    expected_size = 32 + int(vertex_count) * 3 * 4 * int(sample_count)
    actual_size = Path(path).stat().st_size
    return {
        "valid": signature.startswith(b"POINTCACHE2"),
        "version": int(version),
        "vertex_count": int(vertex_count),
        "start_frame": float(start_frame),
        "sample_rate": float(sample_rate),
        "sample_count": int(sample_count),
        "expected_size": int(expected_size),
        "actual_size": int(actual_size),
        "size_matches": int(expected_size) == int(actual_size),
    }


def _evaluated_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        points = np.array([obj.matrix_world @ vert.co for vert in mesh.vertices], dtype=np.float32)
    finally:
        evaluated.to_mesh_clear()
    return points


def _ring_twist_metric(points):
    # The towel mesh is generated as x-rings with 24 radial vertices per ring.
    radial_segments = 24
    if len(points) % radial_segments != 0:
        return None
    ring_count = len(points) // radial_segments
    if ring_count < 2:
        return None
    rings = points.reshape((ring_count, radial_segments, 3))
    left = rings[0, 0]
    right = rings[-1, 0]
    center_left = np.mean(rings[0, :, 1:3], axis=0)
    center_right = np.mean(rings[-1, :, 1:3], axis=0)
    left_angle = math.atan2(float(left[2] - center_left[1]), float(left[1] - center_left[0]))
    right_angle = math.atan2(float(right[2] - center_right[1]), float(right[1] - center_right[0]))
    delta = math.atan2(math.sin(right_angle - left_angle), math.cos(right_angle - left_angle))
    return math.degrees(delta)


def _sample_frames(obj):
    samples = {}
    reference = None
    for frame in SAMPLE_FRAMES:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        points = _evaluated_vertices(obj)
        if reference is None:
            reference = points.copy()
        delta = points - reference
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        samples[str(frame)] = {
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "max_abs_delta_from_frame_1": float(np.max(np.abs(delta))),
            "mean_abs_delta_from_frame_1": float(np.mean(np.abs(delta))),
            "finite": bool(np.isfinite(points).all()),
            "ring_endpoint_twist_degrees": _ring_twist_metric(points),
        }
    return samples


def _configure_render(scene):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.light = "STUDIO"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 24
    scene.eevee.taa_render_samples = 16 if hasattr(scene, "eevee") else 16


def _render_stills(scene, out_dir):
    still_dir = out_dir / "frames"
    still_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in (1, 20, 40, 60, 80):
        scene.frame_set(frame)
        scene.render.filepath = str(still_dir / f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        paths.append(scene.render.filepath)
    return paths


def _render_video(scene, out_dir):
    sequence_dir = out_dir / "sequence"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    scene.frame_start = 1
    scene.frame_end = 81
    scene.render.image_settings.file_format = "JPEG"
    paths = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        scene.render.filepath = str(sequence_dir / f"frame_{frame:04d}.jpg")
        bpy.ops.render.render(write_still=True)
        paths.append(scene.render.filepath)
    return {
        "sequence_dir": str(sequence_dir),
        "frame_pattern": str(sequence_dir / "frame_%04d.jpg"),
        "expected_video": str(out_dir / "current_scene_cache_playback.mp4"),
        "frame_count": len(paths),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    obj = bpy.data.objects.get(OBJECT_NAME)
    if obj is None:
        raise RuntimeError(f"Missing object: {OBJECT_NAME}")

    modifiers = _mesh_cache_modifier_info(obj)
    mesh_cache = next((item for item in modifiers if item.get("type") == "MESH_CACHE"), None)
    pc2_header = _read_pc2_header(mesh_cache.get("filepath") if mesh_cache else None)
    samples = _sample_frames(obj)

    _configure_render(scene)
    stills = _render_stills(scene, OUT_DIR)
    video = _render_video(scene, OUT_DIR)

    settings = getattr(scene, "ssbl_preview", None)
    result = {
        "blend": bpy.data.filepath,
        "scene_frame_before_video": int(scene.frame_current),
        "object": obj.name,
        "object_mode": obj.mode,
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "modifiers": modifiers,
        "pc2_header": pc2_header,
        "samples": samples,
        "ssbl_settings": {
            "hardness": float(getattr(settings, "hardness", -1.0)) if settings else None,
            "use_volume_pressure": bool(getattr(settings, "use_volume_pressure", False))
            if settings
            else None,
            "pin_vertex_group": str(getattr(settings, "pin_vertex_group", "")) if settings else None,
            "substeps": int(getattr(settings, "substeps", 0)) if settings else None,
            "iterations": int(getattr(settings, "iterations", 0)) if settings else None,
        },
        "stills": stills,
        "video": video,
    }
    summary_path = OUT_DIR / "diagnosis.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_json_safe), encoding="utf-8")
    print("SSBL_WRING_CURRENT_SCENE_DIAGNOSIS", json.dumps(result, ensure_ascii=False, default=_json_safe))


if __name__ == "__main__":
    main()
