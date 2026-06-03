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


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


FRAME_COUNT = max(int(os.environ.get("SSBL_CONTACT_RECORD_FRAMES", "32")), 1)
GRID_SUBDIVISIONS = max(int(os.environ.get("SSBL_CONTACT_RECORD_GRID", "25")), 4)
OUTPUT_DIR = Path(os.environ.get(
    "SSBL_CONTACT_RECORD_DIR",
    Path(__file__).resolve().parents[1] / "recordings" / "multicloth_contact_after_fix",
))


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def _configure_settings(settings) -> None:
    settings.enabled = True
    settings.pin_vertex_group = ""
    settings.use_evaluated_mesh = True
    settings.preview_writeback_interval = 1
    settings.preview_target_fps = 60.0
    settings.dt = 1.0 / 60.0
    settings.substeps = 6
    settings.iterations = 2
    settings.frame_count = FRAME_COUNT + 4
    settings.damping = 0.98
    settings.gravity = (0.0, 0.0, 0.0)
    settings.hardness = 0.55
    settings.self_collision = False
    settings.self_collision_mode = "off"
    settings.use_volume_pressure = False
    settings.collision_margin = 0.008
    settings.cloth_thickness = 0.05
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.sphere_object = None
    settings.static_collider_collection = None


def _make_scene() -> tuple[bpy.types.Object, bpy.types.Object]:
    cloth_mat = _material("SSBL_Contact_Cloth_Blue", (0.14, 0.38, 0.95, 1.0))
    sphere_mat = _material("SSBL_Contact_Sphere_Orange", (1.0, 0.46, 0.12, 1.0))
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=GRID_SUBDIVISIONS,
        y_subdivisions=GRID_SUBDIVISIONS,
        size=1.8,
        location=(0.0, 0.0, 0.0),
    )
    cloth = bpy.context.object
    cloth.name = "SSBL_Record_Contact_Cloth"
    cloth.data.materials.append(cloth_mat)
    _configure_settings(cloth.ssbl_cloth)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.42, location=(0.0, 0.0, 0.36))
    sphere = bpy.context.object
    sphere.name = "SSBL_Record_Contact_SphereCloth"
    sphere.data.materials.append(sphere_mat)
    _configure_settings(sphere.ssbl_cloth)
    sphere.ssbl_cloth.hardness = 0.8
    sphere.ssbl_cloth.use_volume_pressure = True
    sphere.ssbl_cloth.volume_compliance = 1.0e-6
    sphere.ssbl_cloth.pressure_strength = 0.25

    bpy.ops.object.light_add(type="AREA", location=(0.0, -2.8, 3.2))
    light = bpy.context.object
    light.name = "SSBL_Record_KeyLight"
    light.data.energy = 450.0
    light.data.size = 4.0

    bpy.ops.object.camera_add(location=(2.2, -3.0, 1.7))
    camera = bpy.context.object
    _look_at(camera, Vector((0.0, 0.0, 0.22)))
    bpy.context.scene.camera = camera

    return cloth, sphere


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


def _render_frame(index: int) -> Path:
    scene = bpy.context.scene
    scene.frame_set(index + 1)
    frame_path = OUTPUT_DIR / "frames" / f"frame_{index + 1:04d}.png"
    scene.render.filepath = str(frame_path)
    bpy.ops.render.render(write_still=True)
    return frame_path


def _encode_mp4() -> Path | None:
    mp4_path = OUTPUT_DIR / "multicloth_contact_after_fix.mp4"
    frames_pattern = OUTPUT_DIR / "frames" / "frame_%04d.png"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        "12",
        "-i",
        str(frames_pattern),
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(mp4_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None
    return mp4_path


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "frames").mkdir(parents=True, exist_ok=True)
        _clear_scene()
        scene = bpy.context.scene
        for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
            try:
                scene.render.engine = engine
                break
            except Exception:
                continue
        scene.render.resolution_x = 960
        scene.render.resolution_y = 720
        scene.render.fps = 12
        scene.view_settings.view_transform = "Standard"
        cloth, sphere = _make_scene()
        bpy.ops.object.select_all(action="DESELECT")
        cloth.select_set(True)
        sphere.select_set(True)
        bpy.context.view_layer.objects.active = cloth
        session = ssbl.solver.start_preview(bpy.context, cloth)
        max_edge_ratio = 0.0
        finite = True
        for index in range(FRAME_COUNT):
            ssbl.solver.step_preview(bpy.context, cloth.name)
            diagnostics = ssbl.solver.session_diagnostics(cloth)
            finite = finite and bool(diagnostics.finite)
            for slot in session.slots.values():
                max_edge_ratio = max(max_edge_ratio, _slot_max_edge_ratio(slot))
            _render_frame(index)
        diagnostics = ssbl.solver.session_diagnostics(cloth)
        ssbl.solver.request_stop(cloth)
        mp4_path = _encode_mp4()
        summary = {
            "output_dir": str(OUTPUT_DIR),
            "mp4": str(mp4_path) if mp4_path is not None else None,
            "frames": FRAME_COUNT,
            "slots": len(session.slots),
            "cross_mode": str(session.cross_cloth_mode),
            "finite": bool(finite),
            "max_edge_ratio": float(max_edge_ratio),
            "dynamic_triangle_count": int(diagnostics.dynamic_triangle_count),
            "dynamic_collision_ms": float(diagnostics.dynamic_collision_ms),
            "dynamic_upload_ms": float(diagnostics.dynamic_upload_ms),
        }
        summary_path = OUTPUT_DIR / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print("SSBL_MULTICLOTH_CONTACT_RECORD", json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if not finite or len(session.slots) != 2 or str(session.cross_cloth_mode) != "all_selected":
            raise RuntimeError(f"Contact recording validation failed: {summary}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
