import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from wring_towel_smoke import (
    RADIAL_SEGMENTS,
    TWIST_RADIANS,
    WRING_FRAME_COUNT,
    _all_finite,
    _clear_scene,
    _make_towel,
    _max_abs_delta,
    _snapshot_coords,
    _step_wring_frame,
)


FRAME_COUNT = WRING_FRAME_COUNT
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "recordings" / "wring_towel_preview"
FRAMES_DIR = OUTPUT_DIR / "frames"
VIDEO_PATH = OUTPUT_DIR / "wring_towel_h0_no_pressure.mp4"


def _prepare_output():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for path in FRAMES_DIR.glob("*.png"):
        path.unlink()
    if VIDEO_PATH.exists():
        VIDEO_PATH.unlink()


def _material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def _setup_scene(scene, towel):
    towel.data.materials.append(_material("SSBL_Towel_Blue", (0.22, 0.42, 0.9, 1.0)))
    towel.data.materials.append(_material("SSBL_Towel_Light_Stripe", (0.76, 0.84, 1.0, 1.0)))
    towel.data.materials.append(_material("SSBL_Towel_Orange_Stripe", (1.0, 0.52, 0.18, 1.0)))
    for poly in towel.data.polygons:
        radial_index = int(poly.index) % RADIAL_SEGMENTS
        if radial_index in (0, 1, 2):
            poly.material_index = 2
        elif radial_index % 6 == 0:
            poly.material_index = 1
        else:
            poly.material_index = 0
    towel.show_wire = True

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 24
    scene.render.image_settings.file_format = "PNG"
    if scene.world is not None:
        scene.world.color = (0.025, 0.03, 0.035)

    camera_data = bpy.data.cameras.new("SSBL_Wring_Record_Camera")
    camera = bpy.data.objects.new("SSBL_Wring_Record_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (3.2, -5.2, 2.35)
    _look_at(camera, (0.0, 0.0, 0.8))
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 4.6
    scene.camera = camera

    light_data = bpy.data.lights.new("SSBL_Wring_Record_Key", "AREA")
    light = bpy.data.objects.new("SSBL_Wring_Record_Key", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, -3.0, 4.5)
    light_data.energy = 550
    light_data.size = 5


def _render_frame(scene, frame_index):
    scene.render.filepath = str(FRAMES_DIR / f"frame_{frame_index:04d}.png")
    bpy.ops.render.render(write_still=True)


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _encode_video():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        "24",
        "-i",
        str(FRAMES_DIR / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(VIDEO_PATH),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(VIDEO_PATH)


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    _prepare_output()
    _clear_scene()

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT + 2
    scene.frame_set(1)
    towel = _make_towel()
    _setup_scene(scene, towel)
    source_mesh = towel.data
    source_before = _snapshot_coords(source_mesh)

    settings = scene.ssbl_preview
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = "ssbl_pin"
    settings.hardness = 0.0
    settings.hardness_initialized = True
    settings.use_volume_pressure = False
    settings.self_collision = False
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 2
    settings.max_self_collision_neighbors = 32
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.static_collider_collection = None
    settings.collision_margin = 0.01
    settings.cloth_thickness = 0.02
    settings.substeps = 16
    settings.iterations = 4
    settings.damping = 0.995
    # Keep one spare preview frame so the last recorded step is not replaced by
    # the preview session auto-restore frame.
    settings.frame_count = FRAME_COUNT + 1

    session = ssbl.solver.start_preview(bpy.context, towel)
    frame_metrics = []
    _render_frame(scene, 0)
    finite = True
    min_z = float("inf")
    max_radius = 0.0
    for frame in range(1, FRAME_COUNT + 1):
        _step_wring_frame(session, towel, frame, FRAME_COUNT)
        finite = finite and _all_finite(towel)
        frame_min_z = float("inf")
        frame_max_radius = 0.0
        for vert in towel.data.vertices:
            world_z = float((towel.matrix_world @ vert.co).z)
            radius = math.hypot(float(vert.co.y), float(vert.co.z - 1.25))
            frame_min_z = min(frame_min_z, world_z)
            frame_max_radius = max(frame_max_radius, radius)
        min_z = min(min_z, frame_min_z)
        max_radius = max(max_radius, frame_max_radius)
        frame_metrics.append({"frame": frame, "min_z": frame_min_z, "max_twist_radius": frame_max_radius})
        _render_frame(scene, frame)

    tethers = len(session.cloth.lra_edges)
    ssbl.solver.request_stop(towel)
    source_after = _snapshot_coords(source_mesh)
    original_mesh_max_abs_delta = _max_abs_delta(source_before, source_after)
    video_path = _encode_video()

    summary = {
        "output_dir": str(OUTPUT_DIR),
        "frames_dir": str(FRAMES_DIR),
        "video_path": video_path,
        "hardness": float(settings.hardness),
        "use_volume_pressure": bool(settings.use_volume_pressure),
        "self_collision_mode": str(settings.self_collision_mode),
        "frame_count": FRAME_COUNT,
        "vertices": len(towel.data.vertices),
        "polygons": len(towel.data.polygons),
        "triangles": len(session.cloth.triangles),
        "tethers": tethers,
        "finite": finite,
        "min_z": min_z,
        "max_twist_radius": max_radius,
        "twist_degrees": math.degrees(TWIST_RADIANS),
        "original_mesh_max_abs_delta": original_mesh_max_abs_delta,
        "frames": [str(path) for path in sorted(FRAMES_DIR.glob("*.png"))],
        "frame_metrics": frame_metrics,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SSBL_WRING_TOWEL_RECORDING", str(summary_path))
    print(json.dumps(summary, ensure_ascii=False))

    if not finite:
        raise RuntimeError("wring towel recording produced non-finite vertex coordinates")
    if tethers != 0:
        raise RuntimeError(f"hardness=0 should not create hidden tether constraints, got {tethers}")
    if settings.use_volume_pressure:
        raise RuntimeError("wring towel recording must keep volume pressure disabled")
    if original_mesh_max_abs_delta > 1.0e-7:
        raise RuntimeError(f"preview did not restore the source mesh: {original_mesh_max_abs_delta}")
    if video_path is None or not VIDEO_PATH.exists():
        raise RuntimeError("ffmpeg did not produce the wring towel mp4")
    ssbl.unregister()


if __name__ == "__main__":
    main()
