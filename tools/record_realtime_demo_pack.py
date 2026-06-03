from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
SSBL_ROOT = TOOLS_DIR.parent
ADDONS_ROOT = SSBL_ROOT.parent
if str(ADDONS_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDONS_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import ssbl
from ssbl.xpbd_core import to_local
from wring_towel_smoke import (
    RADIAL_SEGMENTS,
    TWIST_RADIANS,
    WRING_FRAME_COUNT,
    _all_finite as _wring_all_finite,
    _make_towel,
    _max_abs_delta,
    _snapshot_coords,
    _step_wring_frame,
)


OUTPUT_ROOT = SSBL_ROOT / "recordings" / "realtime_demo_2026-06-03"
RESOLUTION = (1280, 720)
VIDEO_FPS = 24
RESTORE_TOLERANCE = 1.0e-7


@dataclass
class Overlay:
    title: bpy.types.Object
    metrics: bpy.types.Object
    notes: bpy.types.Object


@dataclass
class DemoResult:
    name: str
    title: str
    video: str
    frames_dir: str
    frame_count: int
    simulation_steps: int
    simulation_elapsed_s: float
    average_simulation_fps: float
    average_step_ms: float
    p95_step_ms: float
    finite: bool
    restore_delta: float
    validation_passed: bool
    metrics: dict[str, object] = field(default_factory=dict)
    ffprobe: dict[str, object] = field(default_factory=dict)
    representative_frames: list[str] = field(default_factory=list)


def _clear_scene() -> None:
    try:
        ssbl.solver.cleanup_all_sessions()
    except Exception:
        pass
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.users == 0 or collection.name.startswith("SSBL_"):
            bpy.data.collections.remove(collection)


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def _look_at(obj: bpy.types.Object, target: Vector | tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _configure_scene_render(
    scene: bpy.types.Scene,
    *,
    camera_location: tuple[float, float, float],
    target: tuple[float, float, float],
    ortho_scale: float,
) -> bpy.types.Object:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.fps = VIDEO_FPS
    scene.render.image_settings.file_format = "PNG"
    if scene.world is not None:
        scene.world.color = (0.02, 0.024, 0.03)

    camera_data = bpy.data.cameras.new("SSBL_Demo_Camera")
    camera = bpy.data.objects.new("SSBL_Demo_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = camera_location
    _look_at(camera, target)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    scene.camera = camera

    light_data = bpy.data.lights.new("SSBL_Demo_KeyLight", "AREA")
    light = bpy.data.objects.new("SSBL_Demo_KeyLight", light_data)
    scene.collection.objects.link(light)
    light.location = (camera_location[0] * 0.4, camera_location[1] * 0.35, camera_location[2] + 2.5)
    light_data.energy = 550
    light_data.size = 5.0
    return camera


def _camera_plane(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    *,
    name: str,
    location: tuple[float, float, float],
    size: tuple[float, float],
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    half_x = size[0] * 0.5
    half_y = size[1] * 0.5
    center = Vector(location)
    corners = [
        camera.matrix_world @ (center + Vector((-half_x, -half_y, 0.0))),
        camera.matrix_world @ (center + Vector((half_x, -half_y, 0.0))),
        camera.matrix_world @ (center + Vector((half_x, half_y, 0.0))),
        camera.matrix_world @ (center + Vector((-half_x, half_y, 0.0))),
    ]
    mesh.from_pydata(
        [tuple(corner) for corner in corners],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(_material(f"{name}_Mat", color))
    scene.collection.objects.link(obj)
    return obj


def _camera_text(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    *,
    name: str,
    body: str,
    location: tuple[float, float, float],
    size: float,
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    curve = bpy.data.curves.new(f"{name}_Curve", "FONT")
    curve.body = body
    curve.align_x = "LEFT"
    curve.align_y = "TOP"
    curve.size = size
    curve.space_line = 0.9
    obj = bpy.data.objects.new(name, curve)
    obj.location = camera.matrix_world @ Vector(location)
    direction = camera.location - obj.location
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(_material(f"{name}_Mat", color))
    scene.collection.objects.link(obj)
    return obj


def _create_overlay(scene: bpy.types.Scene, camera: bpy.types.Object, title: str) -> Overlay:
    height = float(camera.data.ortho_scale)
    width = height * RESOLUTION[0] / RESOLUTION[1]
    left = -width * 0.5 + 0.18
    top = height * 0.5 - 0.56
    _camera_plane(
        scene,
        camera,
        name="SSBL_Demo_Overlay_Backdrop",
        location=(left + 2.32, top - 0.34, -4.06),
        size=(4.95, 0.86),
        color=(0.015, 0.018, 0.024, 1.0),
    )
    title_obj = _camera_text(
        scene,
        camera,
        name="SSBL_Demo_Overlay_Title",
        body=title,
        location=(left, top, -4.0),
        size=0.13,
        color=(0.95, 0.98, 1.0, 1.0),
    )
    metrics_obj = _camera_text(
        scene,
        camera,
        name="SSBL_Demo_Overlay_Metrics",
        body="",
        location=(left, top - 0.27, -4.0),
        size=0.092,
        color=(0.77, 0.93, 1.0, 1.0),
    )
    notes_obj = _camera_text(
        scene,
        camera,
        name="SSBL_Demo_Overlay_Notes",
        body="",
        location=(left, top - 0.53, -4.0),
        size=0.08,
        color=(1.0, 0.84, 0.52, 1.0),
    )
    return Overlay(title_obj, metrics_obj, notes_obj)


def _update_overlay(
    overlay: Overlay,
    *,
    frame: int,
    total_frames: int,
    average_fps: float,
    last_step_ms: float,
    native_ms: float,
    note: str,
) -> None:
    overlay.metrics.data.body = (
        f"frame {frame:03d}/{total_frames:03d}   "
        f"sim FPS {average_fps:5.1f}   "
        f"step {last_step_ms:5.2f} ms   "
        f"native {native_ms:5.2f} ms"
    )
    overlay.notes.data.body = note
    bpy.context.view_layer.update()


def _render_frame(scene: bpy.types.Scene, frames_dir: Path, index: int) -> str:
    path = frames_dir / f"frame_{index:04d}.png"
    scene.render.filepath = str(path)
    if bpy.app.background:
        bpy.ops.render.render(write_still=True)
    else:
        try:
            bpy.ops.render.opengl(write_still=True, view_context=False)
        except RuntimeError:
            bpy.ops.render.render(write_still=True)
    return str(path)


def _ensure_output_dir(name: str) -> tuple[Path, Path, Path]:
    case_dir = OUTPUT_ROOT / name
    frames_dir = case_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    legacy_video_path = case_dir / f"{name}.mp4"
    if legacy_video_path.exists():
        legacy_video_path.unlink()
    video_path = OUTPUT_ROOT / f"{name}.mp4"
    if video_path.exists():
        video_path.unlink()
    return case_dir, frames_dir, video_path


def _run_checked(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _drawtext_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
    )


def _encode_video(frames_dir: Path, video_path: Path, overlay_lines: list[str]) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    filters = ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    fontfile = "/Windows/Fonts/arialbd.ttf"
    y = 24
    sizes = [30, 25, 23]
    colors = ["white", "0xC7F1FFFF", "0xFFD582FF"]
    for index, line in enumerate(overlay_lines):
        filters.append(
            "drawtext="
            f"fontfile={fontfile}:"
            f"text='{_drawtext_escape(line)}':"
            f"x=28:y={y}:"
            f"fontsize={sizes[min(index, len(sizes) - 1)]}:"
            f"fontcolor={colors[min(index, len(colors) - 1)]}:"
            "box=1:boxcolor=black@0.64:boxborderw=10"
        )
        y += 38 if index == 0 else 33
    _run_checked(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(VIDEO_FPS),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-vf",
            ",".join(filters),
            str(video_path),
        ]
    )
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg did not produce a non-empty video: {video_path}")
    return str(video_path)


def _ffprobe(video_path: Path) -> dict[str, object]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe was not found in PATH")
    proc = _run_checked(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream: {video_path}")
    stream = streams[0]
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise RuntimeError(f"ffprobe returned invalid dimensions: {stream}")
    return stream


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(math.ceil(len(ordered) * 0.95)) - 1)
    return ordered[index]


def _finite_mesh(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _mesh_snapshot(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in obj.data.vertices]


def _mesh_delta(obj: bpy.types.Object, before: list[tuple[float, float, float]]) -> float:
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


def _summarize_demo(
    *,
    name: str,
    title: str,
    video_path: Path,
    frames_dir: Path,
    frame_paths: list[str],
    step_ms_samples: list[float],
    simulation_elapsed: float,
    finite: bool,
    restore_delta: float,
    metrics: dict[str, object],
) -> DemoResult:
    steps = len(step_ms_samples)
    average_fps = steps / max(float(simulation_elapsed), 1.0e-9)
    average_ms = sum(step_ms_samples) / max(steps, 1)
    probe = _ffprobe(video_path)
    validation_passed = bool(finite and restore_delta <= RESTORE_TOLERANCE and video_path.exists() and video_path.stat().st_size > 0)
    if not validation_passed:
        raise RuntimeError(
            f"{name} validation failed: finite={finite} restore_delta={restore_delta} video={video_path}"
        )
    picks = []
    if frame_paths:
        picks = [frame_paths[0], frame_paths[len(frame_paths) // 2], frame_paths[-1]]
    return DemoResult(
        name=name,
        title=title,
        video=str(video_path),
        frames_dir=str(frames_dir),
        frame_count=len(frame_paths),
        simulation_steps=steps,
        simulation_elapsed_s=float(simulation_elapsed),
        average_simulation_fps=float(average_fps),
        average_step_ms=float(average_ms),
        p95_step_ms=float(_p95(step_ms_samples)),
        finite=bool(finite),
        restore_delta=float(restore_delta),
        validation_passed=True,
        metrics=metrics,
        ffprobe=probe,
        representative_frames=picks,
    )


def _metrics_overlay(step_ms_samples: list[float], simulation_elapsed: float) -> str:
    steps = len(step_ms_samples)
    average_fps = steps / max(float(simulation_elapsed), 1.0e-9)
    return f"sim FPS {average_fps:.1f} | p95 step {_p95(step_ms_samples):.1f} ms | frame %{{n}}"


def _add_world_label(text: str, location: tuple[float, float, float], camera: bpy.types.Object, mat: bpy.types.Material) -> None:
    curve = bpy.data.curves.new(f"{text}_Curve", "FONT")
    curve.body = text
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = 0.13
    obj = bpy.data.objects.new(f"{text}_Label", curve)
    obj.location = location
    direction = camera.location - obj.location
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)


def _apply_world_positions(obj: bpy.types.Object, positions_world: np.ndarray, matrix_world_inv) -> None:
    local = to_local(np.asarray(positions_world, dtype=np.float32), matrix_world_inv)
    obj.data.vertices.foreach_set("co", np.asarray(local, dtype=np.float32).reshape(-1))
    obj.data.update()


def _record_wring_towel() -> DemoResult:
    name = "01_wring_towel_xpbd_realtime"
    title = "SSBL XPBD realtime wring - twist deformation"
    _clear_scene()
    _case_dir, frames_dir, video_path = _ensure_output_dir(name)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = WRING_FRAME_COUNT + 4
    scene.frame_set(1)
    towel = _make_towel()

    blue = _material("SSBL_Demo_Towel_Blue", (0.20, 0.42, 0.95, 1.0))
    stripe = _material("SSBL_Demo_Towel_Stripe", (1.0, 0.54, 0.18, 1.0))
    light = _material("SSBL_Demo_Towel_Light", (0.75, 0.88, 1.0, 1.0))
    towel.data.materials.append(blue)
    towel.data.materials.append(stripe)
    towel.data.materials.append(light)
    for poly in towel.data.polygons:
        radial_index = int(poly.index) % RADIAL_SEGMENTS
        poly.material_index = 1 if radial_index in (0, 1, 2) else (2 if radial_index % 6 == 0 else 0)
    towel.show_wire = True

    camera = _configure_scene_render(
        scene,
        camera_location=(3.2, -5.2, 2.35),
        target=(0.0, 0.0, 0.8),
        ortho_scale=4.65,
    )
    overlay = _create_overlay(scene, camera, title)

    settings = scene.ssbl_preview
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = "ssbl_pin"
    settings.hardness = 0.0
    settings.hardness_initialized = True
    settings.use_volume_pressure = False
    settings.self_collision = True
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 1
    settings.max_self_collision_neighbors = 96
    settings.self_probe_interval = 2
    settings.self_surface_pair_interval = 2
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.static_collider_collection = None
    settings.collision_margin = 0.01
    settings.cloth_thickness = 0.035
    settings.substeps = 20
    settings.iterations = 5
    settings.damping = 0.995
    settings.frame_count = WRING_FRAME_COUNT + 1

    source_mesh = towel.data
    source_before = _snapshot_coords(source_mesh)
    session = ssbl.solver.start_preview(bpy.context, towel)
    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    finite = True
    min_z = float("inf")
    max_radius = 0.0
    simulation_elapsed = 0.0

    for frame in range(0, WRING_FRAME_COUNT + 1):
        last_ms = 0.0
        native_ms = 0.0
        if frame > 0:
            started = time.perf_counter()
            _step_wring_frame(session, towel, frame, WRING_FRAME_COUNT)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            native_ms = float(session.slots[towel.name].native.cached_diagnostics().step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and _wring_all_finite(towel)
        for vert in towel.data.vertices:
            min_z = min(min_z, float((towel.matrix_world @ vert.co).z))
            max_radius = max(max_radius, math.hypot(float(vert.co.y), float(vert.co.z - 1.25)))
        avg_fps = len(step_ms_samples) / max(simulation_elapsed, 1.0e-9)
        _update_overlay(
            overlay,
            frame=frame,
            total_frames=WRING_FRAME_COUNT,
            average_fps=avg_fps,
            last_step_ms=last_ms,
            native_ms=native_ms,
            note=f"540 deg twist, fast self collision, {len(towel.data.vertices)} verts",
        )
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    tethers = len(session.cloth.lra_edges)
    ssbl.solver.request_stop(towel)
    restore_delta = _max_abs_delta(source_before, _snapshot_coords(source_mesh))
    if tethers != 0:
        raise RuntimeError(f"wring towel hardness=0 created hidden tethers: {tethers}")
    _encode_video(
        frames_dir,
        video_path,
        [
            title,
            _metrics_overlay(step_ms_samples, simulation_elapsed),
            f"540 deg twist | self collision fast | restore delta {restore_delta:.1e}",
        ],
    )
    return _summarize_demo(
        name=name,
        title=title,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        step_ms_samples=step_ms_samples,
        simulation_elapsed=simulation_elapsed,
        finite=finite,
        restore_delta=restore_delta,
        metrics={
            "vertices": len(towel.data.vertices),
            "triangles": len(session.cloth.triangles),
            "twist_degrees": math.degrees(TWIST_RADIANS),
            "self_collision_mode": str(settings.self_collision_mode),
            "cloth_thickness": float(settings.cloth_thickness),
            "max_twist_radius": max_radius,
            "min_z": min_z,
            "tethers": tethers,
        },
    )


def _grid_xy(
    name: str,
    *,
    z: float,
    size: float,
    segments: int,
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=segments, y_subdivisions=segments, size=size, location=(0.0, 0.0, z))
    obj = bpy.context.object
    obj.name = name
    obj.show_wire = True
    obj.data.materials.append(_material(f"{name}_Mat", color))
    return obj


def _configure_cloth_settings(settings, *, frame_count: int, pin_group: str = "") -> None:
    settings.enabled = True
    settings.pin_vertex_group = pin_group
    settings.use_evaluated_mesh = True
    settings.preview_writeback_interval = 1
    settings.preview_target_fps = 60.0
    settings.dt = 1.0 / 60.0
    settings.substeps = 6
    settings.iterations = 2
    settings.frame_count = frame_count + 4
    settings.damping = 0.98
    settings.gravity = (0.0, 0.0, 0.0)
    settings.hardness = 0.55
    settings.hardness_initialized = True
    settings.self_collision = False
    settings.self_collision_mode = "off"
    settings.use_volume_pressure = False
    settings.collision_margin = 0.01
    settings.cloth_thickness = 0.05
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.sphere_object = None
    settings.static_collider_collection = None


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


def _record_multicloth_contact() -> DemoResult:
    name = "02_multicloth_contact_realtime"
    title = "SSBL realtime multi-cloth contact - layered cloth collision"
    frame_count = 48
    _clear_scene()
    _case_dir, frames_dir, video_path = _ensure_output_dir(name)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = _configure_scene_render(
        scene,
        camera_location=(2.2, -3.0, 1.7),
        target=(0.0, 0.0, 0.22),
        ortho_scale=2.75,
    )
    overlay = _create_overlay(scene, camera, title)

    lower = _grid_xy("SSBL_Demo_Lower_Cloth", z=1.0, size=2.1, segments=34, color=(0.10, 0.32, 0.95, 1.0))
    upper = _grid_xy("SSBL_Demo_Upper_Cloth", z=1.025, size=1.8, segments=32, color=(0.95, 0.28, 0.12, 1.0))
    for obj, layer in ((lower, 0), (upper, 1)):
        obj.ssbl_collision_layer = layer
        obj.ssbl_enable_cross_cloth_collision = True
        group = obj.vertex_groups.new(name="ssbl_pin")
        size = 2.1 if obj == lower else 1.8
        group.add([v.index for v in obj.data.vertices if v.co.y > size * 0.41], 1.0, "ADD")
        _configure_cloth_settings(obj.ssbl_cloth, frame_count=frame_count, pin_group="ssbl_pin")
        obj.ssbl_cloth.gravity = (0.0, 0.0, -9.8)
        obj.ssbl_cloth.hardness = 0.55
        obj.ssbl_cloth.collision_margin = 0.04
        obj.ssbl_cloth.cloth_thickness = 0.045

    bpy.ops.object.select_all(action="DESELECT")
    lower.select_set(True)
    upper.select_set(True)
    bpy.context.view_layer.objects.active = upper
    before = {lower.name: _mesh_snapshot(lower), upper.name: _mesh_snapshot(upper)}
    session = ssbl.solver.start_preview(bpy.context, upper)
    if len(session.slots) != 2 or str(session.cross_cloth_mode) == "off":
        raise RuntimeError(
            f"multi-cloth demo expected two cloth slots with cross collision, got slots={len(session.slots)} "
            f"mode={session.cross_cloth_mode}"
        )

    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    finite = True
    simulation_elapsed = 0.0
    max_edge_ratio = 0.0
    max_dynamic_triangles = 0
    min_pair_gap = float("inf")

    for frame in range(0, frame_count + 1):
        last_ms = 0.0
        native_ms = 0.0
        diag = ssbl.solver.session_diagnostics(upper)
        if frame > 0:
            started = time.perf_counter()
            ssbl.solver.step_preview(bpy.context, upper.name)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            diag = ssbl.solver.session_diagnostics(upper)
            native_ms = float(diag.step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and _finite_mesh(lower) and _finite_mesh(upper) and bool(diag.finite)
        lower_slot = session.slots.get(lower.name)
        upper_slot = session.slots.get(upper.name)
        if lower_slot is not None and upper_slot is not None:
            lower_z = np.asarray(lower_slot.current_positions_world, dtype=np.float64)[:, 2]
            upper_z = np.asarray(upper_slot.current_positions_world, dtype=np.float64)[:, 2]
            min_pair_gap = min(min_pair_gap, float(np.min(upper_z) - np.max(lower_z)))
        max_dynamic_triangles = max(max_dynamic_triangles, int(diag.dynamic_triangle_count))
        for slot in session.slots.values():
            max_edge_ratio = max(max_edge_ratio, _slot_max_edge_ratio(slot))
        avg_fps = len(step_ms_samples) / max(simulation_elapsed, 1.0e-9)
        _update_overlay(
            overlay,
            frame=frame,
            total_frames=frame_count,
            average_fps=avg_fps,
            last_step_ms=last_ms,
            native_ms=native_ms,
            note=f"slots {len(session.slots)}, cross mode {session.cross_cloth_mode}, min gap {min_pair_gap:.4f}",
        )
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    ssbl.solver.request_stop(upper)
    restore_delta = max(_mesh_delta(lower, before[lower.name]), _mesh_delta(upper, before[upper.name]))
    _encode_video(
        frames_dir,
        video_path,
        [
            title,
            _metrics_overlay(step_ms_samples, simulation_elapsed),
            f"slots {len(session.slots)} | cross {session.cross_cloth_mode} | dynamic triangles {max_dynamic_triangles}",
        ],
    )
    return _summarize_demo(
        name=name,
        title=title,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        step_ms_samples=step_ms_samples,
        simulation_elapsed=simulation_elapsed,
        finite=finite,
        restore_delta=restore_delta,
        metrics={
            "slots": len(session.slots),
            "cross_mode": str(session.cross_cloth_mode),
            "max_edge_ratio": max_edge_ratio,
            "max_dynamic_triangle_count": max_dynamic_triangles,
            "min_pair_gap": min_pair_gap,
        },
    )


def _make_grid_mesh(
    name: str,
    center: tuple[float, float, float],
    *,
    size: float = 0.66,
    segments: int = 12,
    color: tuple[float, float, float, float] = (1.0, 0.58, 0.16, 1.0),
) -> bpy.types.Object:
    verts = []
    faces = []
    half = size * 0.5
    for y in range(segments + 1):
        fy = -half + size * (y / segments)
        for x in range(segments + 1):
            fx = -half + size * (x / segments)
            verts.append((fx, fy, 0.0))
    stride = segments + 1
    for y in range(segments):
        for x in range(segments):
            a = y * stride + x
            faces.append((a, a + 1, a + stride + 1, a + stride))
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    obj.show_wire = True
    obj.data.materials.append(_material(f"{name}_Mat", color))
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _plane_mesh(
    name: str,
    vertices: list[tuple[float, float, float]],
    location: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.data.materials.append(_material(f"{name}_Mat", color))
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _collection_with_object(name: str, obj: bpy.types.Object) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    try:
        bpy.context.scene.collection.objects.unlink(obj)
    except Exception:
        pass
    collection.objects.link(obj)
    return collection


def _configure_collision_settings(settings, *, frame_count: int) -> None:
    _configure_cloth_settings(settings, frame_count=frame_count)
    settings.preview_writeback_interval = 1
    settings.substeps = 8
    settings.iterations = 1
    settings.damping = 0.995
    settings.gravity = (0.0, 0.0, -0.2)
    settings.hardness = 0.6
    settings.collision_margin = 0.035
    settings.cloth_thickness = 0.035


def _record_object_collision_suite() -> DemoResult:
    name = "03_object_collision_suite_realtime"
    title = "SSBL realtime collision suite - ground, wall, static mesh"
    frame_count = 48
    margin = 0.035
    _clear_scene()
    _case_dir, frames_dir, video_path = _ensure_output_dir(name)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = _configure_scene_render(
        scene,
        camera_location=(0.0, -7.0, 4.2),
        target=(0.0, 0.0, 0.38),
        ortho_scale=5.8,
    )
    overlay = _create_overlay(scene, camera, title)
    label_mat = _material("SSBL_Demo_Label_White", (0.92, 0.92, 0.88, 1.0))

    ground_center = (-1.7, 2.1, 0.0)
    _plane_mesh(
        "Ground_Collider_Visual",
        [(-0.58, -0.58, 0.0), (0.58, -0.58, 0.0), (0.58, 0.58, 0.0), (-0.58, 0.58, 0.0)],
        (ground_center[0], ground_center[1], 0.0),
        (0.18, 0.58, 0.34, 1.0),
    )
    ground = _make_grid_mesh("Ground_Cloth", (ground_center[0], ground_center[1], 0.82))
    _configure_collision_settings(ground.ssbl_cloth, frame_count=frame_count)
    ground.ssbl_cloth.use_ground = True
    ground.ssbl_cloth.ground_height = 0.0
    ground.ssbl_cloth.gravity = (0.0, 0.0, -9.8)
    _add_world_label("Ground", (ground_center[0], ground_center[1], 1.42), camera, label_mat)

    monkey_center = (1.7, 2.1, 0.38)
    bpy.ops.mesh.primitive_monkey_add(size=0.7, location=monkey_center)
    monkey = bpy.context.object
    monkey.name = "Static_Monkey_Collider"
    monkey.data.materials.append(_material("Static_Monkey_Mat", (0.18, 0.38, 0.9, 1.0)))
    bpy.context.view_layer.update()
    monkey_top_z = max((monkey.matrix_world @ Vector(corner)).z for corner in monkey.bound_box)
    monkey_collection = _collection_with_object("SSBL_Object_Collision_Static_Monkey_Collection", monkey)
    monkey_cloth = _make_grid_mesh("Monkey_Cloth", (monkey_center[0], monkey_center[1], monkey_top_z + margin * 0.45))
    _configure_collision_settings(monkey_cloth.ssbl_cloth, frame_count=frame_count)
    monkey_cloth.ssbl_cloth.static_collider_collection = monkey_collection
    _add_world_label("Static Monkey", (monkey_center[0], monkey_center[1], 1.42), camera, label_mat)

    wall_y = -2.1
    _plane_mesh(
        "Wall_Collider_Visual",
        [(-0.55, 0.0, -0.55), (0.55, 0.0, -0.55), (0.55, 0.0, 0.55), (-0.55, 0.0, 0.55)],
        (-1.7, wall_y, 0.45),
        (0.82, 0.22, 0.18, 1.0),
    )
    wall = _make_grid_mesh("Wall_Cloth", (-1.7, wall_y - 0.58, 0.45))
    _configure_collision_settings(wall.ssbl_cloth, frame_count=frame_count)
    wall.ssbl_cloth.use_wall = True
    wall.ssbl_cloth.wall_origin = (-1.7, wall_y, 0.0)
    wall.ssbl_cloth.wall_normal = (0.0, -1.0, 0.0)
    wall.ssbl_cloth.gravity = (0.0, 9.8, 0.0)
    _add_world_label("Wall", (-1.7, wall_y, 1.42), camera, label_mat)

    static_center = (1.7, -2.1, 0.0)
    static_plane = _plane_mesh(
        "Static_Mesh_Collider",
        [(-0.58, -0.58, 0.0), (0.58, -0.58, 0.0), (0.58, 0.58, 0.0), (-0.58, 0.58, 0.0)],
        (static_center[0], static_center[1], 0.32),
        (0.08, 0.68, 0.72, 1.0),
    )
    static_collection = _collection_with_object("SSBL_Object_Collision_Static_Collection", static_plane)
    static_cloth = _make_grid_mesh("Static_Mesh_Cloth", (static_center[0], static_center[1], 0.32 + margin * 0.55))
    _configure_collision_settings(static_cloth.ssbl_cloth, frame_count=frame_count)
    static_cloth.ssbl_cloth.static_collider_collection = static_collection
    _add_world_label("Static Mesh", (static_center[0], static_center[1], 1.42), camera, label_mat)

    cloths = [ground, monkey_cloth, wall, static_cloth]
    before = {obj.name: _mesh_snapshot(obj) for obj in cloths}
    bpy.ops.object.select_all(action="DESELECT")
    for obj in cloths:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = ground
    session = ssbl.solver.start_preview(bpy.context, ground)

    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    finite = True
    simulation_elapsed = 0.0
    max_penetration = 0.0
    max_static_collision_ms = 0.0

    for frame in range(0, frame_count + 1):
        last_ms = 0.0
        native_ms = 0.0
        diag = ssbl.solver.session_diagnostics(ground)
        if frame > 0:
            started = time.perf_counter()
            ssbl.solver.step_preview(bpy.context, ground.name)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            diag = ssbl.solver.session_diagnostics(ground)
            native_ms = float(diag.step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and all(_finite_mesh(obj) for obj in cloths) and bool(diag.finite)
        max_penetration = max(max_penetration, float(diag.penetration_depth))
        max_static_collision_ms = max(max_static_collision_ms, float(diag.static_collision_ms))
        avg_fps = len(step_ms_samples) / max(simulation_elapsed, 1.0e-9)
        _update_overlay(
            overlay,
            frame=frame,
            total_frames=frame_count,
            average_fps=avg_fps,
            last_step_ms=last_ms,
            native_ms=native_ms,
            note=f"slots {len(session.slots)}, collision margin {margin:.3f}, penetration {diag.penetration_depth:.4f}",
        )
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    ssbl.solver.request_stop(ground)
    restore_delta = max(_mesh_delta(obj, before[obj.name]) for obj in cloths)
    _encode_video(
        frames_dir,
        video_path,
        [
            title,
            _metrics_overlay(step_ms_samples, simulation_elapsed),
            f"slots {len(session.slots)} | static collision {max_static_collision_ms:.1f} ms | restore delta {restore_delta:.1e}",
        ],
    )
    return _summarize_demo(
        name=name,
        title=title,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        step_ms_samples=step_ms_samples,
        simulation_elapsed=simulation_elapsed,
        finite=finite,
        restore_delta=restore_delta,
        metrics={
            "slots": len(session.slots),
            "cross_mode": str(session.cross_cloth_mode),
            "max_penetration_depth": max_penetration,
            "max_static_collision_ms": max_static_collision_ms,
        },
    )


def _make_yz_flag(name: str, *, segments: int = 24, size_y: float = 1.8, size_z: float = 1.1) -> bpy.types.Object:
    verts = []
    faces = []
    half_y = size_y * 0.5
    half_z = size_z * 0.5
    for iz in range(segments + 1):
        z = -half_z + size_z * iz / segments
        for iy in range(segments + 1):
            y = -half_y + size_y * iy / segments
            verts.append((0.0, y, z))
    stride = segments + 1
    for iz in range(segments):
        for iy in range(segments):
            a = iz * stride + iy
            faces.append((a, a + 1, a + stride + 1, a + stride))
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (0.0, 0.0, 0.9)
    obj.show_wire = True
    obj.data.materials.append(_material(f"{name}_Mat", (0.18, 0.76, 0.62, 1.0)))
    bpy.context.scene.collection.objects.link(obj)
    group = obj.vertex_groups.new(name="ssbl_pin")
    pin_indices = [v.index for v in obj.data.vertices if v.co.y <= -half_y + 0.02]
    group.add(pin_indices, 1.0, "ADD")
    return obj


def _make_arrow() -> None:
    mat = _material("SSBL_Demo_Wind_Arrow_Mat", (1.0, 0.65, 0.16, 1.0))
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.035, depth=1.35, location=(-0.72, 0.0, 0.9), rotation=(0.0, math.pi / 2.0, 0.0))
    shaft = bpy.context.object
    shaft.name = "SSBL_Demo_Wind_Arrow_Shaft"
    shaft.data.materials.append(mat)
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.11, radius2=0.0, depth=0.28, location=(0.0, 0.0, 0.9), rotation=(0.0, math.pi / 2.0, 0.0))
    head = bpy.context.object
    head.name = "SSBL_Demo_Wind_Arrow_Head"
    head.data.materials.append(mat)


def _average_x(obj: bpy.types.Object) -> float:
    return sum(float(vertex.co.x) for vertex in obj.data.vertices) / max(len(obj.data.vertices), 1)


def _record_force_field_tuning() -> DemoResult:
    name = "04_force_field_tuning_realtime"
    title = "SSBL realtime force fields - live wind and material tuning"
    frame_count = 56
    _clear_scene()
    _case_dir, frames_dir, video_path = _ensure_output_dir(name)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = _configure_scene_render(
        scene,
        camera_location=(2.9, -3.7, 1.75),
        target=(0.25, 0.0, 0.82),
        ortho_scale=2.8,
    )
    overlay = _create_overlay(scene, camera, title)

    cloth = _make_yz_flag("SSBL_Demo_Force_Field_Flag")
    _configure_cloth_settings(cloth.ssbl_cloth, frame_count=frame_count, pin_group="ssbl_pin")
    cloth.ssbl_cloth.preview_writeback_interval = 1
    cloth.ssbl_cloth.substeps = 5
    cloth.ssbl_cloth.iterations = 2
    cloth.ssbl_cloth.gravity = (0.0, 0.0, 0.0)
    cloth.ssbl_cloth.hardness = 0.25
    cloth.ssbl_cloth.use_blender_force_fields = True
    _make_arrow()
    bpy.ops.object.effector_add(type="WIND", location=(-1.25, 0.0, 0.9), rotation=(0.0, math.pi / 2.0, 0.0))
    wind = bpy.context.object
    wind.name = "SSBL_Demo_Wind_Field"
    wind.field.strength = 18.0

    before = _mesh_snapshot(cloth)
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    session = ssbl.solver.start_preview(bpy.context, cloth)

    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    finite = True
    simulation_elapsed = 0.0
    max_avg_x = _average_x(cloth)
    force_field_counts: list[int] = []

    for frame in range(0, frame_count + 1):
        if frame == 22:
            wind.field.strength = 52.0
        if frame == 34:
            cloth.ssbl_cloth.hardness = 0.72
            cloth.ssbl_cloth.hardness_initialized = True
        last_ms = 0.0
        native_ms = 0.0
        diag = ssbl.solver.session_diagnostics(cloth)
        if frame > 0:
            scene.frame_set(frame)
            started = time.perf_counter()
            ssbl.solver.step_preview(bpy.context, cloth.name)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            diag = ssbl.solver.session_diagnostics(cloth)
            native_ms = float(diag.step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and _finite_mesh(cloth) and bool(diag.finite)
        max_avg_x = max(max_avg_x, _average_x(cloth))
        force_field_counts.append(int(diag.force_field_count))
        avg_fps = len(step_ms_samples) / max(simulation_elapsed, 1.0e-9)
        phase = "wind 18" if frame < 22 else ("wind 52" if frame < 34 else "wind 52 + stiffer cloth")
        _update_overlay(
            overlay,
            frame=frame,
            total_frames=frame_count,
            average_fps=avg_fps,
            last_step_ms=last_ms,
            native_ms=native_ms,
            note=f"{phase}, force fields {int(diag.force_field_count)}, avg X {_average_x(cloth):.3f}",
        )
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    ssbl.solver.request_stop(cloth)
    restore_delta = _mesh_delta(cloth, before)
    _encode_video(
        frames_dir,
        video_path,
        [
            title,
            _metrics_overlay(step_ms_samples, simulation_elapsed),
            f"wind {float(wind.field.strength):.0f} | force fields {max(force_field_counts) if force_field_counts else 0} | hardness {float(cloth.ssbl_cloth.hardness):.2f}",
        ],
    )
    return _summarize_demo(
        name=name,
        title=title,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        step_ms_samples=step_ms_samples,
        simulation_elapsed=simulation_elapsed,
        finite=finite,
        restore_delta=restore_delta,
        metrics={
            "slots": len(session.slots),
            "max_average_x": max_avg_x,
            "max_force_field_count": max(force_field_counts) if force_field_counts else 0,
            "final_wind_strength": float(wind.field.strength),
            "final_hardness": float(cloth.ssbl_cloth.hardness),
        },
    )


def _make_contact_sheet(results: list[DemoResult]) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    sheet_frames = OUTPUT_ROOT / "contact_sheet_frames"
    sheet_frames.mkdir(parents=True, exist_ok=True)
    for old in sheet_frames.glob("frame_*.*"):
        old.unlink()
    index = 1
    for result in results:
        for frame_number in (0, result.frame_count // 2, max(result.frame_count - 1, 0)):
            out_path = sheet_frames / f"frame_{index:04d}.jpg"
            _run_checked(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    result.video,
                    "-vf",
                    f"select=eq(n\\,{int(frame_number)})",
                    "-frames:v",
                    "1",
                    "-update",
                    "1",
                    str(out_path),
                ]
            )
            index += 1
    contact_sheet = OUTPUT_ROOT / "contact_sheet.jpg"
    if contact_sheet.exists():
        contact_sheet.unlink()
    _run_checked(
        [
            ffmpeg,
            "-y",
            "-framerate",
            "1",
            "-i",
            str(sheet_frames / "frame_%04d.jpg"),
            "-vf",
            "scale=320:180,tile=3x4",
            "-frames:v",
            "1",
            str(contact_sheet),
        ]
    )
    if not contact_sheet.exists() or contact_sheet.stat().st_size <= 0:
        raise RuntimeError("contact sheet was not generated")
    return str(contact_sheet)


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[DemoResult] = []
    started = time.perf_counter()
    try:
        for record in (
            _record_wring_towel,
            _record_multicloth_contact,
            _record_object_collision_suite,
            _record_force_field_tuning,
        ):
            result = record()
            results.append(result)
            print(f"SSBL_REALTIME_DEMO_DONE {result.name} {result.video}")
        contact_sheet = _make_contact_sheet(results)
        summary = {
            "output_dir": str(OUTPUT_ROOT),
            "contact_sheet": contact_sheet,
            "elapsed_s": time.perf_counter() - started,
            "videos": [result.__dict__ for result in results],
        }
        summary_path = OUTPUT_ROOT / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SSBL_REALTIME_DEMO_PACK", json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if len(results) != 4 or not all(result.validation_passed for result in results):
            raise RuntimeError("Realtime demo pack did not generate all four validated videos")
    finally:
        try:
            ssbl.solver.cleanup_all_sessions()
        except Exception:
            pass
        ssbl.unregister()


if __name__ == "__main__":
    main()
