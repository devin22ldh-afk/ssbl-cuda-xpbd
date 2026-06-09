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


OUTPUT_ROOT = SSBL_ROOT / "recordings" / "realtime_demo_2026-06-05"
RESOLUTION = (1280, 720)
VIDEO_FPS = 24
RESTORE_TOLERANCE = 1.0e-7
LENGTH = 4.0
TOWEL_RADIUS = 0.28
X_SEGMENTS = 96
RADIAL_SEGMENTS = 24
PIN_BAND = 0.55
TWIST_RADIANS = math.radians(540.0)
WRING_FRAME_COUNT = 80

SOURCE_FLAGWAVER = "https://github.com/krikienoid/flagwaver"
SOURCE_BINROOT = "https://github.com/BinRoot/Blender-Cloth-Simulation"
SOURCE_GARMENTLAB = "https://github.com/GarmentLab/GarmentLab"
SOURCE_DIFFCLOTH = "https://github.com/omegaiota/DiffCloth"
PLUGIN_NAME = "SSBL"


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
    source_repo: str
    source_scene: str
    keywords: list[str]
    frames_dir: str
    frame_count: int
    simulation_steps: int
    simulation_elapsed_s: float
    average_simulation_fps: float
    average_record_step_fps: float
    average_viewport_preview_fps: float
    encoded_video_fps: float
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


def _beautify_cloth(obj: bpy.types.Object, *, levels: int = 1) -> bpy.types.Object:
    if obj is None or obj.type != "MESH" or obj.data is None:
        return obj
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.update()
    modifier = obj.modifiers.get("SSBL_Demo_Subsurf")
    if modifier is None:
        modifier = obj.modifiers.new("SSBL_Demo_Subsurf", "SUBSURF")
    modifier.levels = int(levels)
    modifier.render_levels = int(levels)
    modifier.subdivision_type = "CATMULL_CLARK"
    obj.show_wire = False
    return obj


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
    top = height * 0.5 - 0.20
    title_obj = _camera_text(
        scene,
        camera,
        name="SSBL_Demo_Overlay_Title",
        body=_format_overlay_header(title),
        location=(left, top, -4.0),
        size=0.13,
        color=(1.0, 1.0, 1.0, 1.0),
    )
    metrics_obj = _camera_text(
        scene,
        camera,
        name="SSBL_Demo_Overlay_Metrics",
        body="",
        location=(left, top - 0.20, -4.0),
        size=0.092,
        color=(0.0, 1.0, 0.0, 1.0),
    )
    notes_obj = _camera_text(
        scene,
        camera,
        name="SSBL_Demo_Overlay_Notes",
        body="",
        location=(left, top - 0.36, -4.0),
        size=0.08,
        color=(1.0, 1.0, 1.0, 0.0),
    )
    return Overlay(title_obj, metrics_obj, notes_obj)


def _update_overlay(
    overlay: Overlay,
    *,
    metrics_line: str,
    note: str,
) -> None:
    overlay.metrics.data.body = metrics_line
    overlay.notes.data.body = ""
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


def _run_checked(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def _display_step_ms(last_step_ms: float, native_ms: float) -> float:
    native_ms = float(native_ms)
    if math.isfinite(native_ms) and native_ms > 0.0:
        return native_ms
    last_step_ms = float(last_step_ms)
    if math.isfinite(last_step_ms) and last_step_ms > 0.0:
        return last_step_ms
    return 0.0


def _current_sim_fps(last_step_ms: float, native_ms: float) -> float:
    step_ms = _display_step_ms(last_step_ms, native_ms)
    if step_ms <= 0.0:
        return 0.0
    return 1000.0 / step_ms


def _valid_fps(value: float | None) -> bool:
    if value is None:
        return False
    value = float(value)
    return math.isfinite(value) and value > 0.0


def _average_fps(values: list[float] | None) -> float:
    if not values:
        return 0.0
    valid = [float(value) for value in values if _valid_fps(value)]
    return sum(valid) / len(valid) if valid else 0.0


def _sample_fps_for_frame(samples: list[float] | None, frame: int) -> float:
    if not samples:
        return 0.0
    index = max(0, min(int(frame), len(samples) - 1))
    value = float(samples[index])
    if _valid_fps(value):
        return value
    for fallback in samples:
        if _valid_fps(fallback):
            return float(fallback)
    return 0.0


def _format_metrics_line(
    frame: int,
    total_frames: int,
    last_step_ms: float,
    native_ms: float,
    viewport_fps: float | None = None,
) -> str:
    fps = float(viewport_fps) if _valid_fps(viewport_fps) else _current_sim_fps(last_step_ms, native_ms)
    return f"Viewport FPS: {fps:5.1f}"


def _compose_overlay_text(title: str, metrics_line: str, note_line: str) -> str:
    return "\n".join(line for line in (_format_overlay_header(title), metrics_line) if line)


def _format_overlay_header(title: str) -> str:
    title = str(title).strip()
    if not title:
        return PLUGIN_NAME
    if title == PLUGIN_NAME or title.startswith(f"{PLUGIN_NAME} "):
        return title
    return f"{PLUGIN_NAME} {title}"


def _ass_color(red: int, green: int, blue: int, alpha: int = 0) -> str:
    red = max(0, min(255, int(red)))
    green = max(0, min(255, int(green)))
    blue = max(0, min(255, int(blue)))
    alpha = max(0, min(255, int(alpha)))
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


def _escape_ass_text(text: str) -> str:
    return str(text).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _write_overlay_ass(frames_dir: Path, overlay_text_frames: list[str]) -> Path:
    if not overlay_text_frames:
        raise RuntimeError("overlay_text_frames must not be empty")
    overlay_path = frames_dir / "overlay.ass"
    entries: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {RESOLUTION[0]}",
        f"PlayResY: {RESOLUTION[1]}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Header,Arial,22,"
        f"{_ass_color(255, 255, 255)},{_ass_color(255, 255, 255)},{_ass_color(0, 0, 0)},{_ass_color(0, 0, 0, 128)},"
        "1,0,0,0,100,100,0,0,1,1,0,7,28,28,22,1",
        "Style: RealtimeFps,Arial,20,"
        f"{_ass_color(0, 255, 0)},{_ass_color(0, 255, 0)},{_ass_color(0, 0, 0)},{_ass_color(0, 0, 0, 128)},"
        "1,0,0,0,100,100,0,0,1,1,0,7,28,28,52,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for index, text in enumerate(overlay_text_frames, start=1):
        start_s = (index - 1) / VIDEO_FPS
        end_s = index / VIDEO_FPS
        lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
        header = _escape_ass_text(lines[0] if lines else "")
        fps_line = _escape_ass_text(lines[1] if len(lines) > 1 else "")
        entries.extend(
            [
                f"Dialogue: 0,{_format_ass_timestamp(start_s)},{_format_ass_timestamp(end_s)},Header,,0,0,0,,{header}",
                f"Dialogue: 0,{_format_ass_timestamp(start_s)},{_format_ass_timestamp(end_s)},RealtimeFps,,0,0,0,,{fps_line}",
            ]
        )
    overlay_path.write_text("\n".join(entries), encoding="utf-8")
    return overlay_path


def _format_ass_timestamp(seconds: float) -> str:
    total_cs = max(0, int(round(float(seconds) * 100.0)))
    hours = total_cs // 360000
    minutes = (total_cs // 6000) % 60
    secs = (total_cs // 100) % 60
    centis = total_cs % 100
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _encode_video(frames_dir: Path, video_path: Path, overlay_text_frames: list[str]) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    overlay_path = _write_overlay_ass(frames_dir, overlay_text_frames)
    filter_chain = (
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,"
        f"subtitles={overlay_path.name}"
    )
    _run_checked(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(VIDEO_FPS),
            "-i",
            "frame_%04d.png",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-vf",
            filter_chain,
            str(video_path),
        ],
        cwd=frames_dir,
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
    source_repo: str,
    source_scene: str,
    keywords: list[str],
    video_path: Path,
    frames_dir: Path,
    frame_paths: list[str],
    step_ms_samples: list[float],
    simulation_elapsed: float,
    finite: bool,
    restore_delta: float,
    metrics: dict[str, object],
    viewport_fps_samples: list[float] | None = None,
) -> DemoResult:
    steps = len(step_ms_samples)
    average_record_step_fps = steps / max(float(simulation_elapsed), 1.0e-9)
    average_ms = sum(step_ms_samples) / max(steps, 1)
    average_viewport_preview_fps = _average_fps(viewport_fps_samples)
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
        source_repo=source_repo,
        source_scene=source_scene,
        keywords=list(keywords),
        frames_dir=str(frames_dir),
        frame_count=len(frame_paths),
        simulation_steps=steps,
        simulation_elapsed_s=float(simulation_elapsed),
        average_simulation_fps=float(average_record_step_fps),
        average_record_step_fps=float(average_record_step_fps),
        average_viewport_preview_fps=float(average_viewport_preview_fps),
        encoded_video_fps=float(VIDEO_FPS),
        average_step_ms=float(average_ms),
        p95_step_ms=float(_p95(step_ms_samples)),
        finite=bool(finite),
        restore_delta=float(restore_delta),
        validation_passed=True,
        metrics=metrics,
        ffprobe=probe,
        representative_frames=picks,
    )


def _keyword_line(*keywords: str) -> str:
    return " | ".join(keyword.strip() for keyword in keywords if keyword and keyword.strip())


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


def _rotate_yz(y: float, z: float, angle: float) -> tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return y * cos_a - z * sin_a, y * sin_a + z * cos_a


def _make_towel() -> bpy.types.Object:
    vertices = []
    faces = []
    for ix in range(X_SEGMENTS + 1):
        x = -LENGTH * 0.5 + LENGTH * ix / X_SEGMENTS
        for ir in range(RADIAL_SEGMENTS):
            angle = math.tau * ir / RADIAL_SEGMENTS
            y = math.cos(angle) * TOWEL_RADIUS
            z = 1.25 + math.sin(angle) * TOWEL_RADIUS
            vertices.append((x, y, z))

    row = RADIAL_SEGMENTS
    for ix in range(X_SEGMENTS):
        for ir in range(RADIAL_SEGMENTS):
            a = ix * row + ir
            b = ix * row + ((ir + 1) % RADIAL_SEGMENTS)
            c = (ix + 1) * row + ((ir + 1) % RADIAL_SEGMENTS)
            d = (ix + 1) * row + ir
            faces.append((a, b, c, d))

    mesh = bpy.data.meshes.new("SSBL_Wring_Towel_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("SSBL_Wring_Towel", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    pin = obj.vertex_groups.new(name="ssbl_pin")
    pin_indices = [vert.index for vert in obj.data.vertices if abs(vert.co.x) >= LENGTH * 0.5 - PIN_BAND]
    pin.add(pin_indices, 1.0, "ADD")
    return obj


def _wring_all_finite(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vert in obj.data.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    )


def _snapshot_coords(mesh: bpy.types.Mesh) -> list[float]:
    return [component for vert in mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]


def _max_abs_delta(before: list[float], after: list[float]) -> float:
    return max((abs(float(a) - float(b)) for a, b in zip(after, before)), default=0.0)


def _wring_pin_targets(cloth, progress: float) -> np.ndarray:
    progress = max(0.0, min(1.0, float(progress)))
    rest = np.asarray(cloth.positions_world[cloth.pin_indices], dtype=np.float32)
    targets = np.array(rest, dtype=np.float32, copy=True)
    left_limit = -LENGTH * 0.5 + PIN_BAND + 1.0e-5
    right_limit = LENGTH * 0.5 - PIN_BAND - 1.0e-5
    for index, point in enumerate(rest):
        x = float(point[0])
        if x <= left_limit:
            angle = TWIST_RADIANS * progress
        elif x >= right_limit:
            angle = -TWIST_RADIANS * progress
        else:
            continue
        y, z_delta = _rotate_yz(float(point[1]), float(point[2]) - 1.25, angle)
        targets[index, 1] = y
        targets[index, 2] = 1.25 + z_delta
    return np.ascontiguousarray(targets, dtype=np.float32)


def _step_wring_frame(session, obj: bpy.types.Object, frame: int, frame_count: int) -> np.ndarray:
    slot = session.slots[obj.name]
    progress = max(0.0, min(1.0, float(frame) / max(float(frame_count), 1.0)))
    slot.native.update_pin_targets(slot.cloth.pin_indices, _wring_pin_targets(slot.cloth, progress))
    slot.native.step(session.substeps, session.iterations)
    slot.current_positions_world = np.array(slot.native.download_positions(), dtype=np.float32, copy=True)
    _apply_world_positions(obj, slot.current_positions_world, slot.cloth.matrix_world_inv)
    return slot.current_positions_world


def _record_wring_towel() -> DemoResult:
    name = "02_wring_towel_realtime"
    title = "SSBL realtime wring towel - hook driven twist"
    keywords = [
        "Hook Driven Wring",
        "Stable Self-Collision",
        "Twist Deformation",
    ]
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
    _beautify_cloth(towel)

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
    overlay_text_frames: list[str] = []
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
        metrics_line = _format_metrics_line(frame, WRING_FRAME_COUNT, last_ms, native_ms)
        _update_overlay(
            overlay,
            metrics_line=metrics_line,
            note=_keyword_line(*keywords),
        )
        overlay_text_frames.append(_compose_overlay_text(title, metrics_line, _keyword_line(*keywords)))
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    tethers = len(session.cloth.lra_edges)
    ssbl.solver.request_stop(towel)
    restore_delta = _max_abs_delta(source_before, _snapshot_coords(source_mesh))
    if tethers != 0:
        raise RuntimeError(f"wring towel hardness=0 created hidden tethers: {tethers}")
    _encode_video(
        frames_dir,
        video_path,
        overlay_text_frames,
    )
    return _summarize_demo(
        name=name,
        title=title,
        source_repo=SOURCE_BINROOT,
        source_scene="Two-point towel wring inspiration",
        keywords=keywords,
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
    settings.use_evaluated_mesh = False
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


def _make_hanging_panel(
    name: str,
    *,
    location: tuple[float, float, float],
    width: float,
    height: float,
    x_segments: int,
    z_segments: int,
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    verts = []
    faces = []
    half_w = width * 0.5
    for iz in range(z_segments + 1):
        z = height - height * iz / z_segments
        for ix in range(x_segments + 1):
            x = -half_w + width * ix / x_segments
            verts.append((x, 0.0, z))
    stride = x_segments + 1
    for iz in range(z_segments):
        for ix in range(x_segments):
            a = iz * stride + ix
            faces.append((a, a + 1, a + stride + 1, a + stride))
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.data.materials.append(_material(f"{name}_Mat", color))
    bpy.context.scene.collection.objects.link(obj)
    _beautify_cloth(obj)
    pin = obj.vertex_groups.new(name="ssbl_pin")
    pin.add([v.index for v in obj.data.vertices if v.co.z >= height - 0.12], 1.0, "ADD")
    return obj


def _make_clothesline_visual() -> None:
    rope_mat = _material("SSBL_Demo_Clothesline_Rope", (0.86, 0.74, 0.56, 1.0))
    clip_mat = _material("SSBL_Demo_Clothesline_Clip", (0.16, 0.16, 0.18, 1.0))
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.03, depth=4.3, location=(0.0, 0.0, 2.12), rotation=(0.0, math.pi / 2.0, 0.0))
    rope = bpy.context.object
    rope.name = "SSBL_Demo_Clothesline_Rope"
    rope.data.materials.append(rope_mat)
    for x in (-0.88, -0.08, 0.74):
        bpy.ops.mesh.primitive_cube_add(location=(x, 0.0, 2.02), scale=(0.05, 0.04, 0.08))
        clip = bpy.context.object
        clip.name = f"SSBL_Demo_Clip_{int((x + 2.0) * 100):03d}"
        clip.data.materials.append(clip_mat)


def _activate_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _apply_object_rotation_scale(obj: bpy.types.Object) -> None:
    _activate_only(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def _apply_modifier(obj: bpy.types.Object, modifier_name: str) -> None:
    _activate_only(obj)
    bpy.ops.object.modifier_apply(modifier=modifier_name)


def _move_object_to_collection(collection: bpy.types.Collection, obj: bpy.types.Object) -> None:
    for owner in list(obj.users_collection):
        try:
            owner.objects.unlink(obj)
        except Exception:
            pass
    if collection.objects.get(obj.name) is None:
        collection.objects.link(obj)


def _make_pressure_collision_chamber() -> bpy.types.Collection:
    chamber_collection = bpy.data.collections.new("SSBL_Demo_Pressure_Chamber_Collection")
    bpy.context.scene.collection.children.link(chamber_collection)
    frame_mat = _material("SSBL_Demo_Chamber_Frame", (0.10, 0.12, 0.16, 1.0))
    wall_mat = _material("SSBL_Demo_Chamber_Wall", (0.09, 0.105, 0.13, 1.0))
    floor_mat = _material("SSBL_Demo_Chamber_Floor", (0.045, 0.052, 0.065, 1.0))
    specs = [
        ("Floor", (0.0, 0.10, 0.00), (1.95, 1.55, 0.06), floor_mat),
        ("Back", (0.0, 1.55, 1.55), (1.95, 0.06, 1.55), wall_mat),
        ("Left", (-1.95, 0.10, 1.55), (0.06, 1.55, 1.55), wall_mat),
        ("Right", (1.95, 0.10, 1.55), (0.06, 1.55, 1.55), wall_mat),
        ("Ceiling", (0.0, 0.10, 3.10), (1.95, 1.55, 0.06), wall_mat),
    ]
    for label, location, scale, material in specs:
        bpy.ops.mesh.primitive_cube_add(location=location, scale=scale)
        wall = bpy.context.object
        wall.name = f"SSBL_Demo_Chamber_{label}"
        wall.data.materials.append(material)
        if label in {"Back", "Left", "Right", "Ceiling"}:
            wall.hide_render = True
            wall.hide_viewport = True
        _move_object_to_collection(chamber_collection, wall)

    frame_specs = [
        ("Front_Left", (-1.95, -1.45, 1.55), (0.035, 0.035, 1.55)),
        ("Front_Right", (1.95, -1.45, 1.55), (0.035, 0.035, 1.55)),
        ("Front_Bottom", (0.0, -1.45, 0.00), (1.95, 0.035, 0.035)),
    ]
    for label, location, scale in frame_specs:
        bpy.ops.mesh.primitive_cube_add(location=location, scale=scale)
        frame = bpy.context.object
        frame.name = f"SSBL_Demo_Chamber_Frame_{label}"
        frame.data.materials.append(frame_mat)
    return chamber_collection


def _make_inflated_remeshed_monkey(name: str, color: tuple[float, float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_monkey_add(
        size=0.92,
        location=(-0.66, -0.04, 2.48),
        rotation=(math.radians(88.0), math.radians(-7.0), math.radians(22.0)),
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = (1.12, 0.96, 0.92)
    _apply_object_rotation_scale(obj)
    subdiv = obj.modifiers.new("SSBL_Demo_PreRemesh_Subsurf", "SUBSURF")
    subdiv.levels = 2
    subdiv.render_levels = 2
    subdiv.subdivision_type = "CATMULL_CLARK"
    _apply_modifier(obj, "SSBL_Demo_PreRemesh_Subsurf")
    remesh = obj.modifiers.new("SSBL_Demo_Remesher", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = 0.12
    if hasattr(remesh, "adaptivity"):
        remesh.adaptivity = 0.0
    if hasattr(remesh, "use_remove_disconnected"):
        remesh.use_remove_disconnected = False
    _apply_modifier(obj, "SSBL_Demo_Remesher")
    obj.data.materials.clear()
    obj.data.materials.append(_material(f"{name}_Mat", color))
    _beautify_cloth(obj, levels=2)
    return obj


def _make_inflated_torus(name: str, color: tuple[float, float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_segments=42,
        minor_segments=22,
        major_radius=0.62,
        minor_radius=0.19,
        location=(0.48, -0.16, 2.26),
        rotation=(math.radians(72.0), math.radians(13.0), math.radians(20.0)),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.clear()
    obj.data.materials.append(_material(f"{name}_Mat", color))
    _beautify_cloth(obj, levels=2)
    return obj


def _make_inflated_sphere(name: str, color: tuple[float, float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=26,
        ring_count=18,
        radius=0.56,
        location=(-0.04, 0.36, 2.82),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.clear()
    obj.data.materials.append(_material(f"{name}_Mat", color))
    _beautify_cloth(obj, levels=2)
    return obj


def _make_preview_auto_sphere_blocker() -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16,
        ring_count=8,
        radius=0.1,
        location=(50.0, 50.0, -50.0),
    )
    obj = bpy.context.object
    obj.name = "SSBL_Demo_AutoSphere_Blocker"
    return obj


def _configure_pressure_cloth(settings, *, frame_count: int, pressure_strength: float, collider_collection) -> None:
    _configure_cloth_settings(settings, frame_count=frame_count)
    settings.gravity = (0.0, 0.0, -8.6)
    settings.substeps = 8
    settings.iterations = 2
    settings.damping = 0.988
    settings.hardness = 0.34
    settings.self_collision = True
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 1
    settings.max_self_collision_neighbors = 56
    settings.fast_self_collision_passes = 4
    settings.use_volume_pressure = True
    settings.pressure_strength = pressure_strength
    settings.collision_margin = 0.028
    settings.cloth_thickness = 0.03
    settings.static_collider_collection = collider_collection


def _record_multicloth_contact() -> DemoResult:
    name = "03_clothesline_multicloth_realtime"
    title = "SSBL realtime inflated trio - chamber contact"
    keywords = [
        "Remeshed Suzanne",
        "Inflated Multi-Cloth",
        "Closed-Shell Collision",
    ]
    frame_count = 112
    _clear_scene()
    _case_dir, frames_dir, video_path = _ensure_output_dir(name)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = _configure_scene_render(
        scene,
        camera_location=(3.15, -5.35, 2.85),
        target=(0.0, 0.02, 2.18),
        ortho_scale=4.4,
    )
    overlay = _create_overlay(scene, camera, title)

    chamber_collection = _make_pressure_collision_chamber()
    cloths = [
        _make_inflated_remeshed_monkey("SSBL_Demo_Inflated_Monkey", (0.91, 0.35, 0.18, 1.0)),
        _make_inflated_torus("SSBL_Demo_Inflated_Torus", (0.17, 0.55, 0.95, 1.0)),
        _make_inflated_sphere("SSBL_Demo_Inflated_Sphere", (0.97, 0.80, 0.23, 1.0)),
    ]
    preview_driver = cloths[1]
    pressure_strengths = [0.15, 0.20, 0.18]
    for layer, (obj, pressure_strength) in enumerate(zip(cloths, pressure_strengths)):
        obj.ssbl_collision_layer = layer
        obj.ssbl_enable_cross_cloth_collision = True
        _configure_pressure_cloth(
            obj.ssbl_cloth,
            frame_count=frame_count,
            pressure_strength=pressure_strength,
            collider_collection=chamber_collection,
        )
    auto_sphere_blocker = _make_preview_auto_sphere_blocker()
    preview_driver.ssbl_cloth.use_sphere = True
    preview_driver.ssbl_cloth.sphere_object = auto_sphere_blocker

    bpy.ops.object.select_all(action="DESELECT")
    for obj in cloths:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = preview_driver
    before = {obj.name: _mesh_snapshot(obj) for obj in cloths}
    session = ssbl.solver.start_preview(bpy.context, preview_driver)
    if len(session.slots) != len(cloths) or str(session.cross_cloth_mode) == "off":
        raise RuntimeError(
            f"multi-cloth demo expected three cloth slots with cross collision, got slots={len(session.slots)} "
            f"mode={session.cross_cloth_mode}"
        )

    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    overlay_text_frames: list[str] = []
    finite = True
    simulation_elapsed = 0.0
    max_edge_ratio = 0.0
    max_dynamic_triangles = 0
    min_centroid_distance = float("inf")

    for frame in range(0, frame_count + 1):
        last_ms = 0.0
        native_ms = 0.0
        diag = ssbl.solver.session_diagnostics(cloths[0])
        if frame > 0:
            started = time.perf_counter()
            scene.frame_set(frame)
            ssbl.solver.step_preview(bpy.context, cloths[0].name)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            diag = ssbl.solver.session_diagnostics(cloths[0])
            native_ms = float(diag.step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and all(_finite_mesh(obj) for obj in cloths) and bool(diag.finite)
        max_dynamic_triangles = max(max_dynamic_triangles, int(diag.dynamic_triangle_count))
        for slot in session.slots.values():
            max_edge_ratio = max(max_edge_ratio, _slot_max_edge_ratio(slot))
        centers = []
        for obj in cloths:
            positions = np.asarray(session.slots[obj.name].current_positions_world, dtype=np.float64)
            if len(positions):
                centers.append(np.mean(positions, axis=0))
        for index, first_center in enumerate(centers):
            for second_center in centers[index + 1 :]:
                min_centroid_distance = min(
                    min_centroid_distance,
                    float(np.linalg.norm(first_center - second_center)),
                )
        metrics_line = _format_metrics_line(frame, frame_count, last_ms, native_ms)
        _update_overlay(
            overlay,
            metrics_line=metrics_line,
            note=_keyword_line(*keywords),
        )
        overlay_text_frames.append(_compose_overlay_text(title, metrics_line, _keyword_line(*keywords)))
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    ssbl.solver.request_stop(cloths[0])
    restore_delta = max(_mesh_delta(obj, before[obj.name]) for obj in cloths)
    _encode_video(
        frames_dir,
        video_path,
        overlay_text_frames,
    )
    return _summarize_demo(
        name=name,
        title=title,
        source_repo="",
        source_scene="Custom pressure chamber with remeshed Suzanne, torus, and sphere",
        keywords=keywords,
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
            "min_centroid_distance": min_centroid_distance,
            "pressure_strengths": pressure_strengths,
            "monkey_vertices": len(cloths[0].data.vertices),
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
    obj.data.materials.append(_material(f"{name}_Mat", color))
    bpy.context.scene.collection.objects.link(obj)
    _beautify_cloth(obj)
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


def _make_tablecloth(size: float, color: tuple[float, float, float, float]) -> bpy.types.Object:
    cloth = _grid_xy("SSBL_Demo_Tablecloth", z=1.28, size=size, segments=36, color=color)
    _beautify_cloth(cloth)
    pin = cloth.vertex_groups.new(name="ssbl_pin")
    pin_indices = [
        vert.index
        for vert in cloth.data.vertices
        if vert.co.x >= size * 0.42 and abs(vert.co.y) >= size * 0.28
    ]
    pin.add(pin_indices, 1.0, "ADD")
    return cloth


def _make_table_scene_visual() -> tuple[bpy.types.Collection, bpy.types.Object, bpy.types.Object]:
    table_mat = _material("SSBL_Demo_Table_Wood", (0.50, 0.33, 0.17, 1.0))
    leg_mat = _material("SSBL_Demo_Table_Leg", (0.18, 0.15, 0.12, 1.0))
    handle_mat = _material("SSBL_Demo_Handle", (1.0, 0.74, 0.18, 1.0))
    floor_mat = _material("SSBL_Demo_Floor", (0.08, 0.09, 0.10, 1.0))

    floor = _plane_mesh(
        "SSBL_Demo_Table_Floor",
        [(-3.6, -3.0, 0.0), (3.6, -3.0, 0.0), (3.6, 3.0, 0.0), (-3.6, 3.0, 0.0)],
        (0.0, 0.0, 0.0),
        (0.08, 0.09, 0.10, 1.0),
    )
    floor.data.materials.clear()
    floor.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.88), scale=(0.84, 0.56, 0.26))
    tabletop = bpy.context.object
    tabletop.name = "SSBL_Demo_Tabletop"
    tabletop.data.materials.append(table_mat)
    collider_collection = _collection_with_object("SSBL_Demo_Table_Collider_Collection", tabletop)

    for x in (-0.66, 0.66):
        for y in (-0.38, 0.38):
            bpy.ops.mesh.primitive_cube_add(location=(x, y, 0.42), scale=(0.05, 0.05, 0.42))
            leg = bpy.context.object
            leg.name = f"SSBL_Demo_Table_Leg_{int((x + 1.0) * 100):03d}_{int((y + 1.0) * 100):03d}"
            leg.data.materials.append(leg_mat)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.08, location=(1.05, 0.58, 1.34))
    handle_a = bpy.context.object
    handle_a.name = "SSBL_Demo_Table_Handle_A"
    handle_a.data.materials.append(handle_mat)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.08, location=(1.05, -0.58, 1.34))
    handle_b = bpy.context.object
    handle_b.name = "SSBL_Demo_Table_Handle_B"
    handle_b.data.materials.append(handle_mat)
    return collider_collection, handle_a, handle_b


def _ease_out_cubic(progress: float) -> float:
    clamped = max(0.0, min(1.0, float(progress)))
    return 1.0 - (1.0 - clamped) ** 3


def _tablecloth_pin_targets(cloth, pull_progress: float) -> np.ndarray:
    rest = np.asarray(cloth.positions_world[cloth.pin_indices], dtype=np.float32)
    targets = np.array(rest, dtype=np.float32, copy=True)
    progress = _ease_out_cubic(pull_progress)
    for index, point in enumerate(rest):
        sign_y = 1.0 if float(point[1]) >= 0.0 else -1.0
        targets[index, 0] = float(point[0]) + 1.35 * progress
        targets[index, 1] = float(point[1]) + 0.10 * sign_y * progress
        targets[index, 2] = float(point[2]) + 0.34 * progress
    return np.ascontiguousarray(targets, dtype=np.float32)


def _pin_cluster_centers(pin_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = pin_targets[pin_targets[:, 1] >= 0.0]
    negative = pin_targets[pin_targets[:, 1] < 0.0]
    positive_center = np.mean(positive, axis=0) if len(positive) else np.zeros(3, dtype=np.float32)
    negative_center = np.mean(negative, axis=0) if len(negative) else np.zeros(3, dtype=np.float32)
    return positive_center, negative_center


def _manual_native_step(session, obj: bpy.types.Object, pin_targets: np.ndarray | None = None):
    slot = session.slots[obj.name]
    if pin_targets is not None and len(slot.cloth.pin_indices) > 0:
        slot.native.update_pin_targets(slot.cloth.pin_indices, pin_targets)
        slot.pin_targets_world = np.array(pin_targets, dtype=np.float32, copy=True)
    slot.native.step(slot.substeps, slot.iterations)
    slot.current_positions_world = np.array(slot.native.download_positions(), dtype=np.float32, copy=True)
    _apply_world_positions(obj, slot.current_positions_world, slot.cloth.matrix_world_inv)
    return slot.native.cached_diagnostics()


def _record_object_collision_suite() -> DemoResult:
    name = "04_tablecloth_pull_collision_realtime"
    title = "SSBL realtime tablecloth pull - rigid edge collision"
    keywords = [
        "Pinned Corner Pull",
        "Rigid Edge Collision",
        "Stable Drape",
    ]
    frame_count = 112
    _clear_scene()
    _case_dir, frames_dir, video_path = _ensure_output_dir(name)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = _configure_scene_render(
        scene,
        camera_location=(4.9, -4.8, 3.45),
        target=(0.1, 0.0, 0.92),
        ortho_scale=4.35,
    )
    overlay = _create_overlay(scene, camera, title)
    collider_collection, handle_a, handle_b = _make_table_scene_visual()
    cloth = _make_tablecloth(2.42, (0.94, 0.92, 0.86, 1.0))
    _configure_collision_settings(cloth.ssbl_cloth, frame_count=frame_count)
    cloth.ssbl_cloth.pin_vertex_group = "ssbl_pin"
    cloth.ssbl_cloth.gravity = (0.0, 0.0, -9.8)
    cloth.ssbl_cloth.static_collider_collection = collider_collection
    cloth.ssbl_cloth.damping = 0.992
    cloth.ssbl_cloth.collision_margin = 0.028
    cloth.ssbl_cloth.cloth_thickness = 0.03

    before = {cloth.name: _mesh_snapshot(cloth)}
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    session = ssbl.solver.start_preview(bpy.context, cloth)
    initial_center_a, initial_center_b = _pin_cluster_centers(np.asarray(session.slots[cloth.name].pin_targets_world, dtype=np.float32))
    handle_a.location = Vector(tuple(initial_center_a))
    handle_b.location = Vector(tuple(initial_center_b))

    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    overlay_text_frames: list[str] = []
    finite = True
    simulation_elapsed = 0.0
    max_penetration = 0.0
    max_static_collision_ms = 0.0
    max_pull_distance = 0.0

    for frame in range(0, frame_count + 1):
        last_ms = 0.0
        native_ms = 0.0
        diag = session.slots[cloth.name].native.cached_diagnostics()
        if frame > 0:
            pull_progress = 0.0 if frame <= 14 else (frame - 14) / max(frame_count - 14, 1)
            pin_targets = _tablecloth_pin_targets(session.slots[cloth.name].cloth, pull_progress)
            started = time.perf_counter()
            diag = _manual_native_step(session, cloth, pin_targets=pin_targets)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            native_ms = float(diag.step_ms)
            step_ms_samples.append(last_ms)
            center_a, center_b = _pin_cluster_centers(pin_targets)
            handle_a.location = Vector(tuple(center_a))
            handle_b.location = Vector(tuple(center_b))
            max_pull_distance = max(
                max_pull_distance,
                float(np.linalg.norm(center_a - initial_center_a)),
                float(np.linalg.norm(center_b - initial_center_b)),
            )
        finite = finite and _finite_mesh(cloth) and bool(diag.finite)
        max_penetration = max(max_penetration, float(diag.penetration_depth))
        max_static_collision_ms = max(max_static_collision_ms, float(diag.static_collision_ms))
        metrics_line = _format_metrics_line(frame, frame_count, last_ms, native_ms)
        _update_overlay(
            overlay,
            metrics_line=metrics_line,
            note=_keyword_line(*keywords),
        )
        overlay_text_frames.append(_compose_overlay_text(title, metrics_line, _keyword_line(*keywords)))
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    ssbl.solver.request_stop(cloth)
    restore_delta = _mesh_delta(cloth, before[cloth.name])
    _encode_video(
        frames_dir,
        video_path,
        overlay_text_frames,
    )
    return _summarize_demo(
        name=name,
        title=title,
        source_repo=SOURCE_GARMENTLAB,
        source_scene="Mobile scene inspiration adapted to a two-handle tablecloth pull",
        keywords=keywords,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        step_ms_samples=step_ms_samples,
        simulation_elapsed=simulation_elapsed,
        finite=finite,
        restore_delta=restore_delta,
        metrics={
            "slots": len(session.slots),
            "max_penetration_depth": max_penetration,
            "max_static_collision_ms": max_static_collision_ms,
            "max_pull_distance": max_pull_distance,
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
    obj.data.materials.append(_material(f"{name}_Mat_Base", (0.18, 0.76, 0.62, 1.0)))
    obj.data.materials.append(_material(f"{name}_Mat_Accent", (0.95, 0.28, 0.14, 1.0)))
    obj.data.materials.append(_material(f"{name}_Mat_Light", (0.94, 0.95, 0.97, 1.0)))
    bpy.context.scene.collection.objects.link(obj)
    _beautify_cloth(obj)
    group = obj.vertex_groups.new(name="ssbl_pin")
    pin_indices = [v.index for v in obj.data.vertices if v.co.y <= -half_y + 0.02]
    group.add(pin_indices, 1.0, "ADD")
    stride = segments
    for poly in obj.data.polygons:
        row = poly.index // stride
        obj.data.polygons[poly.index].material_index = 2 if row < 4 else (1 if row % 5 == 0 else 0)
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


def _average_world_z(obj: bpy.types.Object) -> float:
    return sum(float((obj.matrix_world @ vertex.co).z) for vertex in obj.data.vertices) / max(len(obj.data.vertices), 1)


def _record_force_field_tuning() -> DemoResult:
    name = "01_brand_flag_wind_realtime"
    title = "SSBL realtime flag - live wind control"
    keywords = [
        "Realtime Wind",
        "Turbulent Gusts",
        "Live Tuning",
    ]
    frame_count = 112
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
    cloth.ssbl_cloth.gravity = (0.0, 0.0, -2.4)
    cloth.ssbl_cloth.hardness = 0.25
    cloth.ssbl_cloth.use_blender_force_fields = True
    bpy.ops.object.effector_add(type="WIND", location=(-1.25, 0.0, 0.9), rotation=(0.0, math.pi / 2.0, 0.0))
    wind = bpy.context.object
    wind.name = "SSBL_Demo_Wind_Field"
    wind.field.strength = 26.0
    bpy.ops.object.effector_add(type="TURBULENCE", location=(-0.55, 0.0, 0.95))
    turbulence = bpy.context.object
    turbulence.name = "SSBL_Demo_Turbulence_Field"
    turbulence.field.strength = 18.0
    turbulence.field.size = 1.35
    turbulence.field.flow = 1.1
    turbulence.field.noise = 1.7
    turbulence.field.seed = 11

    before = _mesh_snapshot(cloth)
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    session = ssbl.solver.start_preview(bpy.context, cloth)

    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    overlay_text_frames: list[str] = []
    finite = True
    simulation_elapsed = 0.0
    max_avg_x = _average_x(cloth)
    start_avg_z = _average_world_z(cloth)
    max_avg_z = start_avg_z
    final_avg_z = start_avg_z
    force_field_counts: list[int] = []
    max_turbulence_strength = float(turbulence.field.strength)

    def _gust_curve(value: float) -> float:
        return 0.5 + 0.5 * math.sin(value)

    for frame in range(0, frame_count + 1):
        progress = frame / max(frame_count, 1)
        gust = _gust_curve(progress * math.tau * 3.5)
        turbulence.field.strength = 6.0 + 10.0 * gust
        turbulence.field.size = 0.85 + 0.35 * (1.0 - gust)
        turbulence.field.flow = 0.55 + 0.55 * _gust_curve(progress * math.tau * 2.2 + 0.9)
        turbulence.field.noise = 0.80 + 0.65 * _gust_curve(progress * math.tau * 4.8 + 1.7)
        turbulence.location.y = math.sin(progress * math.tau * 1.6) * 0.32
        turbulence.location.z = 0.88 + math.cos(progress * math.tau * 1.9) * 0.04
        wind.field.strength = 18.0 + 20.0 * _gust_curve(progress * math.tau * 2.8 + 0.4)
        if frame >= frame_count * 0.72:
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
        final_avg_z = _average_world_z(cloth)
        max_avg_z = max(max_avg_z, final_avg_z)
        force_field_counts.append(int(diag.force_field_count))
        max_turbulence_strength = max(max_turbulence_strength, float(turbulence.field.strength))
        metrics_line = _format_metrics_line(frame, frame_count, last_ms, native_ms)
        _update_overlay(
            overlay,
            metrics_line=metrics_line,
            note=_keyword_line(*keywords),
        )
        overlay_text_frames.append(_compose_overlay_text(title, metrics_line, _keyword_line(*keywords)))
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    ssbl.solver.request_stop(cloth)
    restore_delta = _mesh_delta(cloth, before)
    _encode_video(
        frames_dir,
        video_path,
        overlay_text_frames,
    )
    return _summarize_demo(
        name=name,
        title=title,
        source_repo=SOURCE_FLAGWAVER,
        source_scene="Flagwaver-style wind flag shot adapted to SSBL cloth controls",
        keywords=keywords,
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
            "final_turbulence_strength": float(turbulence.field.strength),
            "max_turbulence_strength": max_turbulence_strength,
            "start_average_z": start_avg_z,
            "max_average_z": max_avg_z,
            "final_average_z": final_avg_z,
            "final_hardness": float(cloth.ssbl_cloth.hardness),
        },
    )


def _smoothstep(value: float) -> float:
    clamped = max(0.0, min(1.0, float(value)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _make_spiral_floor_visual() -> None:
    floor = _plane_mesh(
        "SSBL_Demo_Spiral_Floor",
        [(-3.6, -3.0, 0.0), (3.6, -3.0, 0.0), (3.6, 3.0, 0.0), (-3.6, 3.0, 0.0)],
        (0.0, 0.0, 0.0),
        (0.085, 0.090, 0.102, 1.0),
    )
    floor.show_wire = False


def _make_spiral_drop_cloth(name: str, color: tuple[float, float, float, float]) -> bpy.types.Object:
    segments = 42
    size = 2.18
    start_angle = -0.45
    start_radius = 0.82
    center = (
        math.cos(start_angle) * start_radius,
        math.sin(start_angle) * start_radius * 0.72,
        2.28,
    )
    half = size * 0.5
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for y in range(segments + 1):
        fy = -half + size * y / segments
        for x in range(segments + 1):
            fx = -half + size * x / segments
            radial = max(abs(fx), abs(fy)) / max(half, 1.0e-6)
            edge_lift = 0.035 * _smoothstep((radial - 0.58) / 0.42)
            # A small asymmetric rest ripple helps the free settling keep soft folds.
            fz = 0.026 * math.sin(fx * 4.2) * math.cos(fy * 3.5)
            fz += edge_lift * math.sin(fx * 2.7 + fy * 1.9)
            verts.append((fx, fy, fz))
    stride = segments + 1
    for y in range(segments):
        for x in range(segments):
            base = y * stride + x
            faces.append((base, base + 1, base + stride + 1, base + stride))

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    obj.rotation_euler = (0.0, 0.0, math.radians(8.0))
    obj.data.materials.append(_material(f"{name}_Mat", color))
    bpy.context.scene.collection.objects.link(obj)

    pin = obj.vertex_groups.new(name="ssbl_pin")
    for vert in obj.data.vertices:
        fx = float(vert.co.x)
        fy = float(vert.co.y)
        edge = max(abs(fx), abs(fy)) / max(half, 1.0e-6)
        corner = min(abs(fx), abs(fy)) / max(half, 1.0e-6)
        near_center = math.hypot(fx, fy) <= size * 0.075
        if edge < 0.60 and not near_center:
            continue
        weight = 0.22 + 0.30 * _smoothstep((edge - 0.60) / 0.40)
        if corner > 0.70:
            weight += 0.08
        if near_center:
            weight = 0.18
        pin.add([vert.index], min(weight, 0.66), "ADD")
    _beautify_cloth(obj, levels=1)
    return obj


def _spiral_drop_pin_targets(cloth, frame: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    pin_indices = np.asarray(cloth.pin_indices, dtype=np.intp)
    rest = np.asarray(cloth.positions_world[pin_indices], dtype=np.float32)
    rest_center = np.mean(rest, axis=0) if len(rest) else np.zeros(3, dtype=np.float32)
    local = rest - rest_center

    progress = max(0.0, min(1.0, float(frame) / max(float(frame_count), 1.0)))
    guide_end = 0.54
    gather_start = 0.32
    gather_end = 0.76
    release_start = 0.70
    release_end = 0.88
    guide = _smoothstep(progress / guide_end)
    drop = _smoothstep(progress / 0.62)
    gather = _smoothstep((progress - gather_start) / max(gather_end - gather_start, 1.0e-6))
    release = _smoothstep((progress - release_start) / max(release_end - release_start, 1.0e-6))

    start_angle = -0.45
    angle = start_angle + math.tau * 1.42 * guide
    radius = 0.82 + (0.10 - 0.82) * guide
    center = np.array(
        (
            math.cos(angle) * radius,
            math.sin(angle) * radius * 0.72,
            2.28 + (0.48 - 2.28) * drop,
        ),
        dtype=np.float32,
    )

    spin = math.radians(8.0) + math.tau * 1.18 * guide + math.radians(62.0) * gather
    cos_spin = math.cos(spin)
    sin_spin = math.sin(spin)
    targets = np.empty_like(rest, dtype=np.float32)
    phase = math.tau * 2.2 * guide + math.tau * 0.65 * gather
    wave_strength = (0.13 + 0.16 * gather) * (1.0 - release)
    for index, point in enumerate(local):
        x = float(point[0])
        y = float(point[1])
        radial = min(1.25, math.hypot(x, y) / 1.30)
        edge = _smoothstep((radial - 0.32) / 0.68)
        fold_scale = 1.0 - gather * edge * 0.62
        rotated_x = cos_spin * x - sin_spin * y
        rotated_y = sin_spin * x + cos_spin * y
        rotated_x *= fold_scale
        rotated_y *= fold_scale
        curl = gather * edge * 0.18
        rotated_x += curl * math.sin(y * 5.2 + phase)
        rotated_y += curl * math.cos(x * 4.7 - phase * 0.8)
        ripple = wave_strength * math.sin(x * 4.1 + phase) * math.cos(y * 3.4 - phase * 0.65)
        fold_lift = gather * edge * (0.10 + 0.17 * (0.5 + 0.5 * math.sin(math.atan2(y, x) * 4.0 + phase)))
        floor_bias = 0.08 * gather * edge
        targets[index] = (
            center[0] + rotated_x,
            center[1] + rotated_y,
            max(0.10, center[2] + float(point[2]) + ripple + fold_lift - floor_bias),
        )

    weights = np.asarray(cloth.pin_weights, dtype=np.float32) * (1.0 - release)
    if progress >= release_end:
        weights = np.zeros_like(weights, dtype=np.float32)
    return np.ascontiguousarray(targets, dtype=np.float32), np.ascontiguousarray(weights, dtype=np.float32)


def _step_spiral_drop_frame(session, obj: bpy.types.Object, frame: int, frame_count: int):
    slot = session.slots[obj.name]
    if len(slot.cloth.pin_indices) > 0:
        pin_targets, pin_weights = _spiral_drop_pin_targets(slot.cloth, frame, frame_count)
        slot.native.update_pin_targets(slot.cloth.pin_indices, pin_targets, pin_weights)
        slot.pin_targets_world = np.array(pin_targets, dtype=np.float32, copy=True)
    slot.native.step(slot.substeps, slot.iterations)
    slot.current_positions_world = np.array(slot.native.download_positions(), dtype=np.float32, copy=True)
    _apply_world_positions(obj, slot.current_positions_world, slot.cloth.matrix_world_inv)
    return slot.native.cached_diagnostics()


def _record_tshirt_drape() -> DemoResult:
    name = "05_tshirt_drape_realtime"
    title = "SSBL realtime spiral cloth drop - floor pile"
    keywords = [
        "Spiral Cloth Drop",
        "Ground Contact",
        "Stable Floor Pile",
    ]
    frame_count = 144
    _clear_scene()
    _case_dir, frames_dir, video_path = _ensure_output_dir(name)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = _configure_scene_render(
        scene,
        camera_location=(4.20, -5.35, 2.85),
        target=(0.0, -0.04, 1.18),
        ortho_scale=5.05,
    )
    overlay = _create_overlay(scene, camera, title)

    _make_spiral_floor_visual()
    cloth = _make_spiral_drop_cloth("SSBL_Demo_Tshirt", (0.86, 0.48, 0.62, 1.0))
    _configure_collision_settings(cloth.ssbl_cloth, frame_count=frame_count)
    cloth.ssbl_cloth.gravity = (0.0, 0.0, -9.8)
    cloth.ssbl_cloth.pin_vertex_group = "ssbl_pin"
    cloth.ssbl_cloth.use_ground = True
    cloth.ssbl_cloth.ground_height = 0.0
    cloth.ssbl_cloth.self_collision = True
    cloth.ssbl_cloth.self_collision_mode = "fast"
    cloth.ssbl_cloth.self_collision_interval = 1
    cloth.ssbl_cloth.max_self_collision_neighbors = 96
    cloth.ssbl_cloth.fast_self_collision_passes = 6
    cloth.ssbl_cloth.static_collider_collection = None
    cloth.ssbl_cloth.damping = 0.992
    cloth.ssbl_cloth.hardness = 0.34
    cloth.ssbl_cloth.hardness_initialized = True
    cloth.ssbl_cloth.collision_margin = 0.032
    cloth.ssbl_cloth.cloth_thickness = 0.032
    cloth.ssbl_cloth.substeps = 16
    cloth.ssbl_cloth.iterations = 4
    cloth.ssbl_cloth.contact_friction = 0.94
    cloth.ssbl_cloth.contact_tangent_damping = 0.72

    before = _mesh_snapshot(cloth)
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    session = ssbl.solver.start_preview(bpy.context, cloth)

    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    overlay_text_frames: list[str] = []
    finite = True
    simulation_elapsed = 0.0
    max_static_collision_ms = 0.0
    min_cloth_z = float("inf")
    final_min_z = 0.0
    final_height_span = 0.0
    max_below_ground_depth = 0.0
    max_x_span = 0.0

    for frame in range(0, frame_count + 1):
        last_ms = 0.0
        native_ms = 0.0
        diag = ssbl.solver.session_diagnostics(cloth)
        if frame > 0:
            started = time.perf_counter()
            scene.frame_set(frame)
            diag = _step_spiral_drop_frame(session, cloth, frame, frame_count)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            native_ms = float(diag.step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and _finite_mesh(cloth) and bool(diag.finite)
        positions = np.asarray(session.slots[cloth.name].current_positions_world, dtype=np.float64)
        if len(positions):
            current_min_z = float(np.min(positions[:, 2]))
            current_max_z = float(np.max(positions[:, 2]))
            min_cloth_z = min(min_cloth_z, current_min_z)
            final_min_z = current_min_z
            final_height_span = current_max_z - current_min_z
            max_below_ground_depth = max(max_below_ground_depth, max(0.0, -current_min_z))
            max_x_span = max(max_x_span, float(np.max(positions[:, 0]) - np.min(positions[:, 0])))
        max_static_collision_ms = max(max_static_collision_ms, float(diag.static_collision_ms))
        metrics_line = _format_metrics_line(frame, frame_count, last_ms, native_ms)
        _update_overlay(
            overlay,
            metrics_line=metrics_line,
            note=_keyword_line(*keywords),
        )
        overlay_text_frames.append(_compose_overlay_text(title, metrics_line, _keyword_line(*keywords)))
        frame_paths.append(_render_frame(scene, frames_dir, frame))

    ssbl.solver.request_stop(cloth)
    restore_delta = _mesh_delta(cloth, before)
    _encode_video(
        frames_dir,
        video_path,
        overlay_text_frames,
    )
    return _summarize_demo(
        name=name,
        title=title,
        source_repo=SOURCE_DIFFCLOTH,
        source_scene="Custom spiral-guided cloth drop onto an SSBL ground plane",
        keywords=keywords,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        step_ms_samples=step_ms_samples,
        simulation_elapsed=simulation_elapsed,
        finite=finite,
        restore_delta=restore_delta,
        metrics={
            "slots": len(session.slots),
            "max_static_collision_ms": max_static_collision_ms,
            "min_cloth_z": min_cloth_z,
            "final_min_z": final_min_z,
            "max_below_ground_depth": max_below_ground_depth,
            "final_height_span": final_height_span,
            "self_collision_mode": str(cloth.ssbl_cloth.self_collision_mode),
            "max_x_span": max_x_span,
        },
    )


def _make_contact_sheet(results: list[DemoResult]) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    columns = 3
    rows = max(1, int(math.ceil((len(results) * 3) / columns)))
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
            f"scale=320:180,tile={columns}x{rows}",
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
            _record_force_field_tuning,
            _record_wring_towel,
            _record_multicloth_contact,
            _record_object_collision_suite,
            _record_tshirt_drape,
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
        if len(results) != 5 or not all(result.validation_passed for result in results):
            raise RuntimeError("Realtime demo pack did not generate all five validated videos")
    finally:
        try:
            ssbl.solver.cleanup_all_sessions()
        except Exception:
            pass
        ssbl.unregister()


if __name__ == "__main__":
    main()
