from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import bpy
from mathutils import Vector
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
SSBL_ROOT = TOOLS_DIR.parent
ADDONS_ROOT = SSBL_ROOT.parent
if str(ADDONS_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDONS_ROOT))


BLEND_PATH = Path(os.environ.get("SSBL_CS3_BLEND", r"C:\Users\Administrator\Desktop\cs3.blend"))
OUTPUT_DIR = Path(os.environ.get("SSBL_CS3_RECORD_DIR", SSBL_ROOT / "recordings" / "cs3_boiling_check"))
FRAME_COUNT = max(int(os.environ.get("SSBL_CS3_RECORD_FRAMES", "200")), 1)
VIDEO_FPS = max(int(os.environ.get("SSBL_CS3_VIDEO_FPS", "24")), 1)
MAX_EDGE_RATIO_LIMIT = float(os.environ.get("SSBL_CS3_MAX_EDGE_RATIO", "1.25"))
MEAN_RMS_STEP_LIMIT = float(os.environ.get("SSBL_CS3_MEAN_RMS_STEP", "0.015"))
HARDNESS1_MAX_EDGE_RATIO_LIMIT = float(os.environ.get("SSBL_CS3_HARDNESS1_MAX_EDGE_RATIO", "1.35"))
HARDNESS1_MEAN_RMS_STEP_LIMIT = float(os.environ.get("SSBL_CS3_HARDNESS1_MEAN_RMS_STEP", "0.025"))
REVERSE_JITTER_FRACTION_LIMIT = float(os.environ.get("SSBL_CS3_REVERSE_JITTER_FRACTION", "0.015"))
REVERSE_JITTER_PEAK_FRAME_FRACTION_LIMIT = float(
    os.environ.get("SSBL_CS3_REVERSE_JITTER_PEAK_FRAME_FRACTION", "0.06")
)
RESTORE_TOLERANCE = float(os.environ.get("SSBL_CS3_RESTORE_TOLERANCE", "1.0e-7"))
SKIP_RENDER = os.environ.get("SSBL_CS3_SKIP_RENDER", "0").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_JITTER = os.environ.get("SSBL_CS3_DEBUG_JITTER", "0").strip().lower() in {"1", "true", "yes", "on"}
FORCE_OBJECT_HARDNESS_RAW = os.environ.get("SSBL_CS3_FORCE_OBJECT_HARDNESS")
FORCE_OBJECT_HARDNESS = (
    None
    if FORCE_OBJECT_HARDNESS_RAW is None or FORCE_OBJECT_HARDNESS_RAW.strip() == ""
    else float(FORCE_OBJECT_HARDNESS_RAW)
)


def _run_checked(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _setup_render(scene: bpy.types.Scene, obj: bpy.types.Object) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = VIDEO_FPS
    scene.render.image_settings.file_format = "PNG"
    if scene.world is not None:
        scene.world.color = (0.025, 0.03, 0.035)

    if not obj.data.materials:
        material = bpy.data.materials.new("SSBL_CS3_Cloth_Mat")
        material.diffuse_color = (0.58, 0.72, 0.96, 1.0)
        obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = True

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector()) / max(len(corners), 1)
    radius = max((corner - center).length for corner in corners) if corners else 2.0

    camera_data = bpy.data.cameras.new("SSBL_CS3_Check_Camera")
    camera = bpy.data.objects.new("SSBL_CS3_Check_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((2.6, -3.2, 1.9))
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(radius * 2.35, 3.0)
    _look_at(camera, center)
    scene.camera = camera

    light_data = bpy.data.lights.new("SSBL_CS3_Check_Key", "AREA")
    light = bpy.data.objects.new("SSBL_CS3_Check_Key", light_data)
    scene.collection.objects.link(light)
    light.location = center + Vector((0.8, -2.2, 3.0))
    light_data.energy = 450
    light_data.size = 4.0


def _render_frame(scene: bpy.types.Scene, frames_dir: Path, frame_index: int) -> str:
    path = frames_dir / f"frame_{frame_index:04d}.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return str(path)


def _encode_video(frames_dir: Path, video_path: Path, frame_count: int) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    _run_checked(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(VIDEO_FPS),
            "-start_number",
            "1",
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
    )
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg did not produce a non-empty video: {video_path}")
    return str(video_path)


def _snapshot_source_mesh(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in obj.data.vertices]


def _restore_delta(obj: bpy.types.Object, source_mesh: bpy.types.Mesh, before: list[tuple[float, float, float]]) -> float:
    if obj.data is not source_mesh or len(source_mesh.vertices) != len(before):
        return float("inf")
    return max(
        (
            max(
                abs(float(vertex.co.x) - old[0]),
                abs(float(vertex.co.y) - old[1]),
                abs(float(vertex.co.z) - old[2]),
            )
            for vertex, old in zip(source_mesh.vertices, before)
        ),
        default=0.0,
    )


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


def _active_settings(scene: bpy.types.Scene, obj: bpy.types.Object):
    if hasattr(obj, "ssbl_cloth") and bool(getattr(obj.ssbl_cloth, "enabled", False)):
        return obj.ssbl_cloth, "object"
    return scene.ssbl_preview, "scene"


def _settings_snapshot(settings, scope: str) -> dict:
    snapshot = {"settings_scope": scope}
    for key in ("hardness", "use_lra", "lra_compliance", "lra_slack", "stretch_compliance", "bend_compliance"):
        if not hasattr(settings, key):
            continue
        value = getattr(settings, key)
        if isinstance(value, bool):
            snapshot[key] = bool(value)
        else:
            try:
                snapshot[key] = float(value)
            except (TypeError, ValueError):
                snapshot[key] = str(value)
    return snapshot


def _summary_failures(summary: dict) -> list[str]:
    failures: list[str] = []
    limits = summary.get("limits", {})
    max_edge_ratio_limit = float(limits.get("max_edge_ratio", MAX_EDGE_RATIO_LIMIT))
    mean_rms_step_limit = float(limits.get("mean_rms_step", MEAN_RMS_STEP_LIMIT))
    reverse_jitter_fraction_limit = float(limits.get("reverse_jitter_fraction", REVERSE_JITTER_FRACTION_LIMIT))
    reverse_jitter_peak_limit = float(
        limits.get("reverse_jitter_peak_frame_fraction", REVERSE_JITTER_PEAK_FRAME_FRACTION_LIMIT)
    )
    if int(summary["frames_sampled"]) != int(FRAME_COUNT):
        failures.append(f"frames_sampled {summary['frames_sampled']} != {FRAME_COUNT}")
    if not bool(summary["finite"]):
        failures.append("simulation produced non-finite vertices")
    if float(summary["restore_delta"]) > RESTORE_TOLERANCE:
        failures.append(f"restore_delta {summary['restore_delta']} > {RESTORE_TOLERANCE}")
    if not bool(summary["skip_render"]) and int(summary["video_bytes"]) <= 0:
        failures.append("video was not created or is empty")
    if float(summary["max_edge_ratio"]) > max_edge_ratio_limit:
        failures.append(f"max_edge_ratio {summary['max_edge_ratio']} > {max_edge_ratio_limit}")
    if float(summary["mean_rms_step"]) > mean_rms_step_limit:
        failures.append(f"mean_rms_step {summary['mean_rms_step']} > {mean_rms_step_limit}")
    if float(summary["reverse_jitter_fraction"]) > reverse_jitter_fraction_limit:
        failures.append(f"reverse_jitter_fraction {summary['reverse_jitter_fraction']} > {reverse_jitter_fraction_limit}")
    if float(summary["reverse_jitter_peak_frame_fraction"]) > reverse_jitter_peak_limit:
        failures.append(
            f"reverse_jitter_peak_frame_fraction {summary['reverse_jitter_peak_frame_fraction']} > {reverse_jitter_peak_limit}"
        )
    return failures


def main() -> None:
    if not BLEND_PATH.exists():
        raise RuntimeError(f"CS3 blend file not found: {BLEND_PATH}")

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH), load_ui=False)
    import ssbl
    from ssbl.xpbd_core import sync_hardness_settings

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUTPUT_DIR / "cs3_boiling_check.mp4"
    if video_path.exists():
        video_path.unlink()

    scene = bpy.context.scene
    obj = bpy.data.objects.get("Plane")
    if obj is None or obj.type != "MESH":
        meshes = [candidate for candidate in scene.objects if candidate.type == "MESH"]
        if not meshes:
            raise RuntimeError("CS3 scene has no mesh object to simulate")
        obj = meshes[0]

    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_viewport = False
    obj.hide_render = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    settings, settings_scope = _active_settings(scene, obj)
    if FORCE_OBJECT_HARDNESS is not None:
        if hasattr(obj, "ssbl_cloth"):
            settings = obj.ssbl_cloth
            settings.enabled = True
            settings_scope = "object"
        settings.hardness = float(FORCE_OBJECT_HARDNESS)
        settings.hardness_initialized = True
        sync_hardness_settings(settings)
    settings.frame_count = FRAME_COUNT
    settings.preview_writeback_interval = 1
    settings_snapshot = _settings_snapshot(settings, settings_scope)
    forced_hardness = FORCE_OBJECT_HARDNESS is not None
    max_edge_ratio_limit = HARDNESS1_MAX_EDGE_RATIO_LIMIT if forced_hardness else MAX_EDGE_RATIO_LIMIT
    mean_rms_step_limit = HARDNESS1_MEAN_RMS_STEP_LIMIT if forced_hardness else MEAN_RMS_STEP_LIMIT
    scene.frame_start = 1
    scene.frame_end = max(int(scene.frame_end), FRAME_COUNT + 1)
    scene.frame_set(1)
    _setup_render(scene, obj)

    source_mesh = obj.data
    source_before = _snapshot_source_mesh(obj)
    session = ssbl.solver.start_preview(bpy.context, obj)
    slot = session.slots[obj.name]
    previous_positions = np.asarray(slot.current_positions_world, dtype=np.float64).copy()
    previous_velocity: np.ndarray | None = None
    previous_reversal_mask: np.ndarray | None = None
    dynamic_mask = np.asarray(slot.cloth.inv_mass, dtype=np.float64) > 0.0
    if len(dynamic_mask) != len(previous_positions) or not np.any(dynamic_mask):
        dynamic_mask = np.ones(len(previous_positions), dtype=bool)
    dynamic_count = int(np.count_nonzero(dynamic_mask))

    finite = True
    max_step = 0.0
    raw_max_step = 0.0
    max_accel = 0.0
    rms_steps: list[float] = []
    raw_rms_steps: list[float] = []
    rms_accels: list[float] = []
    reverse_jitter_samples = 0
    reverse_jitter_total_samples = 0
    reverse_jitter_frame_fractions: list[float] = []
    reverse_jitter_sq_sum = 0.0
    reverse_jitter_count = 0
    pcg_iterations = 0
    pcg_solve_ms = 0.0
    pcg_system_ms = 0.0
    pcg_ad_ms = 0.0
    pcg_max_delta = 0.0
    abi41_lra_tack_count = 0
    abi41_tack_jitter_guarded = 0
    max_edge_ratio = 1.0
    frame_paths: list[str] = []
    frames_sampled = 0
    started = time.perf_counter()

    for frame in range(1, FRAME_COUNT + 1):
        finished = ssbl.solver.step_preview(bpy.context, obj.name)
        slot = session.slots[obj.name]
        positions = np.asarray(slot.current_positions_world, dtype=np.float64).copy()
        delta = positions - previous_positions
        raw_step_lengths = np.linalg.norm(delta, axis=1)
        local_delta = delta - np.mean(delta, axis=0, keepdims=True) if len(delta) else delta
        step_lengths = np.linalg.norm(local_delta, axis=1)
        velocity = local_delta
        if previous_velocity is not None and len(local_delta) == len(previous_velocity):
            current_mag = np.linalg.norm(local_delta, axis=1)
            previous_mag = np.linalg.norm(previous_velocity, axis=1)
            reversal_dot = np.einsum("ij,ij->i", local_delta, previous_velocity, optimize=True)
            reversal_mask = (
                (reversal_dot < (-0.25 * current_mag * previous_mag))
                & (np.minimum(current_mag, previous_mag) >= 0.001)
                & dynamic_mask
            )
            if previous_reversal_mask is not None and len(previous_reversal_mask) == len(reversal_mask):
                jitter_mask = reversal_mask & previous_reversal_mask
                jitter_count = int(np.count_nonzero(jitter_mask))
                reverse_jitter_samples += jitter_count
                reverse_jitter_total_samples += dynamic_count
                reverse_jitter_frame_fractions.append(float(jitter_count) / float(max(dynamic_count, 1)))
                if jitter_count > 0:
                    jitter_amplitude = np.minimum(current_mag, previous_mag)[jitter_mask]
                    reverse_jitter_sq_sum += float(np.sum(jitter_amplitude * jitter_amplitude))
                    reverse_jitter_count += jitter_count
            previous_reversal_mask = reversal_mask
        if previous_velocity is not None:
            accel = np.linalg.norm(velocity - previous_velocity, axis=1)
            max_accel = max(max_accel, float(np.max(accel)))
            rms_accels.append(float(np.sqrt(np.mean(accel * accel))))
        max_step = max(max_step, float(np.max(step_lengths)))
        raw_max_step = max(raw_max_step, float(np.max(raw_step_lengths)))
        rms_steps.append(float(np.sqrt(np.mean(step_lengths * step_lengths))))
        raw_rms_steps.append(float(np.sqrt(np.mean(raw_step_lengths * raw_step_lengths))))
        max_edge_ratio = max(max_edge_ratio, _slot_max_edge_ratio(slot))

        diag = ssbl.solver.session_diagnostics(obj)
        pcg_iterations += int(getattr(diag, "abi41_pcg_iterations", 0))
        pcg_solve_ms += float(getattr(diag, "abi41_pcg_solve_ms", 0.0))
        pcg_system_ms += float(getattr(diag, "abi41_pcg_system_ms", 0.0))
        pcg_ad_ms += float(getattr(diag, "abi41_pcg_ad_ms", 0.0))
        pcg_max_delta = max(pcg_max_delta, float(getattr(diag, "abi41_pcg_max_delta", 0.0)))
        abi41_lra_tack_count += int(getattr(diag, "abi41_lra_tack_count", 0))
        abi41_tack_jitter_guarded += int(getattr(diag, "abi41_tack_jitter_guarded", 0))
        finite = finite and bool(diag.finite) and bool(np.all(np.isfinite(positions)))
        frames_sampled += 1
        if not SKIP_RENDER:
            frame_paths.append(_render_frame(scene, frames_dir, frame))
        previous_positions = positions
        previous_velocity = velocity
        if finished:
            break

    stopped = ssbl.solver.request_stop(obj)
    restore_delta = _restore_delta(obj, source_mesh, source_before)
    video = "" if SKIP_RENDER else _encode_video(frames_dir, video_path, frames_sampled)
    elapsed_s = time.perf_counter() - started

    summary = {
        "blend_file": bpy.data.filepath,
        "object": obj.name,
        "output_dir": str(OUTPUT_DIR),
        "frames_dir": str(frames_dir),
        "video": video,
        "video_bytes": int(video_path.stat().st_size) if video_path.exists() else 0,
        "skip_render": bool(SKIP_RENDER),
        "frames_requested": int(FRAME_COUNT),
        "frames_sampled": int(frames_sampled),
        "simulation_and_render_elapsed_s": float(elapsed_s),
        "finite": bool(finite),
        "stopped": bool(stopped),
        "restore_delta": float(restore_delta),
        "max_edge_ratio": float(max_edge_ratio),
        "max_step": float(max_step),
        "mean_rms_step": float(np.mean(rms_steps)) if rms_steps else 0.0,
        "raw_max_step": float(raw_max_step),
        "raw_mean_rms_step": float(np.mean(raw_rms_steps)) if raw_rms_steps else 0.0,
        "step_metric": "centroid_removed_local_deformation",
        "reverse_jitter_metric": "centroid_removed_local_deformation",
        "max_accel": float(max_accel),
        "mean_rms_accel": float(np.mean(rms_accels)) if rms_accels else 0.0,
        "reverse_jitter_fraction": (
            float(reverse_jitter_samples) / float(reverse_jitter_total_samples)
            if reverse_jitter_total_samples > 0
            else 0.0
        ),
        "reverse_jitter_peak_frame_fraction": (
            float(np.max(reverse_jitter_frame_fractions)) if reverse_jitter_frame_fractions else 0.0
        ),
        "reverse_jitter_rms": (
            float(np.sqrt(reverse_jitter_sq_sum / float(reverse_jitter_count)))
            if reverse_jitter_count > 0
            else 0.0
        ),
        "pcg_iterations": int(pcg_iterations),
        "pcg_solve_ms": float(pcg_solve_ms),
        "pcg_system_ms": float(pcg_system_ms),
        "pcg_ad_ms": float(pcg_ad_ms),
        "pcg_max_delta": float(pcg_max_delta),
        "abi41_lra_tack_count": int(abi41_lra_tack_count),
        "abi41_tack_jitter_guarded": int(abi41_tack_jitter_guarded),
        "limits": {
            "max_edge_ratio": float(max_edge_ratio_limit),
            "mean_rms_step": float(mean_rms_step_limit),
            "reverse_jitter_fraction": float(REVERSE_JITTER_FRACTION_LIMIT),
            "reverse_jitter_peak_frame_fraction": float(REVERSE_JITTER_PEAK_FRAME_FRACTION_LIMIT),
            "restore_delta": float(RESTORE_TOLERANCE),
        },
        "frame_paths": frame_paths,
    }
    if DEBUG_JITTER:
        summary["reverse_jitter_frame_fractions"] = [float(value) for value in reverse_jitter_frame_fractions]
    summary.update(settings_snapshot)
    failures = _summary_failures(summary)
    summary["passed"] = not failures
    summary["failures"] = failures
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    public_summary = {k: v for k, v in summary.items() if k != "frame_paths"}
    marker = "SSBL_CS3_BOILING_CHECK" if not failures else "SSBL_CS3_BOILING_CHECK_FAIL"
    print(marker + " " + json.dumps(public_summary, ensure_ascii=False, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"SSBL_CS3_BOILING_CHECK_ERROR {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
