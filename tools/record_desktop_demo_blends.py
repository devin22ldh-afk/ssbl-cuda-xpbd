from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
SSBL_ROOT = TOOLS_DIR.parent
ADDONS_ROOT = SSBL_ROOT.parent
if str(ADDONS_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDONS_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import record_realtime_demo_pack as demo
import ssbl
from ssbl.session_manager import _apply_world_positions


DESKTOP_BLEND_DIR = Path(r"C:\Users\Administrator\Desktop\演示视频")

KEYWORDS = {
    "01_brand_flag_wind_realtime": ["Realtime Wind", "Turbulent Gusts", "Live Tuning"],
    "02_wring_towel_realtime": ["Hook Driven Wring", "Stable Self-Collision", "Twist Deformation"],
    "03_clothesline_multicloth_realtime": ["Remeshed Suzanne", "Inflated Multi-Cloth", "Closed-Shell Collision"],
    "04_tablecloth_pull_collision_realtime": ["Pinned Corner Pull", "Rigid Edge Collision", "Stable Drape"],
    "05_tshirt_drape_realtime": ["Spiral Cloth Drop", "Ground Contact", "Stable Floor Pile"],
}

SOURCE_REPOS = {
    "01_brand_flag_wind_realtime": demo.SOURCE_FLAGWAVER,
    "02_wring_towel_realtime": demo.SOURCE_BINROOT,
    "03_clothesline_multicloth_realtime": "",
    "04_tablecloth_pull_collision_realtime": demo.SOURCE_GARMENTLAB,
    "05_tshirt_drape_realtime": demo.SOURCE_DIFFCLOTH,
}


def _open_demo_blend(name: str) -> Path:
    path = DESKTOP_BLEND_DIR / f"{name}.blend"
    if not path.exists():
        raise RuntimeError(f"Missing desktop demo blend: {path}")
    try:
        ssbl.solver.cleanup_all_sessions()
    except Exception:
        pass
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x = demo.RESOLUTION[0]
    scene.render.resolution_y = demo.RESOLUTION[1]
    scene.render.fps = demo.VIDEO_FPS
    scene.render.image_settings.file_format = "PNG"
    scene.frame_set(int(scene.frame_start))
    return path


def _frame_count(scene: bpy.types.Scene, fallback: int) -> int:
    value = int(scene.get("ssbl_demo_frame_count", int(fallback) + 1))
    return max(value - 1, 1)


def _title(scene: bpy.types.Scene, fallback: str) -> str:
    return str(scene.get("ssbl_demo_title", fallback))


def _source_scene(scene: bpy.types.Scene, blend_path: Path) -> str:
    source = str(scene.get("ssbl_demo_source", "Desktop editable demo blend"))
    return f"{source}; source_blend={blend_path}"


def _overlay(scene: bpy.types.Scene) -> demo.Overlay:
    return demo.Overlay(
        title=bpy.data.objects["SSBL_Demo_Overlay_Title"],
        metrics=bpy.data.objects["SSBL_Demo_Overlay_Metrics"],
        notes=bpy.data.objects["SSBL_Demo_Overlay_Notes"],
    )


def _active_mesh_or_named(*names: str) -> bpy.types.Object:
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "MESH":
            return obj
    obj = bpy.context.view_layer.objects.active
    if obj is not None and obj.type == "MESH":
        return obj
    for candidate in bpy.data.objects:
        if candidate.type == "MESH" and hasattr(candidate, "ssbl_cloth") and bool(candidate.ssbl_cloth.enabled):
            return candidate
    raise RuntimeError("No mesh cloth object found in desktop blend.")


def _select_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _render_case_frame(
    scene: bpy.types.Scene,
    frames_dir: Path,
    frame_paths: list[str],
    overlay_text_frames: list[str],
    overlay: demo.Overlay,
    title: str,
    keywords: list[str],
    frame: int,
    frame_count: int,
    last_ms: float,
    native_ms: float,
) -> None:
    metrics_line = demo._format_metrics_line(frame, frame_count, last_ms, native_ms)
    note = demo._keyword_line(*keywords)
    demo._update_overlay(overlay, metrics_line=metrics_line, note=note)
    overlay_text_frames.append(demo._compose_overlay_text(title, metrics_line, note))
    frame_paths.append(demo._render_frame(scene, frames_dir, frame))


def _finish(
    *,
    name: str,
    title: str,
    blend_path: Path,
    keywords: list[str],
    video_path: Path,
    frames_dir: Path,
    frame_paths: list[str],
    step_ms_samples: list[float],
    simulation_elapsed: float,
    finite: bool,
    restore_delta: float,
    metrics: dict[str, object],
) -> demo.DemoResult:
    demo._encode_video(frames_dir, video_path, overlay_text_frames=_OVERLAY_TEXT[name])
    return demo._summarize_demo(
        name=name,
        title=title,
        source_repo=SOURCE_REPOS.get(name, ""),
        source_scene=_source_scene(bpy.context.scene, blend_path),
        keywords=keywords,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        step_ms_samples=step_ms_samples,
        simulation_elapsed=simulation_elapsed,
        finite=finite,
        restore_delta=restore_delta,
        metrics=metrics,
    )


_OVERLAY_TEXT: dict[str, list[str]] = {}


def _record_flag() -> demo.DemoResult:
    name = "01_brand_flag_wind_realtime"
    blend_path = _open_demo_blend(name)
    scene = bpy.context.scene
    frame_count = _frame_count(scene, 112)
    title = _title(scene, "SSBL realtime flag - live wind control")
    keywords = KEYWORDS[name]
    _case_dir, frames_dir, video_path = demo._ensure_output_dir(name)
    overlay = _overlay(scene)
    cloth = _active_mesh_or_named("SSBL_Demo_Force_Field_Flag")
    wind = bpy.data.objects["SSBL_Demo_Wind_Field"]
    turbulence = bpy.data.objects["SSBL_Demo_Turbulence_Field"]
    _select_active(cloth)
    before = demo._mesh_snapshot(cloth)
    session = ssbl.solver.start_preview(bpy.context, cloth)
    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    _OVERLAY_TEXT[name] = []
    simulation_elapsed = 0.0
    finite = True
    max_avg_x = demo._average_x(cloth)
    start_avg_z = demo._average_world_z(cloth)
    max_avg_z = start_avg_z
    final_avg_z = start_avg_z
    force_field_counts: list[int] = []
    max_turbulence_strength = float(turbulence.field.strength)

    def gust_curve(value: float) -> float:
        return 0.5 + 0.5 * math.sin(value)

    for frame in range(0, frame_count + 1):
        progress = frame / max(frame_count, 1)
        gust = gust_curve(progress * math.tau * 3.5)
        turbulence.field.strength = 6.0 + 10.0 * gust
        turbulence.field.size = 0.85 + 0.35 * (1.0 - gust)
        turbulence.field.flow = 0.55 + 0.55 * gust_curve(progress * math.tau * 2.2 + 0.9)
        turbulence.field.noise = 0.80 + 0.65 * gust_curve(progress * math.tau * 4.8 + 1.7)
        turbulence.location.y = math.sin(progress * math.tau * 1.6) * 0.32
        turbulence.location.z = 0.88 + math.cos(progress * math.tau * 1.9) * 0.04
        wind.field.strength = 18.0 + 20.0 * gust_curve(progress * math.tau * 2.8 + 0.4)
        if frame >= frame_count * 0.72:
            cloth.ssbl_cloth.hardness = 0.72
            cloth.ssbl_cloth.hardness_initialized = True
        last_ms = 0.0
        native_ms = 0.0
        diag = ssbl.solver.session_diagnostics(cloth)
        if frame > 0:
            started = time.perf_counter()
            ssbl.solver.step_preview(bpy.context, cloth.name)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            diag = ssbl.solver.session_diagnostics(cloth)
            native_ms = float(diag.step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and demo._finite_mesh(cloth) and bool(diag.finite)
        max_avg_x = max(max_avg_x, demo._average_x(cloth))
        final_avg_z = demo._average_world_z(cloth)
        max_avg_z = max(max_avg_z, final_avg_z)
        force_field_counts.append(int(diag.force_field_count))
        max_turbulence_strength = max(max_turbulence_strength, float(turbulence.field.strength))
        _render_case_frame(scene, frames_dir, frame_paths, _OVERLAY_TEXT[name], overlay, title, keywords, frame, frame_count, last_ms, native_ms)

    ssbl.solver.request_stop(cloth)
    restore_delta = demo._mesh_delta(cloth, before)
    return _finish(
        name=name,
        title=title,
        blend_path=blend_path,
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
            "max_force_field_count": max(force_field_counts) if force_field_counts else 0,
            "max_turbulence_strength": max_turbulence_strength,
            "final_wind_strength": float(wind.field.strength),
            "start_average_z": start_avg_z,
            "max_average_z": max_avg_z,
            "final_average_z": final_avg_z,
            "max_average_x": max_avg_x,
            "final_hardness": float(cloth.ssbl_cloth.hardness),
            "source_blend": str(blend_path),
        },
    )


def _record_wring() -> demo.DemoResult:
    name = "02_wring_towel_realtime"
    blend_path = _open_demo_blend(name)
    scene = bpy.context.scene
    frame_count = _frame_count(scene, demo.WRING_FRAME_COUNT)
    title = _title(scene, "SSBL realtime wring towel - hook driven twist")
    keywords = KEYWORDS[name]
    _case_dir, frames_dir, video_path = demo._ensure_output_dir(name)
    overlay = _overlay(scene)
    towel = _active_mesh_or_named("SSBL_Wring_Towel")
    _select_active(towel)
    source_mesh = towel.data
    source_before = demo._snapshot_coords(source_mesh)
    session = ssbl.solver.start_preview(bpy.context, towel)
    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    _OVERLAY_TEXT[name] = []
    finite = True
    min_z = float("inf")
    max_radius = 0.0
    simulation_elapsed = 0.0

    for frame in range(0, frame_count + 1):
        last_ms = 0.0
        native_ms = 0.0
        if frame > 0:
            started = time.perf_counter()
            demo._step_wring_frame(session, towel, frame, frame_count)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            native_ms = float(session.slots[towel.name].native.cached_diagnostics().step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and demo._wring_all_finite(towel)
        for vert in towel.data.vertices:
            min_z = min(min_z, float((towel.matrix_world @ vert.co).z))
            max_radius = max(max_radius, math.hypot(float(vert.co.y), float(vert.co.z - 1.25)))
        _render_case_frame(scene, frames_dir, frame_paths, _OVERLAY_TEXT[name], overlay, title, keywords, frame, frame_count, last_ms, native_ms)

    tethers = len(session.cloth.lra_edges)
    ssbl.solver.request_stop(towel)
    restore_delta = demo._max_abs_delta(source_before, demo._snapshot_coords(source_mesh))
    if tethers != 0:
        raise RuntimeError(f"{name}: hardness=0 created hidden tethers: {tethers}")
    return _finish(
        name=name,
        title=title,
        blend_path=blend_path,
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
            "tethers": tethers,
            "min_z": min_z,
            "max_twist_radius": max_radius,
            "cloth_thickness": float(towel.ssbl_cloth.cloth_thickness if bool(towel.ssbl_cloth.enabled) else scene.ssbl_preview.cloth_thickness),
            "source_blend": str(blend_path),
        },
    )


def _record_timeline(name: str) -> demo.DemoResult:
    blend_path = _open_demo_blend(name)
    scene = bpy.context.scene
    fallback = 112 if name != "05_tshirt_drape_realtime" else 144
    frame_count = _frame_count(scene, fallback)
    title = _title(scene, name)
    keywords = KEYWORDS[name]
    _case_dir, frames_dir, video_path = demo._ensure_output_dir(name)
    overlay = _overlay(scene)
    scene.frame_set(int(scene.frame_start))
    before: dict[str, list[tuple[float, float, float]]] = {
        obj.name: demo._mesh_snapshot(obj)
        for obj in scene.objects
        if obj.type == "MESH" and hasattr(obj, "ssbl_cloth") and bool(obj.ssbl_cloth.enabled)
    }
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        raise RuntimeError(f"{name}: no enabled cloth objects in desktop blend")

    step_ms_samples: list[float] = []
    frame_paths: list[str] = []
    _OVERLAY_TEXT[name] = []
    finite = True
    simulation_elapsed = 0.0
    max_penetration = 0.0
    max_static_collision_ms = 0.0
    max_dynamic_triangles = 0
    max_displacement = 0.0

    for frame in range(0, frame_count + 1):
        last_ms = 0.0
        native_ms = 0.0
        diag = session.last_diagnostics
        if frame > 0:
            scene.frame_set(int(scene.frame_start) + frame)
            started = time.perf_counter()
            ssbl.solver.step_timeline_preview(bpy.context, scene)
            elapsed = time.perf_counter() - started
            simulation_elapsed += elapsed
            last_ms = elapsed * 1000.0
            diag = session.last_diagnostics
            native_ms = float(diag.step_ms)
            step_ms_samples.append(last_ms)
        finite = finite and bool(diag.finite)
        max_penetration = max(max_penetration, float(getattr(diag, "penetration_depth", 0.0)))
        max_static_collision_ms = max(max_static_collision_ms, float(getattr(diag, "static_collision_ms", 0.0)))
        max_dynamic_triangles = max(max_dynamic_triangles, int(getattr(diag, "dynamic_triangle_count", 0)))
        for slot_name, slot in session.slots.items():
            obj = bpy.data.objects.get(slot_name)
            if obj is None:
                continue
            finite = finite and demo._finite_mesh(obj)
            original = before.get(slot_name)
            if original is not None and len(original) == len(obj.data.vertices):
                max_displacement = max(max_displacement, demo._mesh_delta(obj, original))
        _render_case_frame(scene, frames_dir, frame_paths, _OVERLAY_TEXT[name], overlay, title, keywords, frame, frame_count, last_ms, native_ms)

    stopper = bpy.data.objects.get(session.object_name)
    if stopper is not None:
        ssbl.solver.request_stop(stopper)
    restore_delta = max(
        (demo._mesh_delta(bpy.data.objects[name], snapshot) for name, snapshot in before.items() if bpy.data.objects.get(name) is not None),
        default=0.0,
    )
    metrics = {
        "slots": len(session.slots),
        "cross_mode": str(session.cross_cloth_mode),
        "max_penetration_depth": max_penetration,
        "max_static_collision_ms": max_static_collision_ms,
        "max_dynamic_triangle_count": max_dynamic_triangles,
        "max_displacement": max_displacement,
        "source_blend": str(blend_path),
    }
    return _finish(
        name=name,
        title=title,
        blend_path=blend_path,
        keywords=keywords,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        step_ms_samples=step_ms_samples,
        simulation_elapsed=simulation_elapsed,
        finite=finite,
        restore_delta=restore_delta,
        metrics=metrics,
    )


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    demo.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[demo.DemoResult] = []
    started = time.perf_counter()
    try:
        recorders = [
            _record_flag,
            _record_wring,
            lambda: _record_timeline("03_clothesline_multicloth_realtime"),
            lambda: _record_timeline("04_tablecloth_pull_collision_realtime"),
            lambda: _record_timeline("05_tshirt_drape_realtime"),
        ]
        for recorder in recorders:
            result = recorder()
            results.append(result)
            print(f"SSBL_DESKTOP_BLEND_DEMO_DONE {result.name} {result.video}")
        contact_sheet = demo._make_contact_sheet(results)
        summary = {
            "input_dir": str(DESKTOP_BLEND_DIR),
            "output_dir": str(demo.OUTPUT_ROOT),
            "contact_sheet": contact_sheet,
            "elapsed_s": time.perf_counter() - started,
            "videos": [result.__dict__ for result in results],
        }
        summary_path = demo.OUTPUT_ROOT / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SSBL_DESKTOP_BLEND_DEMO_PACK", json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if len(results) != 5 or not all(result.validation_passed for result in results):
            raise RuntimeError("Desktop blend demo pack did not generate all five validated videos")
    finally:
        try:
            ssbl.solver.cleanup_all_sessions()
        except Exception:
            pass
        ssbl.unregister()


if __name__ == "__main__":
    main()
