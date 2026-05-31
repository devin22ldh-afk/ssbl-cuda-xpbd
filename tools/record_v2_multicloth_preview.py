from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _clear_scene() -> None:
    for existing in list(bpy.context.scene.objects):
        bpy.data.objects.remove(existing, do_unlink=True)


def _grid(name: str, z: float, layer: int, size: float, color: tuple[float, float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=45, y_subdivisions=45, size=size, location=(0.0, 0.0, z))
    obj = bpy.context.object
    obj.name = name
    obj.ssbl_collision_layer = layer
    obj.ssbl_enable_cross_cloth_collision = True
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([v.index for v in obj.data.vertices if v.co.y > size * 0.41], 1.0, "ADD")
    material = bpy.data.materials.new(f"{name}_Mat")
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
    obj.data.materials.append(material)
    return obj


def _setup_render(scene: bpy.types.Scene) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 12
    if scene.world is not None:
        scene.world.color = (0.03, 0.035, 0.04)
    camera_data = bpy.data.cameras.new("SSBL_V2_Record_Camera")
    camera = bpy.data.objects.new("SSBL_V2_Record_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.0, -4.2, 2.15)
    camera.rotation_euler = (math.radians(66.0), 0.0, 0.0)
    camera_data.lens = 38
    scene.camera = camera
    light_data = bpy.data.lights.new("SSBL_V2_Record_Key", "AREA")
    light = bpy.data.objects.new("SSBL_V2_Record_Key", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, -2.0, 4.0)
    light_data.energy = 450
    light_data.size = 4


def _render(scene: bpy.types.Scene, frames_dir: Path, index: int, label: str) -> str:
    path = frames_dir / f"{index:04d}_{label}.png"
    scene.render.filepath = str(path)
    try:
        bpy.ops.render.opengl(write_still=True, view_context=False)
    except RuntimeError:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        bpy.ops.render.render(write_still=True)
    return str(path)


def _finite(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _snapshot(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(v.co.x), float(v.co.y), float(v.co.z)) for v in obj.data.vertices]


def _restore_delta(obj: bpy.types.Object, before: list[tuple[float, float, float]]) -> float:
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


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    _clear_scene()

    output_override = os.environ.get("SSBL_RECORD_OUTPUT_DIR")
    output_dir = Path(output_override) if output_override else Path(tempfile.gettempdir()) / "ssbl_v2_multicloth_recording"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("*.png"):
        path.unlink()

    lower = _grid("SSBL_V2_Lower_Layer", 1.0, 0, 2.1, (0.10, 0.32, 0.95, 1.0))
    upper = _grid("SSBL_V2_Upper_Layer", 1.02, 1, 1.8, (0.95, 0.28, 0.12, 1.0))
    before_lower = _snapshot(lower)
    before_upper = _snapshot(upper)

    scene = bpy.context.scene
    scene.frame_set(1)
    _setup_render(scene)
    settings = scene.ssbl_preview
    settings.pin_vertex_group = "ssbl_pin"
    settings.use_ground = False
    settings.multi_cloth_preview = True
    settings.cross_cloth_collision = "lower_layers"
    settings.self_collision_mode = "off"
    settings.collision_margin = 0.04
    settings.substeps = 4
    settings.iterations = 1
    settings.frame_count = 36

    bpy.ops.object.select_all(action="DESELECT")
    lower.select_set(True)
    upper.select_set(True)
    bpy.context.view_layer.objects.active = upper

    frame_paths = [_render(scene, frames_dir, 0, "before")]
    session = ssbl.solver.start_preview(bpy.context, upper)
    frame_paths.append(_render(scene, frames_dir, 1, "start"))
    simulation_elapsed = 0.0
    simulation_steps = 0
    for index in range(1, 37):
        step_started = time.perf_counter()
        finished = ssbl.solver.step_preview(bpy.context, upper.name)
        simulation_elapsed += max(time.perf_counter() - step_started, 0.0)
        simulation_steps += 1
        if index <= 8 or index % 4 == 0 or index == 36:
            frame_paths.append(_render(scene, frames_dir, index + 1, f"step{index:02d}"))
        if finished:
            break

    lower_world = [(lower.matrix_world @ vertex.co).z for vertex in lower.data.vertices]
    upper_world = [(upper.matrix_world @ vertex.co).z for vertex in upper.data.vertices]
    min_pair_gap = min(u - l for u, l in zip(upper_world, lower_world))
    finite = _finite(lower) and _finite(upper)
    ssbl.solver.request_stop(upper)
    frame_paths.append(_render(scene, frames_dir, 1000, "stopped"))

    summary = {
        "output_dir": str(output_dir),
        "frames_dir": str(frames_dir),
        "frame_paths": frame_paths,
        "slots": len(session.slots),
        "simulation_fps": simulation_steps / max(simulation_elapsed, 1.0e-6),
        "recording_fps_with_render": session.actual_fps,
        "finite": finite,
        "min_pair_gap": float(min_pair_gap),
        "restore_delta_lower": _restore_delta(lower, before_lower),
        "restore_delta_upper": _restore_delta(upper, before_upper),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SSBL_V2_MULTICLOTH_RECORDING", str(summary_path))
    print(json.dumps(summary, ensure_ascii=False))
    ssbl.unregister()


if __name__ == "__main__":
    main()
