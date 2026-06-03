from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


DEFAULT_BLEND_PATH = r"C:\Users\Administrator\Desktop\cs2.blend"
BLEND_PATH = os.environ.get("SSBL_BLOCK_SMOKE_BLEND", DEFAULT_BLEND_PATH)
FRAME_COUNT = max(int(os.environ.get("SSBL_BLOCK_SMOKE_FRAMES", "60")), 1)
OUTPUT_JSON = os.environ.get("SSBL_BLOCK_SMOKE_JSON", "")
MIN_CENTER_ABOVE_SUPPORT = float(os.environ.get("SSBL_BLOCK_SMOKE_MIN_CENTER_ABOVE_SUPPORT", "-0.15"))
MAX_FULL_OVERLAP_FRAMES = max(int(os.environ.get("SSBL_BLOCK_SMOKE_MAX_FULL_OVERLAP_FRAMES", "0")), 0)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.name.startswith("SSBL_BlockSmoke"):
            bpy.data.collections.remove(collection)


def _configure_common(settings) -> None:
    settings.enabled = True
    settings.pin_vertex_group = ""
    settings.use_evaluated_mesh = True
    settings.preview_writeback_interval = 0
    settings.preview_target_fps = 30.0
    settings.frame_count = FRAME_COUNT + 4
    settings.dt = 1.0 / 50.0
    settings.substeps = 14
    settings.iterations = 2
    settings.damping = 1.0
    settings.gravity = (0.0, 0.0, -9.8)
    settings.collision_margin = 0.005
    settings.cloth_thickness = 0.02
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.sphere_object = None
    settings.static_collider_collection = None


def _make_generated_scene() -> tuple[bpy.types.Object, bpy.types.Object]:
    _clear_scene()
    if not hasattr(bpy.ops.mesh, "primitive_monkey_add"):
        raise RuntimeError("This Blender build does not expose primitive_monkey_add")

    bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0.0, 0.0, 0.0))
    monkey = bpy.context.object
    monkey.name = "Suzanne"
    monkey.data.name = "SSBL_BlockSmoke_SuzanneMesh"
    _configure_common(monkey.ssbl_cloth)
    monkey.ssbl_cloth.self_collision = True
    monkey.ssbl_cloth.self_collision_mode = "fast"

    z_values = [vert.co.z for vert in monkey.data.vertices]
    z_min = min(z_values)
    z_max = max(z_values)
    pin_threshold = z_max - (z_max - z_min) * 0.18
    pin_indices = [vert.index for vert in monkey.data.vertices if vert.co.z >= pin_threshold]
    if not pin_indices:
        pin_indices = [max(monkey.data.vertices, key=lambda vert: vert.co.z).index]
    pin = monkey.vertex_groups.new(name="ssbl_pin")
    pin.add(pin_indices, 1.0, "ADD")
    monkey.ssbl_cloth.pin_vertex_group = "ssbl_pin"

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.75, location=(0.0, 0.0, 2.2))
    sphere = bpy.context.object
    sphere.name = "Cube"
    sphere.data.name = "SSBL_BlockSmoke_SphereMesh"
    _configure_common(sphere.ssbl_cloth)
    sphere.ssbl_cloth.use_volume_pressure = True
    sphere.ssbl_cloth.volume_compliance = 1.0e-6
    sphere.ssbl_cloth.pressure_strength = 0.25
    sphere.ssbl_cloth.self_collision = False
    sphere.ssbl_cloth.self_collision_mode = "off"
    return sphere, monkey


def _load_or_create_scene() -> tuple[bpy.types.Object, bpy.types.Object]:
    if BLEND_PATH and Path(BLEND_PATH).exists():
        bpy.ops.wm.open_mainfile(filepath=BLEND_PATH, load_ui=False)
        sphere = bpy.data.objects.get("Cube")
        monkey = bpy.data.objects.get("Suzanne")
        if sphere is None or monkey is None:
            raise RuntimeError(f"Expected Cube and Suzanne in {BLEND_PATH}")
        if sphere.type != "MESH" or monkey.type != "MESH":
            raise RuntimeError("Cube and Suzanne must both be mesh objects")
        return sphere, monkey
    return _make_generated_scene()


def _snapshot(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in obj.data.vertices]


def _max_source_delta(obj: bpy.types.Object, before: list[tuple[float, float, float]]) -> float:
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


def _positions_aabb(positions: np.ndarray) -> dict[str, float]:
    return {
        "min_x": float(np.min(positions[:, 0])),
        "max_x": float(np.max(positions[:, 0])),
        "min_y": float(np.min(positions[:, 1])),
        "max_y": float(np.max(positions[:, 1])),
        "min_z": float(np.min(positions[:, 2])),
        "max_z": float(np.max(positions[:, 2])),
        "center_z": float((np.min(positions[:, 2]) + np.max(positions[:, 2])) * 0.5),
    }


def _all_finite(positions: np.ndarray) -> bool:
    return bool(np.isfinite(positions).all())


def _prepare_objects(sphere: bpy.types.Object, monkey: bpy.types.Object) -> None:
    scene = bpy.context.scene
    for obj in (sphere, monkey):
        obj.hide_viewport = False
        obj.hide_render = False
        obj.ssbl_cloth.enabled = True
        obj.ssbl_cloth.static_collider_collection = None
        obj.ssbl_cloth.use_sphere = False
        obj.ssbl_cloth.sphere_object = None
    sphere.select_set(True)
    monkey.select_set(True)
    bpy.context.view_layer.objects.active = sphere
    scene.frame_set(1)


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        sphere, monkey = _load_or_create_scene()
        _prepare_objects(sphere, monkey)
        sphere_before = _snapshot(sphere)
        monkey_before = _snapshot(monkey)

        session = ssbl.solver.start_preview(bpy.context, sphere)
        initial_monkey_positions = np.asarray(session.slots[monkey.name].current_positions_world, dtype=np.float64)
        initial_monkey_top = float(np.max(initial_monkey_positions[:, 2]))
        min_allowed_center_z = initial_monkey_top + MIN_CENTER_ABOVE_SUPPORT

        rows: list[dict[str, object]] = []
        finite = True
        max_dynamic_triangles = 0
        max_resolved_contacts = 0
        full_overlap_frames = 0
        min_center_above_support = float("inf")
        for frame in range(1, FRAME_COUNT + 1):
            ssbl.solver.step_preview(bpy.context, sphere.name)
            diag = ssbl.solver.session_diagnostics(sphere)
            sphere_positions = np.asarray(session.slots[sphere.name].current_positions_world, dtype=np.float64)
            monkey_positions = np.asarray(session.slots[monkey.name].current_positions_world, dtype=np.float64)
            sphere_box = _positions_aabb(sphere_positions)
            monkey_box = _positions_aabb(monkey_positions)
            finite = finite and _all_finite(sphere_positions) and _all_finite(monkey_positions) and bool(diag.finite)
            max_dynamic_triangles = max(max_dynamic_triangles, int(diag.dynamic_triangle_count))
            max_resolved_contacts = max(max_resolved_contacts, int(diag.resolved_contacts))

            center_above_support = float(sphere_box["center_z"] - initial_monkey_top)
            min_center_above_support = min(min_center_above_support, center_above_support)
            sphere_fully_below_initial_top = bool(float(sphere_box["max_z"]) < initial_monkey_top)
            if sphere_fully_below_initial_top:
                full_overlap_frames += 1

            if frame <= 16 or frame % 5 == 0 or sphere_fully_below_initial_top:
                rows.append(
                    {
                        "frame": frame,
                        "sphere_min_z": sphere_box["min_z"],
                        "sphere_max_z": sphere_box["max_z"],
                        "sphere_center_z": sphere_box["center_z"],
                        "monkey_min_z": monkey_box["min_z"],
                        "monkey_max_z": monkey_box["max_z"],
                        "center_above_initial_monkey_top": center_above_support,
                        "sphere_fully_below_initial_monkey_top": sphere_fully_below_initial_top,
                        "dynamic_triangle_count": int(diag.dynamic_triangle_count),
                        "resolved_contacts": int(diag.resolved_contacts),
                        "penetration_depth_diag": float(diag.penetration_depth),
                    }
                )

        stopped = ssbl.solver.request_stop(sphere)
        result = {
            "blend_file": bpy.data.filepath,
            "frames": FRAME_COUNT,
            "slots": len(session.slots),
            "slot_names": list(session.slots.keys()),
            "cross_mode": str(session.cross_cloth_mode),
            "initial_monkey_top": initial_monkey_top,
            "min_allowed_center_z": min_allowed_center_z,
            "min_center_above_support": min_center_above_support,
            "full_overlap_frames": full_overlap_frames,
            "max_dynamic_triangle_count": max_dynamic_triangles,
            "max_resolved_contacts": max_resolved_contacts,
            "finite": bool(finite),
            "stopped": bool(stopped),
            "sphere_restore_delta": _max_source_delta(sphere, sphere_before),
            "monkey_restore_delta": _max_source_delta(monkey, monkey_before),
            "rows": rows,
        }
        if OUTPUT_JSON:
            Path(OUTPUT_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            "SSBL_CLOTH_MONKEY_BLOCKS_SPHERE_SMOKE "
            + json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, sort_keys=True)
        )
        if not (
            result["slots"] == 2
            and result["cross_mode"] == "all_selected"
            and result["finite"]
            and result["max_dynamic_triangle_count"] > 0
            and result["max_resolved_contacts"] > 0
            and result["full_overlap_frames"] <= MAX_FULL_OVERLAP_FRAMES
            and min_center_above_support >= MIN_CENTER_ABOVE_SUPPORT
            and result["stopped"]
            and result["sphere_restore_delta"] == 0.0
            and result["monkey_restore_delta"] == 0.0
        ):
            raise RuntimeError(f"Cloth monkey did not block falling sphere: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
