from __future__ import annotations

import json
import math
import sys
import time

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _clear_scene() -> None:
    for existing in list(bpy.context.scene.objects):
        bpy.data.objects.remove(existing, do_unlink=True)


def _grid(name: str, subdivisions: int, size: float, location: tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdivisions, y_subdivisions=subdivisions, size=size, location=location)
    obj = bpy.context.object
    obj.name = name
    group = obj.vertex_groups.new(name="ssbl_pin")
    top = [v.index for v in obj.data.vertices if v.co.y > size * 0.42]
    group.add(top, 1.0, "ADD")
    return obj


def _configure_common() -> None:
    settings = bpy.context.scene.ssbl_preview
    settings.hardness = 0.6
    settings.hardness_initialized = True
    settings.pin_vertex_group = "ssbl_pin"
    settings.use_ground = False
    settings.use_sphere = False
    settings.use_wall = False
    settings.static_collider_collection = None
    settings.collision_margin = 0.035
    settings.self_collision_mode = "off"
    settings.substeps = 4
    settings.iterations = 1
    settings.frame_count = 80
    settings.multi_cloth_preview = False
    settings.cross_cloth_collision = "lower_layers"


def _finite_object(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _max_source_delta(obj: bpy.types.Object, original: list[tuple[float, float, float]]) -> float:
    if len(original) != len(obj.data.vertices):
        return float("inf")
    max_delta = 0.0
    for vertex, before in zip(obj.data.vertices, original):
        max_delta = max(
            max_delta,
            abs(float(vertex.co.x) - before[0]),
            abs(float(vertex.co.y) - before[1]),
            abs(float(vertex.co.z) - before[2]),
        )
    return max_delta


def _snapshot(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(v.co.x), float(v.co.y), float(v.co.z)) for v in obj.data.vertices]


def _step_session(obj: bpy.types.Object, steps: int) -> tuple[float, object]:
    session = ssbl.solver.start_preview(bpy.context, obj)
    started = time.perf_counter()
    completed = 0
    for _ in range(steps):
        finished = ssbl.solver.step_preview(bpy.context, obj.name)
        completed += 1
        if finished:
            break
    elapsed = max(time.perf_counter() - started, 1.0e-6)
    return completed / elapsed, session


def _case_10k() -> dict[str, object]:
    _clear_scene()
    _configure_common()
    cloth = _grid("SSBL_v2_10k", 101, 2.4, (0.0, 0.0, 1.4))
    settings = bpy.context.scene.ssbl_preview
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 2
    before = _snapshot(cloth)
    fps, session = _step_session(cloth, 6)
    finite = _finite_object(cloth)
    ssbl.solver.request_stop(cloth)
    return {
        "case": "10k_fast",
        "fps": round(fps, 2),
        "verts": len(session.cloth.positions_world),
        "tris": len(session.cloth.triangles),
        "finite": finite,
        "restore_delta": _max_source_delta(cloth, before),
    }


def _case_multicloth() -> dict[str, object]:
    _clear_scene()
    _configure_common()
    lower = _grid("SSBL_v2_lower", 41, 2.0, (0.0, 0.0, 1.0))
    upper = _grid("SSBL_v2_upper", 41, 2.0, (0.0, 0.0, 1.02))
    lower.ssbl_collision_layer = 0
    upper.ssbl_collision_layer = 1
    lower.ssbl_enable_cross_cloth_collision = True
    upper.ssbl_enable_cross_cloth_collision = True
    settings = bpy.context.scene.ssbl_preview
    settings.multi_cloth_preview = True
    settings.cross_cloth_collision = "lower_layers"
    settings.collision_margin = 0.04
    before_lower = _snapshot(lower)
    before_upper = _snapshot(upper)
    bpy.ops.object.select_all(action="DESELECT")
    lower.select_set(True)
    upper.select_set(True)
    bpy.context.view_layer.objects.active = upper
    fps, session = _step_session(upper, 10)
    finite = _finite_object(lower) and _finite_object(upper)
    lower_world = [(lower.matrix_world @ vertex.co).z for vertex in lower.data.vertices]
    upper_world = [(upper.matrix_world @ vertex.co).z for vertex in upper.data.vertices]
    min_pair_gap = min(u - l for u, l in zip(upper_world, lower_world))
    ssbl.solver.request_stop(upper)
    return {
        "case": "two_cloth_lower_layers",
        "fps": round(fps, 2),
        "slots": len(session.slots),
        "finite": finite,
        "min_pair_gap": round(float(min_pair_gap), 5),
        "restore_delta_lower": _max_source_delta(lower, before_lower),
        "restore_delta_upper": _max_source_delta(upper, before_upper),
    }


def _case_static_collection() -> dict[str, object]:
    _clear_scene()
    _configure_common()
    cloth = _grid("SSBL_v2_static_cloth", 33, 1.8, (0.0, 0.0, 1.2))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.45, location=(0.0, 0.0, 0.82))
    collider = bpy.context.object
    collider.name = "SSBL_v2_static_collider"
    collection = bpy.data.collections.new("SSBL_v2_static_colliders")
    bpy.context.scene.collection.children.link(collection)
    for parent_collection in list(collider.users_collection):
        parent_collection.objects.unlink(collider)
    collection.objects.link(collider)
    settings = bpy.context.scene.ssbl_preview
    settings.static_collider_collection = collection
    settings.collision_margin = 0.025
    before = _snapshot(cloth)
    fps, session = _step_session(cloth, 8)
    finite = _finite_object(cloth)
    ssbl.solver.request_stop(cloth)
    return {
        "case": "static_collection",
        "fps": round(fps, 2),
        "static_tris": int(session.slots[session.object_name].static_collider_signature[0][2]),
        "finite": finite,
        "restore_delta": _max_source_delta(cloth, before),
    }


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        results = [_case_10k(), _case_multicloth(), _case_static_collection()]
        print("SSBL_V2_BENCHMARK", json.dumps(results, ensure_ascii=False, sort_keys=True))
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
