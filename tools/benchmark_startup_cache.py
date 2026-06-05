from __future__ import annotations

import json
import math
import os
import sys
import time

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from ssbl import collision, xpbd_core


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


def _finite_object(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _make_default_cloth() -> bpy.types.Object:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=65, y_subdivisions=65, size=2.0, location=(0.0, 0.0, 1.0))
    obj = bpy.context.object
    obj.name = "SSBL_Startup_Cache_Cloth"
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([vertex.index for vertex in obj.data.vertices if vertex.co.y > 0.85], 1.0, "ADD")
    settings = bpy.context.scene.ssbl_preview
    settings.pin_vertex_group = "ssbl_pin"
    settings.use_evaluated_mesh = True
    settings.use_volume_pressure = False
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.frame_count = 8
    return obj


def _choose_object() -> bpy.types.Object:
    requested = os.environ.get("SSBL_STARTUP_OBJECT", "").strip()
    if requested:
        obj = bpy.data.objects.get(requested)
        if obj is None or obj.type != "MESH":
            raise RuntimeError(f"SSBL_STARTUP_OBJECT is not a mesh object: {requested}")
        return obj
    active = bpy.context.view_layer.objects.active
    if active is not None and active.type == "MESH":
        return active
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if meshes:
        return max(meshes, key=lambda item: len(item.data.vertices))
    return _make_default_cloth()


def _activate_object(obj: bpy.types.Object) -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _run_start_stop(obj: bpy.types.Object) -> dict[str, object]:
    before = _snapshot(obj)
    topology_before = xpbd_core.cloth_topology_cache_stats()
    static_before = collision.static_collision_cache_stats()
    started = time.perf_counter()
    session = None
    result: dict[str, object] = {}
    try:
        session = ssbl.solver.start_preview(bpy.context, obj)
        start_ms = (time.perf_counter() - started) * 1000.0
        finite = _finite_object(obj)
        topology_after = xpbd_core.cloth_topology_cache_stats()
        static_after = collision.static_collision_cache_stats()
        result = {
            "start_ms": float(start_ms),
            "topology_cache_hit": int(topology_after["hits"]) > int(topology_before["hits"]),
            "static_cache_hit": int(static_after["hits"]) > int(static_before["hits"]),
            "finite": bool(finite),
            "verts": int(len(session.cloth.positions_world)),
            "tris": int(len(session.cloth.triangles)),
            "topology_cache": topology_after,
            "static_cache": static_after,
        }
    finally:
        if session is not None:
            ssbl.solver.request_stop(obj)
        else:
            ssbl.solver.cleanup_all_sessions()
        _activate_object(obj)
        result["restore_delta"] = float(_max_source_delta(obj, before))
    return result


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        obj = _choose_object()
        _activate_object(obj)
        ssbl.solver.clear_startup_build_caches()
        first = _run_start_stop(obj)
        second = _run_start_stop(obj)
        result = {
            "object": obj.name,
            "first_start_ms": round(float(first["start_ms"]), 3),
            "second_start_ms": round(float(second["start_ms"]), 3),
            "speedup": round(float(first["start_ms"]) / max(float(second["start_ms"]), 1.0e-6), 3),
            "topology_cache_hit": bool(second["topology_cache_hit"]),
            "static_cache_hit": bool(second["static_cache_hit"]),
            "first_topology_cache_hit": bool(first["topology_cache_hit"]),
            "first_static_cache_hit": bool(first["static_cache_hit"]),
            "finite": bool(first["finite"]) and bool(second["finite"]),
            "restore_delta": max(float(first["restore_delta"]), float(second["restore_delta"])),
            "verts": int(second["verts"]),
            "tris": int(second["tris"]),
            "topology_cache": second["topology_cache"],
            "static_cache": second["static_cache"],
        }
        print("SSBL_STARTUP_CACHE_BENCHMARK", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not result["finite"] or result["restore_delta"] != 0.0 or not result["topology_cache_hit"]:
            raise RuntimeError(f"Startup cache benchmark failed: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
