from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


FRAME_COUNT = max(int(os.environ.get("SSBL_CONTACT_SMOKE_FRAMES", "60")), 1)
GRID_SUBDIVISIONS = max(int(os.environ.get("SSBL_CONTACT_SMOKE_GRID", "25")), 4)
SPHERE_SEGMENTS = max(int(os.environ.get("SSBL_CONTACT_SMOKE_SPHERE_SEGMENTS", "12")), 8)
SPHERE_RINGS = max(int(os.environ.get("SSBL_CONTACT_SMOKE_SPHERE_RINGS", "6")), 4)
MAX_EDGE_RATIO_LIMIT = float(os.environ.get("SSBL_CONTACT_SMOKE_MAX_EDGE_RATIO", "20.0"))


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.name.startswith("SSBL_Contact"):
            bpy.data.collections.remove(collection)


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


def _make_cloth() -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=GRID_SUBDIVISIONS,
        y_subdivisions=GRID_SUBDIVISIONS,
        size=1.8,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = "SSBL_Contact_Cloth"
    _configure_settings(obj.ssbl_cloth)
    return obj


def _make_sphere() -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=SPHERE_SEGMENTS,
        ring_count=SPHERE_RINGS,
        radius=0.42,
        location=(0.0, 0.0, 0.36),
    )
    obj = bpy.context.object
    obj.name = "SSBL_Contact_DeformedCloth"
    obj.scale.y = 0.62
    _configure_settings(obj.ssbl_cloth)
    obj.ssbl_cloth.hardness = 0.8
    obj.ssbl_cloth.use_volume_pressure = True
    obj.ssbl_cloth.volume_compliance = 1.0e-6
    obj.ssbl_cloth.pressure_strength = 0.25
    return obj


def _finite_positions(positions: np.ndarray) -> bool:
    return bool(np.isfinite(positions).all())


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


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        _clear_scene()
        scene = bpy.context.scene
        cloth = _make_cloth()
        sphere = _make_sphere()
        before_cloth = _snapshot(cloth)
        before_sphere = _snapshot(sphere)
        bpy.ops.object.select_all(action="DESELECT")
        cloth.select_set(True)
        sphere.select_set(True)
        bpy.context.view_layer.objects.active = cloth
        scene.frame_current = 1
        session = ssbl.solver.start_preview(bpy.context, cloth)
        max_edge_ratio = 0.0
        max_dynamic_triangles = 0
        max_resolved_contacts = 0
        max_contact_cache_hits = 0
        max_contact_cache_count = 0
        max_contact_cache_overflow = 0
        max_friction_corrections = 0
        finite = True
        for _frame in range(FRAME_COUNT):
            ssbl.solver.step_preview(bpy.context, cloth.name)
            diagnostics = ssbl.solver.session_diagnostics(cloth)
            max_dynamic_triangles = max(max_dynamic_triangles, int(diagnostics.dynamic_triangle_count))
            max_resolved_contacts = max(max_resolved_contacts, int(diagnostics.resolved_contacts))
            max_contact_cache_hits = max(max_contact_cache_hits, int(diagnostics.external_contact_cache_hits))
            max_contact_cache_count = max(max_contact_cache_count, int(diagnostics.external_contact_cache_count))
            max_contact_cache_overflow = max(max_contact_cache_overflow, int(diagnostics.external_contact_cache_overflow))
            max_friction_corrections = max(max_friction_corrections, int(diagnostics.external_friction_corrections))
            finite = finite and bool(diagnostics.finite)
            for slot in session.slots.values():
                finite = finite and _finite_positions(slot.current_positions_world)
                max_edge_ratio = max(max_edge_ratio, _slot_max_edge_ratio(slot))
            if not finite:
                break
        diagnostics = ssbl.solver.session_diagnostics(cloth)
        stopped = ssbl.solver.request_stop(cloth)
        result = {
            "slots": len(session.slots),
            "cross_mode": str(session.cross_cloth_mode),
            "frames": FRAME_COUNT,
            "finite": bool(finite),
            "max_edge_ratio": float(max_edge_ratio),
            "max_dynamic_triangle_count": int(max_dynamic_triangles),
            "max_resolved_contacts": int(max_resolved_contacts),
            "max_contact_cache_hits": int(max_contact_cache_hits),
            "max_contact_cache_count": int(max_contact_cache_count),
            "max_contact_cache_overflow": int(max_contact_cache_overflow),
            "max_friction_corrections": int(max_friction_corrections),
            "dynamic_collision_ms": float(diagnostics.dynamic_collision_ms),
            "dynamic_upload_ms": float(diagnostics.dynamic_upload_ms),
            "restored_cloth_mesh": bool(cloth.data == session.slots[cloth.name].original_mesh) if cloth.name in session.slots else True,
            "stopped": bool(stopped),
            "cloth_restore_delta": _max_source_delta(cloth, before_cloth),
            "sphere_restore_delta": _max_source_delta(sphere, before_sphere),
        }
        print("SSBL_MULTICLOTH_CONTACT_STABILITY_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            result["slots"] == 2
            and result["cross_mode"] == "all_selected"
            and result["finite"]
            and result["max_dynamic_triangle_count"] > 0
            and result["max_resolved_contacts"] > 0
            and result["max_contact_cache_hits"] > 0
            and result["max_contact_cache_count"] > 0
            and result["max_contact_cache_overflow"] == 0
            and result["max_edge_ratio"] < MAX_EDGE_RATIO_LIMIT
            and result["stopped"]
            and result["cloth_restore_delta"] == 0.0
            and result["sphere_restore_delta"] == 0.0
        ):
            raise RuntimeError(f"Multi-cloth contact stability smoke failed: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
