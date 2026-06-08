from __future__ import annotations

import json
import math
import statistics
import sys
import time

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


STEPS = 18
WARMUP = 3


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.name.startswith("SSBL_SpherePerf"):
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


def _configure_common(settings) -> None:
    settings.pin_vertex_group = ""
    settings.use_evaluated_mesh = True
    settings.preview_writeback_interval = 1
    settings.preview_target_fps = 60.0
    settings.dt = 1.0 / 60.0
    settings.substeps = 4
    settings.iterations = 1
    settings.frame_count = STEPS + 4
    settings.damping = 1.0
    settings.gravity = (0.0, 0.0, 0.0)
    settings.hardness = 0.5
    settings.self_collision = False
    settings.use_volume_pressure = False
    settings.collision_margin = 0.015
    settings.cloth_thickness = 0.04
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.sphere_object = None
    settings.static_collider_collection = None


def _make_cloth() -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=33, y_subdivisions=33, size=1.6, location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = "SSBL_SpherePerf_Cloth"
    return obj


def _make_sphere(*, cloth_enabled: bool, location=(0.0, 0.0, 0.34)) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.38, location=location)
    obj = bpy.context.object
    obj.name = "SSBL_SpherePerf_Sphere"
    _configure_common(obj.ssbl_cloth)
    obj.ssbl_cloth.enabled = bool(cloth_enabled)
    obj.ssbl_cloth.hardness = 0.8
    obj.ssbl_cloth.use_volume_pressure = True
    obj.ssbl_cloth.volume_compliance = 1.0e-6
    obj.ssbl_cloth.pressure_strength = 0.2
    return obj


def _make_static_collection(obj: bpy.types.Object) -> bpy.types.Collection:
    collection = bpy.data.collections.new("SSBL_SpherePerf_StaticCollection")
    bpy.context.scene.collection.children.link(collection)
    if obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(obj)
    collection.objects.link(obj)
    return collection


def _finite_object(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _run_case(label: str, mode: str) -> dict[str, object]:
    _clear_scene()
    scene = bpy.context.scene
    settings = scene.ssbl_preview
    _configure_common(settings)
    cloth = _make_cloth()
    sphere = None
    if mode == "analytic":
        sphere = _make_sphere(cloth_enabled=True)
        settings.use_sphere = True
        settings.sphere_object = sphere
    elif mode == "static_mesh":
        sphere = _make_sphere(cloth_enabled=True)
        settings.static_collider_collection = _make_static_collection(sphere)
    elif mode == "cloth_enabled":
        sphere = _make_sphere(cloth_enabled=True)
    elif mode != "cloth_only":
        raise ValueError(f"unknown case mode: {mode}")

    before_cloth = _snapshot(cloth)
    before_sphere = _snapshot(sphere) if sphere is not None else []
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    if sphere is not None:
        sphere.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    scene.frame_current = 1
    session = ssbl.solver.start_preview(bpy.context, cloth)
    samples: list[float] = []
    for index in range(STEPS):
        started = time.perf_counter()
        ssbl.solver.step_preview(bpy.context, cloth.name)
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= WARMUP:
            samples.append(elapsed)
    diagnostics = ssbl.solver.session_diagnostics(cloth)
    finite = _finite_object(cloth) and bool(diagnostics.finite)
    if sphere is not None and sphere.name in session.slots:
        finite = finite and _finite_object(sphere)
    stopped = ssbl.solver.request_stop(cloth)
    sphere_delta = _max_source_delta(sphere, before_sphere) if sphere is not None else 0.0
    avg_frame_ms = statistics.fmean(samples) if samples else 0.0
    return {
        "case": label,
        "mode": mode,
        "slots": len(session.slots),
        "cross_mode": str(session.cross_cloth_mode),
        "avg_frame_ms": round(float(avg_frame_ms), 3),
        "finite": bool(finite),
        "stopped": bool(stopped),
        "cloth_restore_delta": _max_source_delta(cloth, before_cloth),
        "sphere_restore_delta": sphere_delta,
        "analytic_collision_ms": round(float(diagnostics.analytic_collision_ms), 3),
        "static_collision_ms": round(float(diagnostics.static_collision_ms), 3),
        "dynamic_collision_ms": round(float(diagnostics.dynamic_collision_ms), 3),
        "dynamic_upload_ms": round(float(diagnostics.dynamic_upload_ms), 3),
        "download_ms": round(float(diagnostics.download_ms), 3),
        "writeback_ms": round(float(diagnostics.writeback_ms), 3),
        "writeback_performed": bool(diagnostics.writeback_performed),
        "hash_build_ms": round(float(diagnostics.hash_build_ms), 3),
        "dynamic_triangle_count": int(diagnostics.dynamic_triangle_count),
        "static_triangle_count": int(diagnostics.static_triangle_count),
        "static_sdf_rebuild_count": int(diagnostics.static_sdf_rebuild_count),
        "static_sdf_voxel_count": int(diagnostics.static_sdf_voxel_count),
        "static_sdf_grid": [
            int(diagnostics.static_sdf_grid_x),
            int(diagnostics.static_sdf_grid_y),
            int(diagnostics.static_sdf_grid_z),
        ],
        "static_sdf_contact_count": int(diagnostics.static_sdf_contact_count),
        "static_sdf_unsigned_fallback_count": int(diagnostics.static_sdf_unsigned_fallback_count),
        "candidate_count": int(diagnostics.candidate_count),
        "resolved_contacts": int(diagnostics.resolved_contacts),
    }


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        cases = [
            _run_case("cloth_only", "cloth_only"),
            _run_case("analytic_sphere_collider", "analytic"),
            _run_case("mesh_sphere_collision_only", "static_mesh"),
            _run_case("mesh_sphere_cloth_enabled_auto_collider", "cloth_enabled"),
        ]
        by_case = {case["case"]: case for case in cases}
        cloth_only = by_case["cloth_only"]
        analytic = by_case["analytic_sphere_collider"]
        static_mesh = by_case["mesh_sphere_collision_only"]
        cloth_enabled = by_case["mesh_sphere_cloth_enabled_auto_collider"]
        analytic_ratio = float(analytic["avg_frame_ms"]) / max(float(cloth_only["avg_frame_ms"]), 1.0e-6)
        result = {
            "cases": cases,
            "analytic_overhead_ratio": round(analytic_ratio, 3),
        }
        print("SSBL_SPHERE_COLLISION_PERF_AB", json.dumps(result, ensure_ascii=False, sort_keys=True))
        analytic_threshold = max(float(cloth_only["avg_frame_ms"]) * 1.75, float(cloth_only["avg_frame_ms"]) + 8.0)
        if not (
            all(case["finite"] and case["stopped"] for case in cases)
            and all(case["cloth_restore_delta"] == 0.0 for case in cases)
            and all(case["sphere_restore_delta"] == 0.0 for case in cases)
            and analytic["slots"] == 1
            and analytic["analytic_collision_ms"] >= 0.0
            and analytic["dynamic_triangle_count"] == 0
            and analytic["static_sdf_rebuild_count"] == 0
            and float(analytic["avg_frame_ms"]) < analytic_threshold
            and static_mesh["slots"] == 1
            and static_mesh["static_triangle_count"] > 0
            and static_mesh["dynamic_triangle_count"] == 0
            and static_mesh["static_sdf_rebuild_count"] > 0
            and static_mesh["static_sdf_voxel_count"] > 0
            and all(value > 1 for value in static_mesh["static_sdf_grid"])
            and cloth_enabled["slots"] == 1
            and cloth_enabled["cross_mode"] == "off"
            and cloth_enabled["dynamic_triangle_count"] == 0
            and cloth_enabled["analytic_collision_ms"] >= 0.0
            and cloth_enabled["static_sdf_rebuild_count"] == 0
        ):
            raise RuntimeError(f"Sphere collision perf A/B failed: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
