from __future__ import annotations

import json
import os
import sys

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


FRAME_COUNT = max(int(os.environ.get("SSBL_PARTICLE_CONTACT_FRAMES", "48")), 1)
GRID_SUBDIVISIONS = max(int(os.environ.get("SSBL_PARTICLE_CONTACT_GRID", "24")), 4)
EXPECT_LARGE_GRID = (
    os.environ.get("SSBL_PARTICLE_CONTACT_EXPECT_LARGE", "").strip().lower() in {"1", "true", "yes", "on"}
    or GRID_SUBDIVISIONS >= 96
)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _configure_settings(settings) -> None:
    settings.enabled = True
    settings.pin_vertex_group = ""
    settings.use_evaluated_mesh = True
    settings.preview_writeback_interval = 0
    settings.preview_target_fps = 60.0
    settings.dt = 1.0 / 60.0
    settings.substeps = 6
    settings.iterations = 2
    settings.frame_count = FRAME_COUNT + 4
    settings.damping = 0.98
    settings.gravity = (0.0, 0.0, 0.0)
    settings.hardness = 0.55
    settings.self_collision = False
    settings.use_volume_pressure = False
    settings.collision_margin = 0.008
    settings.cloth_thickness = 0.05
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.sphere_object = None
    settings.static_collider_collection = None


def _make_cloth_pair() -> tuple[bpy.types.Object, bpy.types.Object]:
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=GRID_SUBDIVISIONS,
        y_subdivisions=GRID_SUBDIVISIONS,
        size=1.8,
        location=(0.0, 0.0, 0.0),
    )
    cloth = bpy.context.object
    cloth.name = "SSBL_ParticleContact_Cloth"
    _configure_settings(cloth.ssbl_cloth)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.42, location=(0.0, 0.0, 0.34))
    sphere = bpy.context.object
    sphere.name = "SSBL_ParticleContact_SphereCloth"
    sphere.scale.y = 0.62
    _configure_settings(sphere.ssbl_cloth)
    sphere.ssbl_cloth.hardness = 0.8
    sphere.ssbl_cloth.use_volume_pressure = True
    sphere.ssbl_cloth.volume_compliance = 1.0e-6
    sphere.ssbl_cloth.pressure_strength = 0.25
    return cloth, sphere


def _finite_positions(positions: np.ndarray) -> bool:
    return bool(np.isfinite(positions).all())


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        _clear_scene()
        cloth, sphere = _make_cloth_pair()
        bpy.ops.object.select_all(action="DESELECT")
        cloth.select_set(True)
        sphere.select_set(True)
        bpy.context.view_layer.objects.active = cloth
        session = ssbl.solver.start_preview(bpy.context, cloth)
        finite = True
        max_dynamic_particles = 0
        max_particle_candidates = 0
        max_particle_contacts = 0
        max_particle_overflow = 0
        max_dynamic_triangles = 0
        max_resolved_contacts = 0
        for _frame in range(FRAME_COUNT):
            ssbl.solver.step_preview(bpy.context, cloth.name)
            diagnostics = ssbl.solver.session_diagnostics(cloth)
            finite = finite and bool(diagnostics.finite)
            max_dynamic_particles = max(max_dynamic_particles, int(diagnostics.dynamic_particle_count))
            max_particle_candidates = max(max_particle_candidates, int(diagnostics.dynamic_particle_candidate_count))
            max_particle_contacts = max(max_particle_contacts, int(diagnostics.dynamic_particle_contacts))
            max_particle_overflow = max(max_particle_overflow, int(diagnostics.dynamic_particle_overflow))
            max_dynamic_triangles = max(max_dynamic_triangles, int(diagnostics.dynamic_triangle_count))
            max_resolved_contacts = max(max_resolved_contacts, int(diagnostics.resolved_contacts))
            for slot in session.slots.values():
                finite = finite and _finite_positions(np.asarray(slot.current_positions_world, dtype=np.float64))
            if not finite:
                break
        diagnostics = ssbl.solver.session_diagnostics(cloth)
        stopped = ssbl.solver.request_stop(cloth)
        result = {
            "slots": len(session.slots),
            "cross_mode": str(session.cross_cloth_mode),
            "expect_large_grid": bool(EXPECT_LARGE_GRID),
            "frames": FRAME_COUNT,
            "finite": bool(finite),
            "grid_subdivisions": int(GRID_SUBDIVISIONS),
            "max_dynamic_particle_count": int(max_dynamic_particles),
            "max_dynamic_particle_candidate_count": int(max_particle_candidates),
            "max_dynamic_particle_contacts": int(max_particle_contacts),
            "max_dynamic_particle_overflow": int(max_particle_overflow),
            "max_dynamic_triangle_count": int(max_dynamic_triangles),
            "max_resolved_contacts": int(max_resolved_contacts),
            "dynamic_particle_collision_ms": float(diagnostics.dynamic_particle_collision_ms),
            "dynamic_collision_ms": float(diagnostics.dynamic_collision_ms),
            "stopped": bool(stopped),
        }
        print("SSBL_MULTICLOTH_PARTICLE_CONTACT_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            result["slots"] == 2
            and result["cross_mode"] == "all_selected"
            and result["finite"]
            and result["max_dynamic_particle_count"] > 0
            and result["max_dynamic_particle_candidate_count"] > 0
            and result["max_dynamic_particle_contacts"] > 0
            and result["max_dynamic_particle_overflow"] == 0
            and result["max_dynamic_triangle_count"] > 0
            and result["max_resolved_contacts"] > 0
            and result["stopped"]
        ):
            raise RuntimeError(f"Multi-cloth particle contact smoke failed: {result}")
        if EXPECT_LARGE_GRID and not (
            result["max_dynamic_particle_count"] > 8192
            and result["max_dynamic_triangle_count"] > 4096
            and result["max_dynamic_particle_candidate_count"] > 0
            and result["max_dynamic_particle_contacts"] > 0
            and result["max_resolved_contacts"] > 0
            and result["finite"]
        ):
            raise RuntimeError(f"Large multi-cloth particle contact smoke failed: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
