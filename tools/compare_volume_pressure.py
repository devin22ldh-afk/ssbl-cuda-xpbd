import json
import math
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _make_open_grid(name: str, size: int = 4) -> bpy.types.Object:
    verts = []
    faces = []
    for y in range(size + 1):
        for x in range(size + 1):
            verts.append((float(x) / float(size), float(y) / float(size), 0.0))
    for y in range(size):
        for x in range(size):
            a = y * (size + 1) + x
            b = a + 1
            d = a + (size + 1)
            c = d + 1
            faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def _average_axis(obj: bpy.types.Object, axis_name: str) -> float:
    return sum(float(getattr(vertex.co, axis_name)) for vertex in obj.data.vertices) / max(len(obj.data.vertices), 1)


def _finite(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _run_case(
    enabled: bool,
    *,
    grid_size: int = 4,
    density: float = 1.0,
    pressure_strength: float = 0.08,
) -> dict[str, object]:
    _clear_scene()
    obj = _make_open_grid("SSBL_Overpressure_On" if enabled else "SSBL_Overpressure_Off", grid_size)
    settings = bpy.context.scene.ssbl_preview
    settings.frame_count = 30
    settings.density = float(density)
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = ""
    settings.use_ground = False
    settings.self_collision = False
    settings.gravity = (0.0, 0.0, 0.0)
    settings.damping = 1.0
    settings.substeps = 2
    settings.iterations = 1
    settings.preview_writeback_interval = 1
    settings.use_volume_pressure = bool(enabled)
    settings.pressure_strength = float(pressure_strength) if enabled else 0.0

    ssbl.solver.start_preview(bpy.context, obj)
    for _index in range(12):
        ssbl.solver.step_preview(bpy.context, obj.name)
    diag = ssbl.solver.session_diagnostics(obj)
    result = {
        "enabled": bool(enabled),
        "grid_size": int(grid_size),
        "density": float(density),
        "finite": bool(_finite(obj) and diag.finite),
        "average_z": _average_axis(obj, "z"),
        "force_field_count": int(diag.force_field_count),
    }
    ssbl.solver.request_stop(obj)
    return result


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        off = _run_case(False)
        on = _run_case(True)
        dense_mesh = _run_case(True, grid_size=16)
        heavy = _run_case(True, density=2.0)
        base_delta = float(on["average_z"]) - float(off["average_z"])
        dense_mesh_delta = float(dense_mesh["average_z"]) - float(off["average_z"])
        heavy_delta = float(heavy["average_z"]) - float(off["average_z"])
        result = {
            "off": off,
            "on": on,
            "average_z_delta": base_delta,
            "mesh_adaptive": {
                "base_delta": base_delta,
                "dense_mesh_delta": dense_mesh_delta,
                "dense_mesh_ratio": dense_mesh_delta / max(base_delta, 1.0e-8),
            },
            "density_adaptive": {
                "density_1_delta": base_delta,
                "density_2_delta": heavy_delta,
                "density_2_ratio": heavy_delta / max(base_delta, 1.0e-8),
            },
        }
        print("SSBL_OVERPRESSURE_COMPARE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            off["finite"]
            and on["finite"]
            and dense_mesh["finite"]
            and heavy["finite"]
            and abs(float(off["average_z"])) < 1.0e-4
            and base_delta > 0.01
            and 0.85 <= result["mesh_adaptive"]["dense_mesh_ratio"] <= 1.15
            and 0.40 <= result["density_adaptive"]["density_2_ratio"] <= 0.60
        ):
            raise RuntimeError(f"Overpressure comparison failed: {result}")
    finally:
        try:
            ssbl.solver.cleanup_all_sessions()
        except Exception:
            pass
        ssbl.unregister()


if __name__ == "__main__":
    main()
