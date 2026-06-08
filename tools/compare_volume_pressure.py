import json
import math
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


PRESSURE_COORD_LIMIT = 25.0
PRESSURE_DISPLACEMENT_LIMIT = 25.0
PRESSURE_STEP_DELTA_LIMIT = 3.0


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


def _positions(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [
        (float(vertex.co.x), float(vertex.co.y), float(vertex.co.z))
        for vertex in obj.data.vertices
    ]


def _bounds(points: list[tuple[float, float, float]]) -> dict[str, list[float]]:
    if not points:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "min": [min(point[axis] for point in points) for axis in range(3)],
        "max": [max(point[axis] for point in points) for axis in range(3)],
    }


def _max_delta(
    before: list[tuple[float, float, float]],
    after: list[tuple[float, float, float]],
) -> float:
    if len(before) != len(after):
        return float("inf")
    return max(
        (
            max(abs(a - b) for a, b in zip(old, new))
            for old, new in zip(before, after)
        ),
        default=0.0,
    )


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

    initial_positions = _positions(obj)
    previous_positions = initial_positions
    max_step_delta = 0.0
    max_displacement = 0.0
    max_abs_coord = 0.0
    step_diagnostics: list[dict[str, object]] = []
    ssbl.solver.start_preview(bpy.context, obj)
    for _index in range(12):
        ssbl.solver.step_preview(bpy.context, obj.name)
        diag = ssbl.solver.session_diagnostics(obj)
        current_positions = _positions(obj)
        max_step_delta = max(max_step_delta, _max_delta(previous_positions, current_positions))
        max_displacement = max(max_displacement, _max_delta(initial_positions, current_positions))
        max_abs_coord = max(
            max_abs_coord,
            max((abs(component) for point in current_positions for component in point), default=0.0),
        )
        finite = bool(_finite(obj) and diag.finite)
        step_diagnostics.append(
            {
                "step": _index + 1,
                "finite": finite,
                "average_z": _average_axis(obj, "z"),
                "max_step_delta": max_step_delta,
                "max_displacement": max_displacement,
                "max_abs_coord": max_abs_coord,
                "volume_ms": float(diag.volume_ms),
            }
        )
        if not finite:
            raise RuntimeError(f"Overpressure case {enabled=} produced non-finite output at step {_index + 1}: {step_diagnostics[-1]}")
        previous_positions = current_positions
    final_positions = _positions(obj)
    diag = ssbl.solver.session_diagnostics(obj)
    result = {
        "enabled": bool(enabled),
        "grid_size": int(grid_size),
        "density": float(density),
        "finite": bool(_finite(obj) and diag.finite),
        "average_z": _average_axis(obj, "z"),
        "bounds": _bounds(final_positions),
        "max_abs_coord": max_abs_coord,
        "max_displacement": max_displacement,
        "max_step_delta": max_step_delta,
        "stable": bool(
            max_abs_coord <= PRESSURE_COORD_LIMIT
            and max_displacement <= PRESSURE_DISPLACEMENT_LIMIT
            and max_step_delta <= PRESSURE_STEP_DELTA_LIMIT
        ),
        "last_step": step_diagnostics[-1] if step_diagnostics else {},
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
        stress = _run_case(True, grid_size=8, pressure_strength=0.18)
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
            "stress": stress,
        }
        print("SSBL_OVERPRESSURE_COMPARE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            off["finite"]
            and on["finite"]
            and dense_mesh["finite"]
            and heavy["finite"]
            and stress["finite"]
            and off["stable"]
            and on["stable"]
            and dense_mesh["stable"]
            and heavy["stable"]
            and stress["stable"]
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
