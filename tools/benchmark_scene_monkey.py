import math
import os
import sys
import time

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _find_target():
    obj = bpy.data.objects.get("Suzanne")
    if obj is not None and obj.type == "MESH":
        return obj
    active = bpy.context.active_object
    if active is not None and active.type == "MESH":
        return active
    raise RuntimeError("Suzanne mesh not found")


def _ensure_pin_group(obj, settings):
    group_name = str(settings.pin_vertex_group or "ssbl_pin")
    settings.pin_vertex_group = group_name
    if obj.vertex_groups.get(group_name) is not None:
        return
    z_values = [vert.co.z for vert in obj.data.vertices]
    threshold = max(z_values) - (max(z_values) - min(z_values)) * 0.18
    indices = [vert.index for vert in obj.data.vertices if vert.co.z >= threshold]
    if not indices:
        indices = [max(obj.data.vertices, key=lambda vert: vert.co.z).index]
    group = obj.vertex_groups.new(name=group_name)
    group.add(indices, 1.0, "ADD")


def _all_finite(obj):
    return all(
        math.isfinite(float(component))
        for vert in obj.data.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    )


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    scene = bpy.context.scene
    obj = _find_target()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    settings = scene.ssbl_preview
    settings.use_ground = _bool_env("SSBL_BENCH_USE_GROUND", False)
    self_mode = os.environ.get("SSBL_BENCH_SELF_COLLISION_MODE", "fast")
    if self_mode == "quality":
        self_mode = "fast"
    settings.self_collision_mode = self_mode
    settings.self_collision_interval = _int_env("SSBL_BENCH_SELF_COLLISION_INTERVAL", 2)
    settings.max_self_collision_neighbors = _int_env("SSBL_BENCH_MAX_SELF_NEIGHBORS", 32)
    settings.collision_margin = _float_env("SSBL_BENCH_COLLISION_MARGIN", 0.005)
    settings.substeps = _int_env("SSBL_BENCH_SUBSTEPS", 16)
    settings.iterations = _int_env("SSBL_BENCH_ITERATIONS", 2)
    settings.use_volume_pressure = _bool_env("SSBL_BENCH_VOLUME_PRESSURE", False)
    settings.volume_compliance = _float_env("SSBL_BENCH_VOLUME_COMPLIANCE", 1.0e-6)
    settings.stretch_compliance = _float_env("SSBL_BENCH_STRETCH_COMPLIANCE", 0.0)
    settings.bend_compliance = _float_env("SSBL_BENCH_BEND_COMPLIANCE", 0.0)
    _ensure_pin_group(obj, settings)

    source_mesh = obj.data
    source_coords_before = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    steps = _int_env("SSBL_BENCH_STEPS", 120)
    ssbl.solver.start_preview(bpy.context, obj)
    start = time.perf_counter()
    finite = True
    for _index in range(steps):
        ssbl.solver.step_preview(bpy.context, obj.name)
        finite = finite and _all_finite(obj)
    elapsed = max(time.perf_counter() - start, 1.0e-6)
    ssbl.solver.request_stop(obj)
    source_coords_after = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    original_mesh_max_abs_delta = max(
        (abs(float(after) - float(before)) for before, after in zip(source_coords_before, source_coords_after)),
        default=0.0,
    )
    print(
        "SSBL_MONKEY_BENCH",
        f"object={obj.name}",
        f"verts={len(obj.data.vertices)}",
        f"steps={steps}",
        f"substeps={int(settings.substeps)}",
        f"iterations={int(settings.iterations)}",
        f"fps={steps / elapsed:.2f}",
        f"finite={finite}",
        f"original_mesh_max_abs_delta={original_mesh_max_abs_delta:.6g}",
    )
    ssbl.unregister()


if __name__ == "__main__":
    main()
