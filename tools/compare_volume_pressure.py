import json
import math
import os
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from ssbl import xpbd_core


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


def _world_positions(obj):
    return [obj.matrix_world @ vert.co for vert in obj.data.vertices]


def _signed_volume(obj, triangles):
    points = _world_positions(obj)
    total = 0.0
    for a_i, b_i, c_i in triangles:
        a = points[int(a_i)]
        b = points[int(b_i)]
        c = points[int(c_i)]
        total += a.dot(b.cross(c)) / 6.0
    return total


def _bbox(obj):
    points = _world_positions(obj)
    mins = [min(point[index] for point in points) for index in range(3)]
    maxs = [max(point[index] for point in points) for index in range(3)]
    return {
        "min": mins,
        "max": maxs,
        "size": [maxs[index] - mins[index] for index in range(3)],
    }


def _finite(obj):
    return all(
        math.isfinite(float(component))
        for vert in obj.data.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    )


def _int_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


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
    settings.frame_count = 120
    settings.use_ground = False
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 2
    settings.max_self_collision_neighbors = 32
    settings.substeps = _int_env("SSBL_COMPARE_SUBSTEPS", int(settings.substeps))
    settings.iterations = _int_env("SSBL_COMPARE_ITERATIONS", int(settings.iterations))
    _ensure_pin_group(obj, settings)

    triangles = xpbd_core.triangulated_faces(obj.data)
    rest_volume = _signed_volume(obj, triangles)
    results = []

    for enabled in (False, True):
        settings.use_volume_pressure = enabled
        settings.volume_compliance = 1e-6
        settings.pressure_strength = 1.0
        settings.volume_target_scale = 1.0
        ssbl.solver.start_preview(bpy.context, obj)
        for _index in range(60):
            ssbl.solver.step_preview(bpy.context, obj.name)
        volume = _signed_volume(obj, triangles)
        results.append(
            {
                "volume_pressure": enabled,
                "finite": _finite(obj),
                "volume": volume,
                "volume_ratio": volume / rest_volume if abs(rest_volume) > 1e-12 else None,
                "bbox": _bbox(obj),
            }
        )
        ssbl.solver.request_stop(obj)

    print(
        "SSBL_VOLUME_COMPARE",
        json.dumps(
            {
                "rest_volume": rest_volume,
                "substeps": int(settings.substeps),
                "iterations": int(settings.iterations),
                "results": results,
            },
            ensure_ascii=False,
        ),
    )
    ssbl.unregister()


if __name__ == "__main__":
    main()
