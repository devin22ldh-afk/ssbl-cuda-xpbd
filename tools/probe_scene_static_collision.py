import json
import math
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)


def _world_positions(obj):
    return [obj.matrix_world @ vert.co for vert in obj.data.vertices]


def _bvh_for_object(obj):
    verts = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    faces = [tuple(poly.vertices) for poly in obj.data.polygons]
    return BVHTree.FromPolygons(verts, faces)


def _distance_stats(points, collider):
    bvh = _bvh_for_object(collider)
    distances = []
    for point in points:
        nearest = bvh.find_nearest(point)
        if nearest is None:
            continue
        _pos, _normal, _index, distance = nearest
        distances.append(float(distance))
    distances.sort()
    if not distances:
        return {"count": 0}
    return {
        "count": len(distances),
        "min": distances[0],
        "p01": distances[min(max(int(len(distances) * 0.01), 0), len(distances) - 1)],
        "p05": distances[min(max(int(len(distances) * 0.05), 0), len(distances) - 1)],
        "p50": distances[len(distances) // 2],
    }


def _position_stats(points):
    if not points:
        return {"count": 0}
    return {
        "count": len(points),
        "min": [min(point[axis] for point in points) for axis in range(3)],
        "max": [max(point[axis] for point in points) for axis in range(3)],
    }


def _bbox(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return {
        "min": [min(point[axis] for point in points) for axis in range(3)],
        "max": [max(point[axis] for point in points) for axis in range(3)],
    }


def _main():
    import ssbl

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    scene = bpy.context.scene
    obj = bpy.context.active_object
    settings = scene.ssbl_preview
    collection = settings.static_collider_collection
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Active object must be the cloth mesh")
    if collection is None:
        raise RuntimeError("Static Collider Collection is not assigned")
    colliders = [item for item in collection.objects if item.type == "MESH" and item != obj]
    if not colliders:
        raise RuntimeError("Static Collider Collection has no direct mesh collider")
    collider = colliders[0]

    source_mesh = obj.data
    source_coords = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    samples = []
    try:
        session = ssbl.solver.start_preview(bpy.context, obj)
        for frame in range(1, min(int(settings.frame_count), 160) + 1):
            ssbl.solver.step_preview(bpy.context, obj.name)
            if frame <= 10 or frame % 10 == 0:
                positions = _world_positions(obj)
                finite = all(
                    math.isfinite(float(component))
                    for point in positions
                    for component in (point.x, point.y, point.z)
                )
                samples.append(
                    {
                        "frame": frame,
                        "finite": finite,
                        "cloth_bbox": _bbox(obj),
                        "cloth_positions": _position_stats(positions),
                        "distance_to_collider": _distance_stats(positions, collider),
                    }
                )
    finally:
        ssbl.solver.request_stop(obj)

    source_after = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    original_delta = max(
        (abs(float(after) - float(before)) for before, after in zip(source_coords, source_after)),
        default=0.0,
    )
    print(
        "SSBL_SCENE_STATIC_COLLISION_PROBE "
        + json.dumps(
            {
                "blend_file": bpy.data.filepath,
                "cloth": obj.name,
                "collider": collider.name,
                "settings": {
                    "frame_count": int(settings.frame_count),
                    "substeps": int(settings.substeps),
                    "iterations": int(settings.iterations),
                    "dt": float(settings.dt),
                    "collision_margin": float(settings.collision_margin),
                },
                "source_mesh_max_abs_delta": original_delta,
                "collider_bbox": _bbox(collider),
                "samples": samples,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    _main()
