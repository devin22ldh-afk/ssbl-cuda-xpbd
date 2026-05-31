import json
import math
import os
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from ssbl import xpbd_core


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


def _segment_triangle_intersects(p0, p1, a, b, c, eps=1.0e-7):
    direction = p1 - p0
    edge1 = b - a
    edge2 = c - a
    h = direction.cross(edge2)
    det = edge1.dot(h)
    if abs(det) < eps:
        return False
    inv_det = 1.0 / det
    s = p0 - a
    u = inv_det * s.dot(h)
    if u < eps or u > 1.0 - eps:
        return False
    q = s.cross(edge1)
    v = inv_det * direction.dot(q)
    if v < eps or u + v > 1.0 - eps:
        return False
    t = inv_det * edge2.dot(q)
    return eps < t < 1.0 - eps


def _triangles_intersect(vertices, tri_a, tri_b):
    a0, a1, a2 = [vertices[index] for index in tri_a]
    b0, b1, b2 = [vertices[index] for index in tri_b]
    for p0, p1 in ((a0, a1), (a1, a2), (a2, a0)):
        if _segment_triangle_intersects(p0, p1, b0, b1, b2):
            return True
    for p0, p1 in ((b0, b1), (b1, b2), (b2, b0)):
        if _segment_triangle_intersects(p0, p1, a0, a1, a2):
            return True
    return False


def _world_vertices(obj):
    return [obj.matrix_world @ vert.co for vert in obj.data.vertices]


def _signed_volume(obj, triangles):
    vertices = _world_vertices(obj)
    total = 0.0
    for ia, ib, ic in triangles:
        a = vertices[int(ia)]
        b = vertices[int(ib)]
        c = vertices[int(ic)]
        total += a.dot(b.cross(c)) / 6.0
    return total


def _bbox(obj):
    vertices = _world_vertices(obj)
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    return {"min": mins, "max": maxs, "size": [maxs[i] - mins[i] for i in range(3)]}


def _check_intersections(obj, triangles, max_pairs):
    vertices = _world_vertices(obj)
    polygons = [tuple(int(v) for v in tri) for tri in triangles]
    tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True, epsilon=0.0)
    overlap_pairs = tree.overlap(tree)
    tested = 0
    intersections = []
    for ia, ib in overlap_pairs:
        if ia >= ib:
            continue
        tri_a = polygons[ia]
        tri_b = polygons[ib]
        if set(tri_a).intersection(tri_b):
            continue
        tested += 1
        if _triangles_intersect(vertices, tri_a, tri_b):
            intersections.append((int(ia), int(ib), tri_a, tri_b))
            if len(intersections) >= max_pairs:
                break
    return {
        "overlap_pairs": len(overlap_pairs),
        "tested_non_adjacent_pairs": tested,
        "intersection_count_capped": len(intersections),
        "sample_intersections": intersections[:10],
    }


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
    settings.frame_count = max(_int_env("SSBL_CHECK_FRAME_COUNT", 120), 61)
    settings.use_ground = _bool_env("SSBL_CHECK_USE_GROUND", False)
    self_mode = os.environ.get("SSBL_CHECK_SELF_COLLISION_MODE", "fast")
    if self_mode == "quality":
        self_mode = "fast"
    settings.self_collision_mode = self_mode
    settings.self_collision_interval = _int_env("SSBL_CHECK_SELF_COLLISION_INTERVAL", 2)
    settings.max_self_collision_neighbors = _int_env("SSBL_CHECK_MAX_SELF_NEIGHBORS", 32)
    settings.collision_margin = float(os.environ.get("SSBL_CHECK_COLLISION_MARGIN", str(settings.collision_margin)))
    settings.substeps = _int_env("SSBL_CHECK_SUBSTEPS", int(settings.substeps))
    settings.iterations = _int_env("SSBL_CHECK_ITERATIONS", int(settings.iterations))
    if "SSBL_CHECK_STRETCH_COMPLIANCE" in os.environ:
        settings.stretch_compliance = float(os.environ["SSBL_CHECK_STRETCH_COMPLIANCE"])
    if "SSBL_CHECK_BEND_COMPLIANCE" in os.environ:
        settings.bend_compliance = float(os.environ["SSBL_CHECK_BEND_COMPLIANCE"])
    settings.use_volume_pressure = _bool_env("SSBL_CHECK_VOLUME_PRESSURE", False)
    settings.volume_compliance = float(os.environ.get("SSBL_CHECK_VOLUME_COMPLIANCE", "0.000001"))
    settings.pressure_strength = float(os.environ.get("SSBL_CHECK_PRESSURE_STRENGTH", "1.0"))
    settings.volume_target_scale = float(os.environ.get("SSBL_CHECK_VOLUME_TARGET_SCALE", "1.0"))
    _ensure_pin_group(obj, settings)

    rest_triangles = xpbd_core.triangulated_faces(obj.data)
    rest_volume = _signed_volume(obj, rest_triangles)
    source_mesh = obj.data
    source_coords_before = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]

    ssbl.solver.start_preview(bpy.context, obj)
    steps = _int_env("SSBL_CHECK_STEPS", 60)
    for _index in range(steps):
        ssbl.solver.step_preview(bpy.context, obj.name)

    triangles = xpbd_core.triangulated_faces(obj.data)
    volume = _signed_volume(obj, triangles)
    coords_finite = all(
        math.isfinite(float(component))
        for vert in obj.data.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    )
    collision = _check_intersections(obj, triangles, _int_env("SSBL_CHECK_MAX_REPORT", 1000))
    bbox = _bbox(obj)
    ssbl.solver.request_stop(obj)

    source_coords_after = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    original_mesh_max_abs_delta = max(
        (abs(float(after) - float(before)) for before, after in zip(source_coords_before, source_coords_after)),
        default=0.0,
    )
    summary = {
        "object": obj.name,
        "steps": steps,
        "substeps": int(settings.substeps),
        "iterations": int(settings.iterations),
        "self_collision_mode": settings.self_collision_mode,
        "self_collision_interval": int(settings.self_collision_interval),
        "max_self_collision_neighbors": int(settings.max_self_collision_neighbors),
        "collision_margin": float(settings.collision_margin),
        "volume_pressure": bool(settings.use_volume_pressure),
        "vertex_count": len(obj.data.vertices),
        "triangle_count": len(triangles),
        "finite": coords_finite,
        "rest_volume": rest_volume,
        "volume": volume,
        "volume_ratio": volume / rest_volume if abs(rest_volume) > 1.0e-12 else None,
        "bbox": bbox,
        "original_mesh_max_abs_delta": original_mesh_max_abs_delta,
        **collision,
    }
    print("SSBL_SELF_INTERSECTION_CHECK", json.dumps(summary, ensure_ascii=False))
    ssbl.unregister()


if __name__ == "__main__":
    main()
