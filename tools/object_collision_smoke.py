import json
import math
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


MARGIN = 0.05
TOLERANCE = 0.012


def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        if collection.name.startswith("SSBL_Object_Collision"):
            bpy.data.collections.remove(collection)


def _make_grid(name, size=0.32, segments=8, location=(0.0, 0.0, 0.0)):
    verts = []
    faces = []
    half = size * 0.5
    for y in range(segments + 1):
        fy = -half + size * (y / segments)
        for x in range(segments + 1):
            fx = -half + size * (x / segments)
            verts.append((fx, fy, 0.0))
    stride = segments + 1
    for y in range(segments):
        for x in range(segments):
            a = y * stride + x
            b = a + 1
            c = a + stride + 1
            d = a + stride
            faces.append((a, b, c, d))

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _make_static_plane_collection(plane_z):
    collection = bpy.data.collections.new("SSBL_Object_Collision_Static")
    bpy.context.scene.collection.children.link(collection)
    mesh = bpy.data.meshes.new("SSBL_Object_Collision_StaticPlane_Mesh")
    mesh.from_pydata(
        [(-0.55, -0.55, 0.0), (0.55, -0.55, 0.0), (0.55, 0.55, 0.0), (-0.55, 0.55, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new("SSBL_Object_Collision_StaticPlane", mesh)
    obj.location.z = plane_z
    collection.objects.link(obj)
    return collection


def _make_static_grid_collection(name, plane_z, size=1.2, segments=50):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    collider = _make_grid(
        f"{name}_Plane",
        size=size,
        segments=segments,
        location=(0.0, 0.0, plane_z),
    )
    bpy.context.scene.collection.objects.unlink(collider)
    collection.objects.link(collider)
    return collection, segments * segments * 2


def _make_static_object_collection(name, obj):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    if obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(obj)
    collection.objects.link(obj)
    return collection


def _make_monkey(size, location):
    if not hasattr(bpy.ops.mesh, "primitive_monkey_add"):
        raise RuntimeError("This Blender build does not expose primitive_monkey_add")
    bpy.ops.mesh.primitive_monkey_add(size=size, location=location)
    monkey = bpy.context.object
    monkey.name = "SSBL_Object_Collision_StaticMonkey"
    monkey.data.name = "SSBL_Object_Collision_StaticMonkeyMesh"
    return monkey


def _object_bbox_world(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def _mesh_bvh_world(obj):
    vertices = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    polygons = [tuple(poly.vertices) for poly in obj.data.polygons]
    return BVHTree.FromPolygons(vertices, polygons)


def _min_distance_to_object_surface(points, collider):
    bvh = _mesh_bvh_world(collider)
    min_distance = float("inf")
    for point in points:
        nearest = bvh.find_nearest(point)
        if nearest is None:
            continue
        _position, _normal, _index, distance = nearest
        min_distance = min(min_distance, float(distance))
    return min_distance


def _world_positions(obj):
    return [obj.matrix_world @ vert.co for vert in obj.data.vertices]


def _flat_mesh_coords(mesh):
    return [float(component) for vert in mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]


def _all_finite(positions):
    return all(math.isfinite(float(component)) for point in positions for component in (point.x, point.y, point.z))


def _configure_common(settings):
    settings.use_evaluated_mesh = True
    settings.pin_vertex_group = ""
    settings.frame_count = 16
    settings.preview_target_fps = 60.0
    settings.dt = 1.0 / 60.0
    settings.substeps = 4
    settings.iterations = 1
    settings.damping = 1.0
    settings.gravity = (0.0, 0.0, 0.0)
    settings.stretch_compliance = 1e-6
    settings.bend_compliance = 1e-4
    settings.self_collision = False
    settings.use_volume_pressure = False
    settings.collision_margin = MARGIN
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.sphere_object = None
    settings.static_collider_collection = None
    settings.static_sdf_voxel_size = 0.04
    settings.static_sdf_band_voxels = 3
    settings.static_sdf_max_resolution = 80


def _static_sdf_summary(obj):
    diag = ssbl.solver.session_diagnostics(obj)
    return {
        "static_triangle_count": int(diag.static_triangle_count),
        "static_sdf_rebuild_count": int(diag.static_sdf_rebuild_count),
        "static_sdf_voxel_count": int(diag.static_sdf_voxel_count),
        "static_sdf_grid": [
            int(diag.static_sdf_grid_x),
            int(diag.static_sdf_grid_y),
            int(diag.static_sdf_grid_z),
        ],
        "static_sdf_build_ms": float(diag.static_sdf_build_ms),
        "static_sdf_contact_count": int(diag.static_sdf_contact_count),
        "static_sdf_unsigned_fallback_count": int(diag.static_sdf_unsigned_fallback_count),
        "resolved_contacts": int(diag.resolved_contacts),
        "finite": bool(diag.finite),
    }


def _assert_static_sdf_diagnostics(label, diag, min_triangles=1):
    if not diag["finite"]:
        raise RuntimeError(f"{label} native diagnostics reported non-finite output")
    if diag["static_triangle_count"] < min_triangles:
        raise RuntimeError(f"{label} static triangle count {diag['static_triangle_count']} below expected {min_triangles}")
    if diag["static_sdf_rebuild_count"] <= 0 or diag["static_sdf_voxel_count"] <= 0:
        raise RuntimeError(f"{label} did not report a built static SDF: {diag}")
    if any(value <= 1 for value in diag["static_sdf_grid"]):
        raise RuntimeError(f"{label} reported invalid static SDF grid: {diag}")
    if diag["static_sdf_contact_count"] <= 0:
        raise RuntimeError(f"{label} did not report SDF contacts: {diag}")


def _run_case(label, build_scene, metric, expect_static_sdf=False, min_static_triangles=1):
    _clear_scene()
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.frame_start = 1
    scene.frame_end = 120

    settings = scene.ssbl_preview
    _configure_common(settings)
    cloth = build_scene(settings)
    bpy.context.view_layer.objects.active = cloth
    cloth.select_set(True)
    bpy.context.view_layer.update()

    original_mesh = cloth.data
    original_before = _flat_mesh_coords(original_mesh)
    initial_metric = metric(cloth, enforce=False)

    session = ssbl.solver.start_preview(bpy.context, cloth)
    final_metric = None
    max_sdf_diag = None
    try:
        for _index in range(4):
            ssbl.solver.step_preview(bpy.context, cloth.name)
            positions = _world_positions(cloth)
            if not _all_finite(positions):
                raise RuntimeError(f"{label} produced non-finite vertex coordinates")
            final_metric = metric(cloth, enforce=True)
            if expect_static_sdf:
                sdf_diag = _static_sdf_summary(cloth)
                if max_sdf_diag is None or sdf_diag["static_sdf_contact_count"] > max_sdf_diag["static_sdf_contact_count"]:
                    max_sdf_diag = sdf_diag
    finally:
        ssbl.solver.request_stop(cloth)

    original_after = _flat_mesh_coords(original_mesh)
    original_delta = max(
        (abs(after - before) for before, after in zip(original_before, original_after)),
        default=0.0,
    )
    if original_delta > 1e-7:
        raise RuntimeError(f"{label} polluted the source mesh: max delta {original_delta}")
    if final_metric is None:
        raise RuntimeError(f"{label} did not step")
    if expect_static_sdf:
        if max_sdf_diag is None:
            raise RuntimeError(f"{label} did not capture static SDF diagnostics")
        _assert_static_sdf_diagnostics(label, max_sdf_diag, min_static_triangles)

    result = {
        "session_object": session.object_name,
        "initial": initial_metric,
        "final": final_metric,
        "original_mesh_max_abs_delta": original_delta,
    }
    if max_sdf_diag is not None:
        result["static_sdf"] = max_sdf_diag
    return result


def _ground_case(settings):
    settings.use_ground = True
    settings.ground_height = 0.0
    return _make_grid("SSBL_Object_Collision_GroundCloth", location=(0.0, 0.0, MARGIN * 0.5))


def _ground_metric(obj, enforce=True):
    min_z = min(point.z for point in _world_positions(obj))
    limit = MARGIN
    penetration = max(0.0, limit - min_z)
    if enforce and penetration > TOLERANCE:
        raise RuntimeError(f"ground penetration {penetration:.6f} exceeds tolerance")
    return {"min_z": min_z, "limit": limit, "penetration": penetration}


def _wall_case(settings):
    settings.use_wall = True
    settings.wall_origin = (0.0, 0.0, 0.0)
    settings.wall_normal = (1.0, 0.0, 0.0)
    return _make_grid("SSBL_Object_Collision_WallCloth", location=(MARGIN * 0.5, 0.0, 0.0))


def _wall_metric(obj, enforce=True):
    origin = Vector((0.0, 0.0, 0.0))
    normal = Vector((1.0, 0.0, 0.0))
    min_signed = min((point - origin).dot(normal) for point in _world_positions(obj))
    penetration = max(0.0, MARGIN - min_signed)
    if enforce and penetration > TOLERANCE:
        raise RuntimeError(f"wall penetration {penetration:.6f} exceeds tolerance")
    return {"min_signed_distance": min_signed, "limit": MARGIN, "penetration": penetration}


def _static_monkey_case(settings):
    monkey = _make_monkey(0.55, (0.0, 0.0, 0.0))
    bpy.context.view_layer.update()
    top_z = max(point.z for point in _object_bbox_world(monkey))
    settings.static_collider_collection = _make_static_object_collection("SSBL_Object_Collision_StaticMonkeyCollection", monkey)
    return _make_grid("SSBL_Object_Collision_StaticMonkeyCloth", size=0.26, location=(0.0, 0.0, top_z + MARGIN * 0.45))


def _static_monkey_metric(obj, enforce=True):
    monkey = bpy.data.objects["SSBL_Object_Collision_StaticMonkey"]
    min_distance = _min_distance_to_object_surface(_world_positions(obj), monkey)
    penetration = max(0.0, MARGIN - min_distance)
    if enforce and penetration > TOLERANCE:
        raise RuntimeError(f"static monkey penetration {penetration:.6f} exceeds tolerance")
    return {"min_surface_distance": min_distance, "limit": MARGIN, "penetration": penetration}


def _static_mesh_case(settings):
    plane_z = 0.0
    settings.static_collider_collection = _make_static_plane_collection(plane_z)
    return _make_grid("SSBL_Object_Collision_StaticMeshCloth", location=(0.0, 0.0, plane_z + MARGIN * 0.5))


def _large_static_mesh_case(settings):
    plane_z = 0.0
    collection, triangle_count = _make_static_grid_collection(
        "SSBL_Object_Collision_LargeStatic",
        plane_z,
        size=1.2,
        segments=50,
    )
    settings.static_collider_collection = collection
    return _make_grid("SSBL_Object_Collision_LargeStaticCloth", location=(0.0, 0.0, plane_z + MARGIN * 0.5))


def _static_mesh_metric(obj, enforce=True):
    limit = MARGIN
    min_z = min(point.z for point in _world_positions(obj))
    penetration = max(0.0, limit - min_z)
    if enforce and penetration > TOLERANCE:
        raise RuntimeError(f"static mesh penetration {penetration:.6f} exceeds tolerance")
    return {"min_z": min_z, "limit": limit, "penetration": penetration}


def _static_plane_metric(obj, plane_z, enforce=True):
    limit = float(plane_z) + MARGIN
    min_z = min(point.z for point in _world_positions(obj))
    penetration = max(0.0, limit - min_z)
    if enforce and penetration > TOLERANCE:
        raise RuntimeError(f"moving static mesh penetration {penetration:.6f} exceeds tolerance")
    return {"min_z": min_z, "limit": limit, "penetration": penetration}


def _run_moving_static_collider_case():
    label = "moving_static_mesh"
    _clear_scene()
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.frame_start = 1
    scene.frame_end = 120

    settings = scene.ssbl_preview
    _configure_common(settings)
    settings.static_collider_collection = _make_static_plane_collection(0.0)
    collider = bpy.data.objects["SSBL_Object_Collision_StaticPlane"]
    cloth = _make_grid("SSBL_Object_Collision_MovingStaticCloth", location=(0.0, 0.0, MARGIN * 0.5))
    bpy.context.view_layer.objects.active = cloth
    cloth.select_set(True)
    bpy.context.view_layer.update()

    original_mesh = cloth.data
    original_before = _flat_mesh_coords(original_mesh)
    session = ssbl.solver.start_preview(bpy.context, cloth)
    try:
        ssbl.solver.step_preview(bpy.context, cloth.name)
        first_metric = _static_plane_metric(cloth, collider.location.z, enforce=True)
        first_diag = _static_sdf_summary(cloth)
        _assert_static_sdf_diagnostics(label, first_diag)

        collider.location.z = MARGIN * 0.6
        bpy.context.view_layer.update()
        ssbl.solver.step_preview(bpy.context, cloth.name)
        second_metric = _static_plane_metric(cloth, collider.location.z, enforce=True)
        second_diag = _static_sdf_summary(cloth)
        _assert_static_sdf_diagnostics(label, second_diag)
    finally:
        ssbl.solver.request_stop(cloth)

    original_after = _flat_mesh_coords(original_mesh)
    original_delta = max(
        (abs(after - before) for before, after in zip(original_before, original_after)),
        default=0.0,
    )
    if original_delta > 1e-7:
        raise RuntimeError(f"{label} polluted the source mesh: max delta {original_delta}")
    if second_diag["static_sdf_rebuild_count"] <= first_diag["static_sdf_rebuild_count"]:
        raise RuntimeError(f"{label} did not rebuild SDF after collider motion: {first_diag} -> {second_diag}")
    if second_metric["min_z"] <= first_metric["min_z"] + MARGIN * 0.3:
        raise RuntimeError(f"{label} cloth response did not follow moved collider: {first_metric} -> {second_metric}")

    return {
        "session_object": session.object_name,
        "first": first_metric,
        "second": second_metric,
        "first_static_sdf": first_diag,
        "second_static_sdf": second_diag,
        "original_mesh_max_abs_delta": original_delta,
    }


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        results = {
            "ground": _run_case("ground", _ground_case, _ground_metric),
            "wall": _run_case("wall", _wall_case, _wall_metric),
            "static_monkey": _run_case(
                "static_monkey",
                _static_monkey_case,
                _static_monkey_metric,
                expect_static_sdf=True,
            ),
            "static_mesh": _run_case(
                "static_mesh",
                _static_mesh_case,
                _static_mesh_metric,
                expect_static_sdf=True,
            ),
            "large_static_mesh": _run_case(
                "large_static_mesh",
                _large_static_mesh_case,
                _static_mesh_metric,
                expect_static_sdf=True,
                min_static_triangles=4097,
            ),
            "moving_static_mesh": _run_moving_static_collider_case(),
        }
        print("SSBL_OBJECT_COLLISION_SMOKE", json.dumps(results, ensure_ascii=False))
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
